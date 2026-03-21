import { callMcpTool } from "./mcpClient.js";

export function createHydrateNode({ loadMessagesFromRedis, RemoveMessage, REMOVE_ALL_MESSAGES }) {
  return async (state) => {
    try {
      const UserID = state?.UserID ?? "default";
      const ContextID = state?.ContextID ?? "default";

 

      const restored = await loadMessagesFromRedis(UserID, ContextID);

      console.log("hydrate state:", state);
      console.log("hydrate restored:", restored);

      if (!Array.isArray(restored) || restored.length === 0) {
        return {};
      }


      return {
        messages: [new RemoveMessage({ id: REMOVE_ALL_MESSAGES }), ...restored],
      };

  
    } catch (e) {
      console.warn("⚠️ Redis hydrate failed:", e?.message ?? e);
      return {};
    }
  };
}

export function createIngestNode({ HumanMessage }) {
  return (state) => {
    // ingest 节点的输入优先级：
    // 1) routerQuestion：由上游 route_input 节点产出的“路由后问题”
    // 2) input：原始用户问题
    // 3) 空字符串兜底，避免出现 undefined/null 传入 HumanMessage
    //
    // 这样可以保证“问题先经过 RouterModel 再进入主链路”：
    // - 如果 Router 成功改写问题，这里就会消费改写结果；
    // - 如果 Router 失败或未产出有效内容，这里会自动回退到原问题。
    const nextInput = String(state?.routerQuestion ?? state?.input ?? "");
    return {
      // 将最终采用的问题写回 state.input，确保后续节点（decide/rag）读取到一致输入
      input: nextInput,
      // 以最终问题构造 HumanMessage，进入消息状态流
      messages: [new HumanMessage(nextInput)],
    };
  };
}

export function createRouteInputNode({
  createRouterModel,
  safeParseJsonObject,
  SystemMessage,
  HumanMessage,
  options,
}) {
  return async (state) => {
    const input = String(state?.input ?? "");
    const routerModel = options?.routerModel;
    if (!routerModel || !String(routerModel).trim()) {
      return {
        routerDecision: "router_model_not_configured",
        routerQuestion: input,
      };
    }
    try {
      const llm = createRouterModel(options);
      const system = [
        "你是问题改写器。",
        "我将给你一个问题和一个历史问题的列表，你来确认这个问题是否具有相似的表达和意图问题存在于这个列表",
        "如果有，则把问题改写成这个相似问题",
        "如果没有，则保持原问题不变",
        "下面是一个些例子：",
        "原始问题：我们公司一个月最多请假多少天？",
        "历史问题列表：['对细菌内毒素检查有影响的因素有哪些？', '一个月最多请多少天假，超过了会怎样？', '我们这边请假和谁请']",
        "推理过程：原始问题与历史问题列表中的第二个问题有相似的表达和意图，都是在询问公司请假的最大天数，第三个问题是一个陷阱，我们需要比较的是一个月最多请假多少天，而不是和谁请假，虽然问题都是与请假有关，但这是两个不同的主题。",
        "改写后的问题：一个月最多请多少天假，超过了会怎样？",
        "只需要输出改写后的问题，绝对禁止输出其他内容",
      ].join("\n");
      const human = input;
      const res = await llm.invoke([
        new SystemMessage(system),
        new HumanMessage(human),
      ]);
      const parsed = safeParseJsonObject(String(res?.content ?? ""));
      const routedQuestion =
        typeof parsed?.question === "string" && parsed.question.trim()
          ? parsed.question.trim()
          : input;
      const reason =
        typeof parsed?.reason === "string" && parsed.reason.trim()
          ? parsed.reason.trim()
          : "router_ok";
      return {
        input: routedQuestion,
        routerQuestion: routedQuestion,
        routerDecision: reason,
      };
    } catch (e) {
      return {
        routerDecision: `router_failed:${e?.message ?? String(e)}`,
        routerQuestion: input,
      };
    }
  };
}

export function createRagNode({ ragChain, AIMessage, computeQuestionMd5Upper, saveQuestionChunkIndex }) {
  return async (state) => {
    // 如果上一步执行了工具调用，把工具结果拼到本轮问题后面，作为补充上下文输入给 RAG。
    // 目的：让模型在生成最终回答时可以参考工具返回的实时信息。
    const toolHint = state.toolUsedResultSummary
      ? `\n\n工具结果：\n${String(state.toolUsedResultSummary)}`
      : "";
    // 调用检索增强链：内部会先检索文档，再把“问题+上下文”交给主模型生成答案。
    console.log("⏳ 等待 LLM 生成回答...");

const question =
  typeof state.routerQuestion === "string" && state.routerQuestion.trim()
    ? state.routerQuestion
    : state.input;

    const queryText = String(question ?? "");
    const questionMd5 =
      typeof computeQuestionMd5Upper === "function"
        ? computeQuestionMd5Upper(queryText)
        : "";

    const res = await ragChain.invoke({ input: `${question}${toolHint}` });
    console.log("✅ LLM 已返回回答");
    // 兼容不同链返回字段（answer/output），统一转为字符串，避免下游收到非字符串类型。
    const answer = String(res?.answer ?? res?.output ?? "");
    // 检索到的上下文文档列表，写回 state.context 供返回层或调试使用。
    const context = res?.context ?? [];
    try {
      if (questionMd5 && typeof saveQuestionChunkIndex === "function") {
        const chunkIds = (Array.isArray(context) ? context : [])
          .map((doc) => doc?.metadata?.langchain_primaryid ?? doc?.metadata?.pk)
          .filter((id) => id !== null && id !== undefined && String(id).trim() !== "");
        const firstScore = (Array.isArray(context) && context.length > 0)
          ? (context[0]?.score ?? context[0]?.metadata?.score)
          : undefined;
        const score = Number(firstScore);
        await saveQuestionChunkIndex({
          questionMd5,
          chunkIds,
          score: Number.isFinite(score) ? score : undefined,
        });
      }
    } catch (e) {
      console.warn("⚠️ Redis chunk index persist failed:", e?.message ?? e);
    }
    return {
      // 最终回答文本，供 rpc/cli 直接返回给调用方。
      answer,
      // 本轮命中的检索上下文。
      context,
      // 追加一条 AIMessage 到消息状态，用于后续记忆持久化与多轮对话。
      messages: [new AIMessage(answer)],
    };
  };
}

export function createMcpListNode() {
  return async () => {
    const tools = [
      {
        name: "mcp.crm.getCustomerInfoBy",
        description: "Search the web for up-to-date info and return summarized results.",
        inputSchema: { type: "object", properties: { query: { type: "string" } }, required: ["query"] },
      },
      {
        name: "mcp.query_db",
        description: "Query internal database by SQL and return rows.",
        inputSchema: { type: "object", properties: { sql: { type: "string" } }, required: ["sql"] },
      },
    ];
    return { mcpTools: tools };
  };
}

export function createSummarizeNode({
  SystemMessage,
  RemoveMessage,
  REMOVE_ALL_MESSAGES,
  summarizationMiddleware,
  summaryModel,
  estimateRedisValueBytes,
  config,
}) {
  return async (state) => {
    try {
      const messages = Array.isArray(state.messages) ? state.messages : [];
      const maxValueBytes = Number(config.redis?.maxValueBytes ?? 0);
      if (!Number.isFinite(maxValueBytes) || maxValueBytes <= 0) return {};
      const estimatedSize = estimateRedisValueBytes(messages);
      if (estimatedSize <= maxValueBytes) return {};
      const systemMessages = messages.filter((m) => SystemMessage.isInstance(m));
      const nonSystemMessages = messages.filter(
        (m) => !SystemMessage.isInstance(m) && !RemoveMessage.isInstance(m)
      );
      if (nonSystemMessages.length === 0) return {};
      const keepLastN = Math.max(0, Number(config.redis?.summaryKeepLastN ?? 6));
      const middleware = summarizationMiddleware({
        model: summaryModel,
        trigger: { messages: 1 },
        keep: { messages: keepLastN },
        summaryPrefix: config.redis?.summaryPrefix ?? "对话摘要：",
      });
      const res = await middleware.beforeModel(
        { messages: nonSystemMessages },
        { context: {} }
      );
      if (!res?.messages) return {};
      const summarizedMessages = res.messages.filter(
        (m) => !RemoveMessage.isInstance(m)
      );
      if (summarizedMessages.length === 0) return {};
      return {
        messages: [
          new RemoveMessage({ id: REMOVE_ALL_MESSAGES }),
          ...systemMessages,
          ...summarizedMessages,
        ],
      };
    } catch (e) {
      console.warn("⚠️ Summarize failed:", e?.message ?? e);
      return {};
    }
  };
}

export function createPersistNode({ saveMessagesToRedis }) {
  return async (state) => {
    try {
      const threadId = state.threadId ?? "default";
      await saveMessagesToRedis(threadId, state.messages ?? []);
    } catch (e) {
      console.warn("⚠️ Redis persist failed:", e?.message ?? e);
    }
    return {};
  };
}

export function createDecideNode({
  createChatModel,
  normalizeToolList,
  safeParseJsonObject,
  normalizeArgs,
  isToolNameAllowed,
  SystemMessage,
  HumanMessage,
  options,
}) {
  return async (state) => {
    const llm = createChatModel(options);
    const tools = normalizeToolList(state.mcpTools);
    const used = Number(state.toolCallsCount ?? 0);
    const max = Number(state.toolCallsMax ?? 1);
    if (used >= max) {
      return {
        toolPlan: {
          needTool: false,
          toolName: null,
          args: null,
          reason: `已达到工具调用上限（${used}/${max}），将直接回答。`,
        },
        toolUsed: false,
        toolUsedReason: `已达到工具调用上限（${used}/${max}）`,
        toolUsedArgs: null,
      };
    }
    if (tools.length === 0) {
      return {
        toolPlan: {
          needTool: false,
          toolName: null,
          args: null,
          reason: "当前没有可用的 MCP 工具。",
        },
        toolUsed: false,
        toolUsedReason: "没有可用工具",
        toolUsedArgs: null,
      };
    }
    const system = [
      "你是一个工具调度器（Tool Router）。你将决定是否需要调用 MCP 工具来回答用户问题。",
      "工具调用成本很高。只有在不用工具无法可靠回答、需要实时信息/外部系统查询/精确计算时才调用。",
      `本轮最多允许调用工具 ${max - used} 次（总上限 ${max} 次）。如果不是必须，请不要调用工具。`,
      "你必须只输出一个 JSON 对象，禁止输出任何额外文本。",
      "JSON 结构如下：",
      `{ "needTool": boolean, "toolName": string|null, "args": object|null, "reason": string }`,
      "规则：",
      "- needTool=false 时：toolName=null, args=null",
      "- needTool=true 时：toolName 必须完全匹配工具列表中的 name；args 必须是一个 JSON object",
      "- 如果工具列表不足以解决问题，则 needTool=false，并说明原因",
    ].join("\n");
    const human = [
      "【可用 MCP 工具列表】",
      JSON.stringify(
        tools.map((t) => ({
          name: t.name,
          description: t.description ?? "",
          inputSchema: t.inputSchema ?? null,
        })),
        null,
        2
      ),
      "",
      `【用户问题】${String(state.input ?? "")}`,
    ].join("\n");
    const res = await llm.invoke([
      new SystemMessage(system),
      new HumanMessage(human),
    ]);
    const plan = safeParseJsonObject(String((res && res.content) || ""));
    if (!plan || typeof plan.needTool !== "boolean") {
      return {
        toolPlan: {
          needTool: false,
          toolName: null,
          args: null,
          reason: "决策输出无法解析为有效 JSON，已降级为不调用工具。",
        },
        toolUsed: false,
        toolUsedReason: "决策解析失败",
        toolUsedArgs: null,
      };
    }
    const needTool = plan.needTool === true;
    const toolName = needTool ? plan.toolName : null;
    const args = needTool ? normalizeArgs(plan.args) : null;
    const reason = typeof plan.reason === "string" ? plan.reason : "";
    if (needTool) {
      if (!isToolNameAllowed(toolName, tools)) {
        return {
          toolPlan: {
            needTool: false,
            toolName: null,
            args: null,
            reason: "模型选择的工具不在 MCP 列表中，已降级为不调用工具。",
          },
          toolUsed: false,
          toolUsedReason: "工具名不合法",
          toolUsedArgs: null,
        };
      }
      if (args == null) {
        return {
          toolPlan: {
            needTool: false,
            toolName: null,
            args: null,
            reason: "模型给出的工具参数不是合法对象，已降级为不调用工具。",
          },
          toolUsed: false,
          toolUsedReason: "参数不合法",
          toolUsedArgs: null,
        };
      }
    }
    return {
      toolPlan: {
        needTool,
        toolName,
        args,
        reason,
      },
      toolUsed: needTool,
      toolUsedReason: reason,
      toolUsedArgs: args,
    };
  };
}

// invoke工厂函数 ， 返回一个异步函数，用于调用 MCP 工具
export function createDefaultMcpInvoke(options) {
  //读取配置， options -> 环境变量 -> 默认值  依次读取
  const baseUrl = options?.mcpBaseUrl || process.env.MCP_BASE_URL || "http://127.0.0.1:5700";
  const authToken = options?.mcpAuthToken || process.env.MCP_AUTH_TOKEN || "";
  const headers = options?.mcpHeaders || null;
  return async ({ name, args, state }) => {
    const sessionId =
      options?.mcpSessionId || state?.threadId || state?.sessionId || "default";
    return callMcpTool({
      baseUrl,
      authToken,
      sessionId,
      headers,
      name,
      args,
    });
  };
}

export function createToolNode({ normalizeToolList, createDefaultMcpInvoke, options }) {
  return async (state) => {
    //清洗工具列表
    const tools = normalizeToolList(state.mcpTools);
    const plan = state.toolPlan;

    //如果不需要调用工具，直接返回
    if (!plan?.needTool) {
      return {
        toolResult: null,
        toolUsedResult: null,
        toolUsedResultSummary: null,
      };
    }

    //从工具列表中筛选出需要的工具名称
    const tool = tools.find((t) => t.name === plan.toolName);
    if (!tool) {
      return {
        toolResult: null,
        toolUsedResult: null,
        toolUsedResultSummary: "工具不存在，已跳过调用。",
      };
    }
    //如果配置了自定义的 MCP 执行器，使用它；否则使用默认执行器
    const invoke = typeof options.mcpInvoke === "function" ? options.mcpInvoke : createDefaultMcpInvoke(options);
    if (typeof invoke !== "function") {
      return {
        toolResult: null,
        toolUsedResult: null,
        toolUsedResultSummary: "未配置 MCP 执行器，已跳过调用。",
      };
    }
    try {
      //运行返回的invoke函数，调用 MCP 工具
      const result = await invoke({
        name: tool.name,
        args: plan.args,
        tool,
        state,
      });
      //成功后生成摘要， 如果是字符串直接使用，否则 JSON.stringify 转换
      const summary = typeof result === "string" ? result : JSON.stringify(result);
      return {
        toolResult: result,
        toolUsedResult: result,
        toolUsedResultSummary: summary,
      };
    } catch (e) {
      return {
        toolResult: null,
        toolUsedResult: null,
        toolUsedResultSummary: `工具调用失败：${e?.message ?? String(e)}`,
      };
    }
  };
}

import { callMcpTool } from "./mcpClient.js";

export function createHydrateNode({ loadMessagesFromRedis, RemoveMessage, REMOVE_ALL_MESSAGES }) {
  return async (state) => {
    try {
      const threadId = state.threadId ?? "default";
      const restored = await loadMessagesFromRedis(threadId);
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
    return {
      messages: [new HumanMessage(state.input)],
    };
  };
}

export function createRagNode({ ragChain, AIMessage }) {
  return async (state) => {
    const toolHint = state.toolUsedResultSummary
      ? `\n\n工具结果：\n${String(state.toolUsedResultSummary)}`
      : "";
    const res = await ragChain.invoke({ input: `${String(state.input ?? "")}${toolHint}` });
    const answer = String(res?.answer ?? res?.output ?? "");
    const context = res?.context ?? [];
    return {
      answer,
      context,
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


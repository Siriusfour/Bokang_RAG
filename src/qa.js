/**
 * 问答逻辑（RAG）：Retriever + Ollama Chat
 * 目标：尽量用 LangChain 现成链路，减少自写 glue code
 */

import { ChatPromptTemplate } from "@langchain/core/prompts";
import { AIMessage, HumanMessage, RemoveMessage, SystemMessage } from "@langchain/core/messages";
import { mapChatMessagesToStoredMessages, mapStoredMessagesToChatMessages } from "@langchain/core/messages";
import { createClient } from "redis";
import { Annotation, END, REMOVE_ALL_MESSAGES, START, StateGraph, messagesStateReducer } from "@langchain/langgraph";
import { createStuffDocumentsChain } from "@langchain/classic/chains/combine_documents";
import { createRetrievalChain } from "@langchain/classic/chains/retrieval";
import { ChatOllama } from "@langchain/ollama";
import { ChatOpenAI } from "@langchain/openai";
import { summarizationMiddleware } from "langchain";
import { config } from "./config.js";
import { safeParseJsonObject, normalizeToolList, isToolNameAllowed, normalizeArgs } from "./prase.js";
import {
  createHydrateNode,
  createIngestNode,
  createRagNode,
  createMcpListNode,
  createSummarizeNode,
  createPersistNode,
  createDecideNode,
  createToolNode,
  createDefaultMcpInvoke,
} from "./node.js";

export function createChatModel(options = {}) {
  const modelUrl = options.modelUrl ?? config.ollama.modelUrl;
  if (modelUrl) {
    return new ChatOpenAI({
      apiKey: options.apiKey ?? config.ollama.apiKey,
      model: options.modelName ?? config.ollama.modelName ?? config.ollama.chatModel,
      temperature: options.temperature ?? config.ollama.temperature,
      streaming: options.streaming ?? false,
      configuration: {
        baseURL: modelUrl,
      },
    });
  }
  return new ChatOllama({
    baseUrl: options.ollamaBaseUrl ?? config.ollama.baseUrl,
    model: options.chatModel ?? config.ollama.chatModel,
    temperature: options.temperature ?? config.ollama.temperature,
    streaming: options.streaming ?? false,
  });
}

// Redis 客户端单例 Promise，避免重复连接
let redisClientPromise = null;

function getRedisClient() {
  // 初始化 Redis 客户端，失败时清空 Promise 以允许重试
  if (redisClientPromise) return redisClientPromise;

  redisClientPromise = Promise.resolve()
    .then(() => {
      const url = config.redis?.url;
      if (!url) {
        throw new Error("Redis url is not configured.");
      }

      const client = createClient({
        url,
        username: config.redis?.username || undefined,
        password: config.redis?.password || undefined,
        database: typeof config.redis?.db === "number" ? config.redis.db : undefined,
      });

      client.on("error", (err) => {
        console.warn("⚠️ Redis error:", err?.message ?? err);
      });

      return client.connect().then(() => client);
    })
    .catch((err) => {
      redisClientPromise = null;
      throw err;
    });

  return redisClientPromise;
}

function redisKeyForThread(threadId) {
  // 按 threadId 生成 Redis key，支持自定义前缀
  const prefix = config.redis?.keyPrefix ?? "rag:mem:";
  const id = String(threadId || "default");
  return `${prefix}${id}`;
}

async function loadMessagesFromRedis(threadId) {
  // 从 Redis 读取并反序列化消息
  const client = await getRedisClient();
  const key = redisKeyForThread(threadId);
  const raw = await client.get(key);
  if (!raw) return [];

  const parsed = JSON.parse(raw);
  const stored = Array.isArray(parsed?.messages) ? parsed.messages : [];
  return mapStoredMessagesToChatMessages(stored);
}

function mapMessagesForStorage(messages) {
  // 清理 <think> 与 think 字段，避免写入 Redis
  // 目的：只持久化对用户可见的内容，避免思考链泄露
  const stripThinkFromContent = (content) => {
    if (typeof content !== "string") return content;
    return content.replace(/<think>[\s\S]*?<\/think>\s*/gi, "");
  };
  const stripThinkFields = (obj) => {
    if (!obj || typeof obj !== "object") return obj;
    const next = { ...obj };
    delete next.think;
    return next;
  };

  return mapChatMessagesToStoredMessages(messages).map((m) => {
    const data = m?.data ?? {};
    return {
      ...m,
      data: {
        ...data,
        content: stripThinkFromContent(data.content),
        additional_kwargs: stripThinkFields(data.additional_kwargs),
        response_metadata: stripThinkFields(data.response_metadata),
      },
    };
  });
}

function buildRedisPayload(messages) {
  // 统一 Redis 存储结构，便于版本演进
  return {
    schemaVersion: 1,
    updatedAt: Date.now(),
    messages: mapMessagesForStorage(messages),
  };
}

function estimateRedisValueBytes(messages) {
  // 估算 Redis value 字节大小，用于判断是否触发压缩
  const payload = buildRedisPayload(messages);
  return Buffer.byteLength(JSON.stringify(payload));
}

async function saveMessagesToRedis(threadId, messages) {
  // 将消息持久化到 Redis，可选 TTL
  const client = await getRedisClient();
  const key = redisKeyForThread(threadId);
  const payload = buildRedisPayload(messages);
  const ttlSeconds = Number(config.redis?.ttlSeconds ?? 0);
  if (Number.isFinite(ttlSeconds) && ttlSeconds > 0) {
    await client.set(key, JSON.stringify(payload), { EX: ttlSeconds });
  } else {
    await client.set(key, JSON.stringify(payload));
  }
}

//
export function createRagChain(vectorStore, options = {}) {
  // 创建检索+生成链路
  // 结构：Retriever -> Stuff Documents Chain -> LLM
  const llm = createChatModel(options);

  // Prompt 结构：system 指令 + human 问题与上下文拼接
  const prompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      [
        "你是一个基于给定上下文回答问题的助手。",
        "只能使用上下文中的信息回答；\"。",
         "并且你要对你的回答做出一些扩展和比较详细的解释；\"。",
        "回答请使用中文。"
        ,
      ].join("\n"),
    ],
    ["human", "问题：{input}\n\n上下文：\n{context}"],
  ]);

  return createStuffDocumentsChain({
    llm,
    prompt,
  }).then((combineDocsChain) => {
    // Retriever 负责从向量库检索 topK 文档
    const retriever = vectorStore.asRetriever({
      k: options.topK ?? config.retrieval.topK,
    });

    // RetrievalChain 将检索结果和问题注入到 prompt，并调用 LLM 生成答案
    return createRetrievalChain({
      retriever,
      combineDocsChain,
    });
  });
}

export function createRagGraph(vectorStore, options = {}) {
  // 构建 LangGraph：hydrate -> ingest -> rag -> summarize -> persist
  // 目标：统一管理消息状态、上下文与持久化
  return createRagChain(vectorStore, options).then((ragChain) => {
    // 摘要模型与主模型保持一致，确保输出风格一致
    const summaryModel = createChatModel(options);

    // 生成全局 State，用于在节点之间传递数据
    const GraphState = Annotation.Root({
      messages: Annotation({
        reducer: messagesStateReducer,
        default: () => [],
      }),
      threadId: Annotation(),
      input: Annotation(),
      answer: Annotation(),
      context: Annotation(),

      mcpTools: Annotation(),     // MCP 列表（数组/字符串）
      toolPlan: Annotation(),     // { needTool, toolName, args, reason }
      toolResult: Annotation(),   // 工具返回
      toolUsed: Annotation(),     // 是否使用了工具
      toolUsedReason: Annotation(), // 工具使用原因
      toolUsedArgs: Annotation(),   // 工具使用参数
      toolUsedResultSummary: Annotation(), // 工具使用结果摘要
      toolUsedResult: Annotation(),       // 工具使用原始结果
      toolCallsCount: Annotation({
        reducer: (a, b) => a + (b?.needTool === true ? 1 : 0),
        default: () => 0,
      }),
      toolCallsMax: Annotation({
        reducer: (a, b) => Math.max(a, b?.needTool === true ? 1 : 0),
        default: () => 0,
      }),

    });

    //初始化图流程编排器
    const graph = new StateGraph(GraphState)
      .addNode("hydrate", createHydrateNode({ loadMessagesFromRedis, RemoveMessage, REMOVE_ALL_MESSAGES }))
      .addNode("ingest", createIngestNode({ HumanMessage }))
      .addNode("rag", createRagNode({ ragChain, AIMessage }))
      .addNode("mcp_list", createMcpListNode())
      .addNode("summarize",createSummarizeNode({SystemMessage,RemoveMessage,REMOVE_ALL_MESSAGES,summarizationMiddleware,summaryModel,estimateRedisValueBytes,config,}))
      .addNode("persist", createPersistNode({ saveMessagesToRedis }))
      .addNode("tool",createToolNode({normalizeToolList,createDefaultMcpInvoke,options,}))
      .addNode("decide",createDecideNode({createChatModel,normalizeToolList,safeParseJsonObject,normalizeArgs,isToolNameAllowed,SystemMessage,HumanMessage,options,}))

.addEdge(START, "hydrate")
.addEdge("hydrate", "ingest")
// 准备 MCP 工具列表 + 决策
.addEdge("ingest", "mcp_list")
.addEdge("mcp_list", "decide")

//条件分支（需要工具才走 tool）
.addConditionalEdges("decide", (state) => {
  // 硬限制：超过上限直接不走工具
  const used = Number(state.toolCallsCount ?? 0);
  const max = Number(state.toolCallsMax ?? 1);
  if (used >= max) return "rag";

  return state.toolPlan?.needTool === true ? "tool" : "rag";
})

// 工具执行完成后，再走 rag（或直接 generate，看业务需求）
.addEdge("tool", "rag")

// 后面不变
.addEdge("rag", "summarize")
.addEdge("summarize", "persist")
.addEdge("persist", END)
      
    return graph.compile();
  });
}

export function ask(ragApp, state, question) {
  // 对外统一入口：传入当前 state 与问题，返回更新后的 state 与答案
  return ragApp.invoke({ ...state, input: question }).then((nextState) => {
    return {
      state: nextState,
      answer: String(nextState?.answer ?? ""),
      context: nextState?.context ?? [],
    };
  });
}

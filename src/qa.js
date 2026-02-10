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
import { summarizationMiddleware } from "langchain";
import { config } from "./config.js";

export function createChatModel(options = {}) {
  // 创建聊天模型，优先使用调用方传入的配置，否则回退到全局配置
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
  const llm = createChatModel(options);

  const prompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      [
        "你是一个基于给定上下文回答问题的助手。",
        "只能使用上下文中的信息回答；\"。",
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
    const retriever = vectorStore.asRetriever({
      k: options.topK ?? config.retrieval.topK,
    });

    return createRetrievalChain({
      retriever,
      combineDocsChain,
    });
  });
}

export function createRagGraph(vectorStore, options = {}) {
  // 构建 LangGraph：hydrate -> ingest -> rag -> summarize -> persist
  return createRagChain(vectorStore, options).then((ragChain) => {
    // 摘要模型与主模型保持一致，确保输出风格一致
    const summaryModel = createChatModel(options);

    //生成state（Graph的全局变量）
    const GraphState = Annotation.Root({
      messages: Annotation({
        reducer: messagesStateReducer,
        default: () => [],
      }),
      threadId: Annotation(),
      input: Annotation(),
      answer: Annotation(),
      context: Annotation(),
    });

    const graph = new StateGraph(GraphState)
      .addNode("hydrate", async (state) => {
        // 从 Redis 恢复历史消息到内存状态
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
      })
      .addNode("ingest", (state) => {
        // 将当前用户输入追加到消息队列
        return {
          messages: [new HumanMessage(state.input)],
        };
      })
      .addNode("rag", async (state) => {
        // 执行 RAG 生成，并写入 AI 消息
        const res = await ragChain.invoke({ input: state.input });
        const answer = String(res?.answer ?? res?.output ?? "");
        const context = res?.context ?? [];
        return {
          answer,
          context,
          messages: [new AIMessage(answer)],
        };
      })
      .addNode("summarize", async (state) => {
        // 当 Redis value 超过阈值时，将历史消息压缩为摘要
        try {
          const messages = Array.isArray(state.messages) ? state.messages : [];
          const maxValueBytes = Number(config.redis?.maxValueBytes ?? 0);
          if (!Number.isFinite(maxValueBytes) || maxValueBytes <= 0) return {};
          const estimatedSize = estimateRedisValueBytes(messages);
          if (estimatedSize <= maxValueBytes) return {};

          // system 消息永久保留
          const systemMessages = messages.filter((m) => SystemMessage.isInstance(m));
          const nonSystemMessages = messages.filter(
            (m) => !SystemMessage.isInstance(m) && !RemoveMessage.isInstance(m)
          );
          if (nonSystemMessages.length === 0) return {};

          // 只对非 system 消息做摘要，并保留最近 N 条原文
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
      })
      .addNode("persist", async (state) => {
        // 将最终消息写回 Redis
        try {
          const threadId = state.threadId ?? "default";
          await saveMessagesToRedis(threadId, state.messages ?? []);
        } catch (e) {
          console.warn("⚠️ Redis persist failed:", e?.message ?? e);
        }
        return {};
      })
      .addEdge(START, "hydrate")
      .addEdge("hydrate", "ingest")
      .addEdge("ingest", "rag")
      .addEdge("rag", "summarize")
      .addEdge("summarize", "persist")
      .addEdge("persist", END);

    return graph.compile();
  });
}

export function ask(ragApp, state, question) {
  // 对外统一入口，返回更新后的状态与答案
  return ragApp.invoke({ ...state, input: question }).then((nextState) => {
    return {
      state: nextState,
      answer: String(nextState?.answer ?? ""),
      context: nextState?.context ?? [],
    };
  });
}

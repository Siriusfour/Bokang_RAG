/**
 * 问答逻辑（RAG）：Retriever + Ollama Chat
 *
 * 目标：尽量用 LangChain 现成链路，减少自写 glue code
 */

import { ChatPromptTemplate } from "@langchain/core/prompts";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { ChatOllama } from "@langchain/ollama";
import { config } from "./config.js";

export function createChatModel(options = {}) {
  return new ChatOllama({
    baseUrl: options.ollamaBaseUrl ?? config.ollama.baseUrl,
    model: options.chatModel ?? config.ollama.chatModel,
    temperature: options.temperature ?? config.ollama.temperature,
  });
}

//
export function createRagChain(vectorStore, options = {}) {
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

export function ask(ragChain, question) {
  return ragChain.invoke({ input: question }).then((res) => {
    const answer = String(res?.answer ?? res?.output ?? "");
    const context = res?.context ?? [];
    return { answer, context };
  });
}

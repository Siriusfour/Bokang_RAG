/**
 * 问答逻辑（RAG）：Retriever + Ollama Chat
 *
 * 目标：尽量用 LangChain 现成链路，减少自写 glue code
 */
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { ChatOllama } from "@langchain/ollama";
export function createChatModel(options = {}) {
    return new ChatOllama({
        baseUrl: options.ollamaBaseUrl ?? process.env.OLLAMA_BASE_URL ?? "http://127.0.0.1:11434",
        model: options.chatModel ?? process.env.OLLAMA_CHAT_MODEL ?? "llama3.1",
        temperature: options.temperature ??
            (process.env.OLLAMA_TEMPERATURE ? Number(process.env.OLLAMA_TEMPERATURE) : 0.2),
    });
}
export function createRagChain(vectorStore, options = {}) {
    const llm = createChatModel(options);
    const prompt = ChatPromptTemplate.fromMessages([
        [
            "system",
            [
                "你是一个严格基于给定上下文回答问题的助手。",
                "只能使用上下文中的信息回答；如果上下文不足以回答，请明确说“我在提供的文档中找不到相关信息”。",
                "回答请使用中文。",
            ].join("\n"),
        ],
        ["human", "问题：{input}\n\n上下文：\n{context}"],
    ]);
    return createStuffDocumentsChain({
        llm,
        prompt,
    }).then((combineDocsChain) => {
        const retriever = vectorStore.asRetriever({
            k: options.topK ?? 4,
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
        const context = (res?.context ?? []);
        return { answer, context };
    });
}

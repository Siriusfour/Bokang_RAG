/**
 * 向量库构建 / 加载（本地持久化）
 *
 * - VectorStore: HNSWLib（本地索引，支持 save/load）
 * - Embeddings: OllamaEmbeddings（走本地 Ollama 服务）
 */
import path from "node:path";
import fs from "node:fs";
import { HNSWLib } from "@langchain/community/vectorstores/hnswlib";
import { OllamaEmbeddings } from "@langchain/ollama";
const DEFAULT_PERSIST_DIR = path.resolve(process.cwd(), "data", "vectorstore");
export function createEmbeddings(options = {}) {
    return new OllamaEmbeddings({
        baseUrl: options.ollamaBaseUrl ?? process.env.OLLAMA_BASE_URL ?? "http://127.0.0.1:11434",
        model: options.embeddingModel ?? process.env.OLLAMA_EMBEDDING_MODEL ?? "nomic-embed-text",
    });
}
export function loadVectorStore(options = {}) {
    const persistDir = options.persistDir ?? DEFAULT_PERSIST_DIR;
    const embeddings = createEmbeddings(options);
    return HNSWLib.load(persistDir, embeddings);
}
/**
 * 如果本地已有索引则加载，否则用传入 documents 构建并持久化
 */
export function buildOrLoadVectorStore(documents, options = {}) {
    const persistDir = options.persistDir ?? DEFAULT_PERSIST_DIR;
    const embeddings = createEmbeddings(options);
    if (fs.existsSync(persistDir)) {
        return HNSWLib.load(persistDir, embeddings);
    }
    if (!documents || documents.length === 0) {
        return Promise.reject(new Error(`向量库不存在（${persistDir}），且未提供可用于构建的 documents。请先加载并切分 .docs 文档。`));
    }
    fs.mkdirSync(persistDir, { recursive: true });
    return HNSWLib.fromDocuments(documents, embeddings).then((vs) => vs.save(persistDir).then(() => vs));
}
export function deleteVectorStore(options = {}) {
    const persistDir = options.persistDir ?? DEFAULT_PERSIST_DIR;
    if (fs.existsSync(persistDir)) {
        fs.rmSync(persistDir, { recursive: true, force: true });
    }
}

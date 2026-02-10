/**
 * CLI：本地 Ollama + LangChain RAG
 *
 * 运行：
 * - npm i
 * - 确保 ollama 已启动，并 pull 了模型：
 *   - ollama pull nomic-embed-text
 *   - ollama pull llama3.1
 * - npm run dev
 */
import "dotenv/config";
import readline from "node:readline";
import { loadDocuments, splitDocuments } from "./loadDocs.js";
import { buildOrLoadVectorStore, deleteVectorStore } from "./buildVectorStore.js";
import { ask, createRagChain } from "./qa.js";
import { createAgent } from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";


function ensureVectorStore() {
    return buildOrLoadVectorStore(null).catch(() => {
        return loadDocuments()
            .then((docs) => splitDocuments(docs, {
            chunkSize: process.env.CHUNK_SIZE ? Number(process.env.CHUNK_SIZE) : 1000,
            chunkOverlap: process.env.CHUNK_OVERLAP ? Number(process.env.CHUNK_OVERLAP) : 200,
        }))
            .then((chunks) => buildOrLoadVectorStore(chunks));
    });
}
function main() {
    ensureVectorStore()
        .then((vectorStore) => createRagChain(vectorStore, {
        topK: process.env.TOP_K ? Number(process.env.TOP_K) : 4,
    }))
        .then((ragChain) => {
        const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
        console.log("本地知识库 RAG CLI 已启动。输入问题；exit 退出；rebuild 重建索引 ；show 显示索引。");
        const loop = () => {
            rl.question("你：", (input) => {
                const q = input.trim();
                if (!q)
                    return loop();
                if (q.toLowerCase() === "exit") {
                    rl.close();
                    return;
                }
                if (q.toLowerCase() === "rebuild") {
                    deleteVectorStore();
                    console.log("已删除本地向量索引。现在会重新从 .docs 构建...");
                    loadDocuments()
                        .then((docs) => splitDocuments(docs, {
                        chunkSize: process.env.CHUNK_SIZE ? Number(process.env.CHUNK_SIZE) : 1000,
                        chunkOverlap: process.env.CHUNK_OVERLAP ? Number(process.env.CHUNK_OVERLAP) : 200,
                    }))
                        .then((chunks) => buildOrLoadVectorStore(chunks))
                        .then((vs) => createRagChain(vs, {
                        topK: process.env.TOP_K ? Number(process.env.TOP_K) : 4,
                    }))
                        .then((newChain) => {
                        ragChain.invoke = newChain.invoke.bind(newChain);
                        console.log("重建完成。");
                        loop();
                    })
                        .catch((err) => {
                        console.error("重建失败：", err);
                        loop();
                    });
                    return;
                }
                ask(ragChain, q)
                    .then((res) => {
                    console.log(`助手：${res.answer}`);
                })
                    .catch((err) => {
                    console.error("发生错误：", err);
                })
                    .finally(() => {
                    loop();
                });
            });
        };
        loop();
    })
        .catch((err) => {
        console.error("初始化失败：", err);
        process.exit(1);
    });
}
void main();

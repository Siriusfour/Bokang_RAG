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
import { config } from "./config.js";

function ensureVectorStore() {
  return buildOrLoadVectorStore(null).catch((error) => {
    // 如果加载失败（向量库不存在或损坏），从文档重新构建
    console.log("📖 向量库不存在或已损坏，正在从 .docs 加载文档并构建...");
    return loadDocuments()
      .then((docs) => {
        console.log(`✅ 已加载 ${docs.length} 个文档`);
        return splitDocuments(docs, {
          chunkSize: config.documents.chunkSize,
          chunkOverlap: config.documents.chunkOverlap,
        });
      })
      .then((chunks) => {
        console.log(`✅ 文档已切分为 ${chunks.length} 个块`);
        console.log("🔄 正在构建向量库（这可能需要一些时间）...");
        return buildOrLoadVectorStore(chunks);
      });
  });
}

function main() {
  ensureVectorStore()
    .then((vectorStore) =>
      createRagChain(vectorStore, {
        topK: config.retrieval.topK,
      })
    )
    .then((ragChain) => {
      const rl = readline.createInterface({ input: process.stdin, output: process.stdout });

      console.log("本地知识库 RAG CLI 已启动。输入问题；exit 退出；rebuild 重建索引。");

      const loop = () => {
        rl.question("你：", (input) => {
          const q = input.trim();
          if (!q) return loop();

          if (q.toLowerCase() === "exit") {
            rl.close();
            return;
          }

          if (q.toLowerCase() === "rebuild") {
            console.log("正在删除 Milvus collection 并重建...");
            Promise.resolve(deleteVectorStore())
              .then(() => {
                console.log("✅ 已删除 Milvus collection。现在会重新从 .docs 构建...");
              })
              .catch(() => {
                console.log("⚠️ 删除 collection 失败或 collection 不存在，将直接重建...");
              })
              .then(() => {
            loadDocuments()
              .then((docs) =>
                splitDocuments(docs, {
                  chunkSize: config.documents.chunkSize,
                  chunkOverlap: config.documents.chunkOverlap,
                })
              )
              .then((chunks) => buildOrLoadVectorStore(chunks))
              .then((vs) =>
                createRagChain(vs, {
                  topK: config.retrieval.topK,
                })
              )
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
              });
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

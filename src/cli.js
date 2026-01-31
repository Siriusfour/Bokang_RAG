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

import path from "node:path";
import "dotenv/config";
import readline from "node:readline";

import { loadDocuments, splitDocuments } from "./loadDocs.js";
import { buildOrLoadVectorStore, deleteVectorStore, showVectorStore } from "./buildVectorStore.js";
import { ask, createRagChain } from "./qa.js";
import { config } from "./config.js";

function ensureVectorStore() {
  return loadDocuments()
    .then((docs) => {
      docs.forEach(doc => {
        doc.metadata.source = path.relative(process.cwd(), doc.metadata.source);
      });
      console.log(`✅ 已加载 ${docs.length} 个文档`);
      return splitDocuments(docs, {
        chunkSize: config.documents.chunkSize,
        chunkOverlap: config.documents.chunkOverlap,
      });
    })
    .then((chunks) => {
      console.log(`✅ 文档已切分为 ${chunks.length} 个块`);
      return buildOrLoadVectorStore(chunks);
    })
    .catch((error) => {
      console.error("📖 加载文档或构建向量库失败:", error);
      throw error;
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

          if (q.toLowerCase() === "show") {
            console.log("🔍 正在查询向量数据库内容...");
            showVectorStore()
              .then((data) => {
                if (data.length === 0) {
                  console.log("ℹ️ 数据库为空，没有可显示的内容。");
                } else {
                  console.log(`✅ 查询到 ${data.length} 条记录 (最多显示 5 条):`);
                  data.forEach((item, index) => {
                    console.log(`\n--- [ 记录 ${index + 1} ] ---`);
                    Object.keys(item).forEach(key => {
                      let value = item[key];
                      if (typeof value === 'string' && value.length > 200) {
                        value = value.substring(0, 200) + '...';
                      }
                      console.log(`${key}: ${value}`);
                    });
                  });
                }
              })
              .catch((err) => {
                console.error("❌ 查询失败:", err.message);
              })
              .finally(() => {
                loop();
              });
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
              .then((docs) => {
                docs.forEach(doc => {
                  doc.metadata.source = path.relative(process.cwd(), doc.metadata.source);
                });
                return splitDocuments(docs, {
                  chunkSize: config.documents.chunkSize,
                  chunkOverlap: config.documents.chunkOverlap,
                })
              })
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

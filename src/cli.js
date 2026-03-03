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
import { performance } from "node:perf_hooks";

import { loadDocuments, splitDocuments } from "./loadDocs.js";
import { buildOrLoadVectorStore, deleteVectorStore, showVectorStore } from "./buildVectorStore.js";
import { ask, createRagGraph } from "./qa.js";
import { config } from "./config.js";

function time(label, fn) {
  // 统一计时包装器：记录开始、结束与错误耗时
  const start = performance.now();
  console.log(`⏱️ [timing] ${label} start`);
  return Promise.resolve()
    .then(fn)
    .then((res) => {
      const costMs = performance.now() - start;
      console.log(`⏱️ [timing] ${label} ${costMs.toFixed(1)}ms`);
      return res;
    })
    .catch((err) => {
      const costMs = performance.now() - start;
      console.log(`⏱️ [timing] ${label} ${costMs.toFixed(1)}ms error`);
      throw err;
    });
}

async function checkOllamaReady() {
  // 调用 /api/tags 检查 Ollama 是否可用
  const baseUrl = String(config.ollama.baseUrl || "").replace(/\/+$/, "");
  const url = `${baseUrl}/api/tags`;
  const controller = new AbortController();
  const t = setTimeout(() => controller.abort(), 5000);
  try {
    await fetch(url, { signal: controller.signal }).then((r) => r.json());
    console.log(`✅ Ollama 可用: ${baseUrl}`);
  } catch (e) {
    console.error(`❌ Ollama 无响应: ${url}`);
    throw e;
  } finally {
    clearTimeout(t);
  }
}

function ensureVectorStore() {
  // 加载文档 -> 切分 -> 构建/加载向量库
  return time("loadDocuments", () => loadDocuments())
    .then((docs) => {
      // 将 source 统一为相对路径，便于展示与持久化
      docs.forEach(doc => {
        doc.metadata.source = path.relative(process.cwd(), doc.metadata.source);
      });
      console.log(`✅ 已加载 ${docs.length} 个文档`);
      return time("splitDocuments", () =>
        splitDocuments(docs, {
          chunkSize: config.documents.chunkSize,
          chunkOverlap: config.documents.chunkOverlap,
          chunkByHeading: true,
        })
      );
    })
    .then((chunks) => {
      console.log(`✅ 文档已切分为 ${chunks.length} 个块`);
      return time("buildOrLoadVectorStore", () => buildOrLoadVectorStore(chunks));
    })
    .catch((error) => {
      console.error("📖 加载文档或构建向量库失败:", error);
      throw error;
    });
}

function main() {
  // 启动流程：检查 Ollama -> 构建向量库 -> 创建 RAG 图 -> 进入交互循环
  time("checkOllamaReady", () => checkOllamaReady())
    .then(() => time("ensureVectorStore", () => ensureVectorStore()))
    .then((vectorStore) =>
      time("createRagGraph", () =>
        createRagGraph(vectorStore, {
          topK: config.retrieval.topK,
        })
      )
    )
    .then((ragApp) => {
      // 创建交互式命令行输入
      const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
      let isClosed = false;   
      rl.on("close", () => {
        isClosed = true;
      });
      // threadId 用于标识对话线程，默认从环境变量读取
      const threadId = String(process.env.THREAD_ID || "default");
      let state = { threadId, messages: [] };

      console.log("本地知识库 RAG CLI 已启动。输入问题；exit 退出；rebuild 重建索引。");

      const loop = () => {
        if (isClosed) return;
        rl.question("你：", (input) => {
          if (isClosed) return;
          const q = input.trim();
          if (!q) return loop();

          if (q.toLowerCase() === "exit") {
            // 退出交互
            rl.close();
            return;
          }

          if (q.toLowerCase() === "show") {
            // 查询并展示向量库中部分记录
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
            // 删除并重建向量库
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
                // 重建时同样统一 source 为相对路径
                docs.forEach(doc => {
                  doc.metadata.source = path.relative(process.cwd(), doc.metadata.source);
                });
                return splitDocuments(docs, {
                  chunkSize: config.documents.chunkSize,
                  chunkOverlap: config.documents.chunkOverlap,
                  chunkByHeading: true,
                })
              })
              .then((chunks) => buildOrLoadVectorStore(chunks))
              .then((vs) =>
                createRagGraph(vs, {
                  topK: config.retrieval.topK,
                })
              )
              .then((newApp) => {
                // 替换 invoke 以复用现有 ragApp 引用
                ragApp.invoke = newApp.invoke.bind(newApp);
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

          // 普通问答流程：调用 RAG 并更新 state
          ask(ragApp, state, q)
            .then((res) => {
              state = res.state;
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

      // 启动主循环
      loop();
    })
    .catch((err) => {
      console.error("初始化失败：", err);
      process.exit(1);
    });
}

// 入口调用
void main();

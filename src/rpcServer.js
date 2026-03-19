import "dotenv/config";
import path from "node:path";
import { performance } from "node:perf_hooks";
import grpc from "@grpc/grpc-js";
import protoLoader from "@grpc/proto-loader";

import { loadDocuments, splitDocuments } from "./loadDocs.js";
import { buildOrLoadVectorStore } from "./buildVectorStore.js";
import { ask, createRagGraph } from "./qa.js";
import { config } from "./config.js";

function time(label, fn) {
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
  const baseUrl = String(config.ollama.baseUrl || "").replace(/\/+$/, "");
  const url = `${baseUrl}/api/tags`;
  const controller = new AbortController();
  const t = setTimeout(() => controller.abort(), 5000);
  try {
    await fetch(url, { signal: controller.signal }).then((r) => r.json());
    console.log(`✅ Ollama 可用: ${baseUrl}`);
  } finally {
    clearTimeout(t);
  }
}

function ensureVectorStore() {
  return time("loadDocuments", () => loadDocuments())
    .then((docs) => {
      docs.forEach((doc) => {
        if (doc?.metadata?.source) {
          doc.metadata.source = path.relative(process.cwd(), doc.metadata.source);
        }
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
    });
}

function loadProto() {
  const protoPath = path.resolve(process.cwd(), "proto", "ask.proto");
  const packageDefinition = protoLoader.loadSync(protoPath, {
    keepCase: true,
    longs: String,
    enums: String,
    defaults: true,
    oneofs: true,
  });
  const loaded = grpc.loadPackageDefinition(packageDefinition);
  return loaded?.rag?.ask?.v1;
}

function createAskHandler(ragApp) {
  return async (call, callback) => {
    const question = call?.request?.question;
    const sessionId = call?.request?.sessionId;
    const userId = call?.request?.userId;
    console.log(`userId: ${userId}`);
    console.log(`sessionId: ${sessionId}`);
    console.log(`question: ${question}`);

    if (!question || typeof question !== "string") {
      callback({
        code: grpc.status.INVALID_ARGUMENT,
        message: "question is required",
      });
      return;
    }
    try {
      const threadId =
        typeof sessionId === "string" && sessionId.trim() ? sessionId : "default";
      const userIdValue = typeof userId === "string" && userId.trim() ? userId : "default";
      const state = { UserID: userIdValue, ContextID: threadId, messages: []};
      const result = await ask(ragApp, state, question);
      callback(null, { answer: result.answer ?? "" });
    } catch (err) {
      callback({
        code: grpc.status.INTERNAL,
        message: err?.message ?? "internal error",
      });
    }
  };
}

async function startGrpcServer(ragApp) {
  const rpcProto = loadProto();
  const server = new grpc.Server();
  server.addService(rpcProto.AskService.service, {
    Ask: createAskHandler(ragApp),
  });
  const port = Number(config.rpc?.port ?? 7071);
  const address = `0.0.0.0:${port}`;
  await new Promise((resolve, reject) => {
    server.bindAsync(address, grpc.ServerCredentials.createInsecure(), (err) => {
      if (err) return reject(err);
      server.start();
      resolve();
    });
  });
  console.log(`gRPC server listening on ${address}`);
}

async function main() {
  await time("checkOllamaReady", () => checkOllamaReady());
  const vectorStore = await time("ensureVectorStore", () => ensureVectorStore());
  const ragApp = await time("createRagGraph", () =>
    createRagGraph(vectorStore, {
      topK: config.retrieval.topK,
    })
  );
  await startGrpcServer(ragApp);
}

main().catch((err) => {
  process.stderr.write(`${err?.message ?? String(err)}\n`);
  process.exit(1);
});

import "dotenv/config";
import http from "node:http";
import path from "node:path";
import { pathToFileURL } from "node:url";

import { loadDocuments, splitDocuments } from "./loadDocs.js";
import { buildOrLoadVectorStore } from "./buildVectorStore.js";
import { ask, createRagGraph } from "./qa.js";
import { config } from "./config.js";

async function ensureVectorStore() {
  const docs = await loadDocuments();
  docs.forEach((doc) => {
    if (doc?.metadata?.source) {
      doc.metadata.source = path.relative(process.cwd(), doc.metadata.source);
    }
  });
  const chunks = await splitDocuments(docs, {
    chunkSize: config.documents.chunkSize,
    chunkOverlap: config.documents.chunkOverlap,
    chunkByHeading: true,
  });
  return buildOrLoadVectorStore(chunks);
}

async function parseJsonBody(req) {
  const chunks = [];
  for await (const chunk of req) {
    chunks.push(chunk);
  }
  const raw = Buffer.concat(chunks).toString("utf8");
  if (!raw) return null;
  return JSON.parse(raw);
}

function sendJson(res, statusCode, payload) {
  const body = JSON.stringify(payload);
  res.writeHead(statusCode, {
    "content-type": "application/json; charset=utf-8",
    "content-length": Buffer.byteLength(body),
  });
  res.end(body);
}

export async function startServer(options = {}) {
  const port = Number(options.port ?? process.env.ASK_PORT ?? 7070);
  const vectorStore = await ensureVectorStore();
  const ragApp = await createRagGraph(vectorStore, {
    topK: config.retrieval.topK,
  });
  const server = http.createServer(async (req, res) => {
    if (req.method !== "POST" || req.url !== "/ask") {
      sendJson(res, 404, { error: "Not Found" });
      return;
    }
    try {
      const payload = await parseJsonBody(req);
      const question = payload?.question;
      if (!question || typeof question !== "string") {
        sendJson(res, 400, { error: "question is required" });
        return;
      }
      const threadId = typeof payload?.threadId === "string" ? payload.threadId : "default";
      const state = { threadId, messages: [] };
      const result = await ask(ragApp, state, question);
      sendJson(res, 200, result);
    } catch (e) {
      sendJson(res, 500, { error: e?.message ?? "internal error" });
    }
  });
  await new Promise((resolve) => server.listen(port, resolve));
  return { server, port };
}

const isMain = (() => {
  if (!process.argv[1]) return false;
  try {
    return pathToFileURL(process.argv[1]).href === import.meta.url;
  } catch {
    return false;
  }
})();

if (isMain) {
  startServer().catch((err) => {
    process.stderr.write(`${err?.message ?? String(err)}\n`);
    process.exit(1);
  });
}

import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";

function normalizeBaseUrl(value) {
  const raw = String(value || "");
  if (!raw) return "http://127.0.0.1:5700";
  return raw.replace(/\/+$/, "");
}

function buildHeaders(authToken, extraHeaders) {
  const headers = {};
  if (authToken) {
    headers.Authorization = authToken.startsWith("Bearer ") ? authToken : `Bearer ${authToken}`;
  }
  if (extraHeaders && typeof extraHeaders === "object") {
    Object.assign(headers, extraHeaders);
  }
  return headers;
}

function extractResultPayload(result) {
  if (result?.structuredContent !== undefined) return result.structuredContent;
  const content = Array.isArray(result?.content) ? result.content : [];
  const textParts = content
    .filter((item) => item && item.type === "text")
    .map((item) => item.text)
    .filter((text) => typeof text === "string" && text.length > 0);
  if (textParts.length > 0) return textParts.join("\n");
  return result;
}

export async function callMcpTool({
  baseUrl,
  authToken,
  sessionId,
  headers,
  name,
  args,
}) {
  const url = new URL(normalizeBaseUrl(baseUrl));
  const transport = new StreamableHTTPClientTransport(url, {
    sessionId: sessionId ? String(sessionId) : undefined,
    headers: buildHeaders(authToken, headers),
  });
  const client = new Client({ name: "rag-client", version: "1.0.0" });
  await client.connect(transport);
  try {
    const result = await client.callTool({
      name,
      arguments: args ?? {},
    });
    return extractResultPayload(result);
  } finally {
    await client.close();
  }
}

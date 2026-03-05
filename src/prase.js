// 更稳健：优先解析 ```json ...``` 代码块；否则退回提取第一个 { ... } 区间
export function safeParseJsonObject(text) {
  if (!text) return null;
  const s = String(text);

  // 1) 尝试匹配 ```json ... ```
  const fenced = s.match(/```(?:json)?\s*([\s\S]*?)\s*```/i);
  if (fenced && fenced[1]) {
    try {
      return JSON.parse(fenced[1]);
    } catch (_) {
      // ignore, fallback
    }
  }

  // 2) fallback：截取从第一个 { 到最后一个 }
  const start = s.indexOf("{");
  const end = s.lastIndexOf("}");
  if (start < 0 || end <= start) return null;

  const jsonLike = s.slice(start, end + 1);
  try {
    return JSON.parse(jsonLike);
  } catch (_) {
    return null;
  }
}

export function normalizeToolList(mcpTools) {
  if (!Array.isArray(mcpTools)) return [];
  return mcpTools.filter((t) => t && typeof t.name === "string");
}

export function isToolNameAllowed(toolName, tools) {
  if (typeof toolName !== "string") return false;
  return Array.isArray(tools) && tools.some((t) => t && t.name === toolName);
}

// 轻量 args 校验：只保证是 plain object（不是数组、不是 null）
export function normalizeArgs(args) {
  if (args == null) return null;
  if (typeof args === "object" && !Array.isArray(args)) return args;
  return null;
}

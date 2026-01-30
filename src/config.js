/**
 * 配置读取模块
 * 
 * 优先级：配置文件 > 环境变量 > 默认值
 */

import fs from "node:fs";
import path from "node:path";

const CONFIG_FILE = path.resolve(process.cwd(), "config.json");

let cachedConfig = null;

/**
 * 读取配置文件
 */
function loadConfigFile() {
  if (cachedConfig !== null) {
    return cachedConfig;
  }

  if (fs.existsSync(CONFIG_FILE)) {
    try {
      const content = fs.readFileSync(CONFIG_FILE, "utf-8");
      cachedConfig = JSON.parse(content);
      console.log(`✅ 已加载配置文件: ${CONFIG_FILE}`);
      return cachedConfig;
    } catch (error) {
      console.warn(`⚠️ 配置文件解析失败: ${error.message}，将使用默认值`);
      return null;
    }
  } else {
    console.log(`ℹ️ 配置文件不存在 (${CONFIG_FILE})，将使用环境变量或默认值`);
    return null;
  }
}

/**
 * 获取配置值（优先级：配置文件 > 环境变量 > 默认值）
 */
function getConfig(path, envVar, defaultValue) {
  const config = loadConfigFile();
  
  // 从配置文件读取
  if (config) {
    const keys = path.split(".");
    let value = config;
    for (const key of keys) {
      if (value && typeof value === "object" && key in value) {
        value = value[key];
      } else {
        value = undefined;
        break;
      }
    }
    if (value !== undefined) {
      return value;
    }
  }
  
  // 从环境变量读取
  if (envVar && process.env[envVar]) {
    const envValue = process.env[envVar];
    // 尝试转换为数字
    if (typeof defaultValue === "number") {
      const num = Number(envValue);
      if (!isNaN(num)) return num;
    }
    return envValue;
  }
  
  // 返回默认值
  return defaultValue;
}

/**
 * 导出配置对象
 */
export const config = {
  ollama: {
    baseUrl: getConfig("ollama.baseUrl", "OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
    embeddingModel: getConfig("ollama.embeddingModel", "OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
    chatModel: getConfig("ollama.chatModel", "OLLAMA_CHAT_MODEL", "llama3.1"),
    temperature: getConfig("ollama.temperature", "OLLAMA_TEMPERATURE", 0.2),
  },
  documents: {
    docsDir: getConfig("documents.docsDir", "DOCS_DIR", ".docs"),
    chunkSize: getConfig("documents.chunkSize", "CHUNK_SIZE", 1000),
    chunkOverlap: getConfig("documents.chunkOverlap", "CHUNK_OVERLAP", 200),
  },
  retrieval: {
    topK: getConfig("retrieval.topK", "TOP_K", 4),
  },
  vectorStore: {
    /**
     * 支持：milvus / hnswlib（目前项目默认切换到 milvus）
     */
    type: getConfig("vectorStore.type", "VECTOR_STORE_TYPE", "milvus"),
    persistDir: getConfig("vectorStore.persistDir", "VECTOR_STORE_DIR", "data/vectorstore"),
  },
  milvus: {
    url: getConfig("milvus.url", "MILVUS_URL", "http://127.0.0.1:19530"),
    username: getConfig("milvus.username", "MILVUS_USERNAME", ""),
    password: getConfig("milvus.password", "MILVUS_PASSWORD", ""),
    ssl: getConfig("milvus.ssl", "MILVUS_SSL", false),
    collectionName: getConfig("milvus.collectionName", "MILVUS_COLLECTION", "langchain_docs"),
    /**
     * 可选：partitionName
     */
    partitionName: getConfig("milvus.partitionName", "MILVUS_PARTITION", ""),
    /**
     * 文本字段最大长度（默认 65535，足够存储大文档块）
     */
    textFieldMaxLength: getConfig("milvus.textFieldMaxLength", "MILVUS_TEXT_FIELD_MAX_LENGTH", 65535),
  },
};

/**
 * 向量库构建 / 加载（本地持久化）
 *
 * - VectorStore: HNSWLib（本地索引，支持 save/load）
 * - Embeddings: OllamaEmbeddings（走本地 Ollama 服务）
 */

import { Milvus } from "@langchain/community/vectorstores/milvus";
import { OllamaEmbeddings } from "@langchain/ollama";
import { DataType } from "@zilliz/milvus2-sdk-node";
import { performance } from "node:perf_hooks";
import { config } from "./config.js";

// 规范化 Milvus 地址，兼容 http(s)://host:port 与 host:port 两种输入
function normalizeMilvusAddress(raw) {
  // 空值或非字符串直接返回
  if (!raw) return raw;
  if (typeof raw !== "string") return raw;
  const trimmed = raw.trim();
  try {
    // 先按 URL 解析，提取 hostname 与端口
    const u = new URL(trimmed);
    if (u.hostname) {
      const port = u.port || "19530";
      return `${u.hostname}:${port}`;
    }
    return trimmed;
  } catch {
    // 不是标准 URL 时，去掉协议头即可
    return trimmed.replace(/^https?:\/\//i, "");
  }
}

// 简单的耗时监控器，定期输出运行时长，结束后打印总耗时
function monitor(label, fn, intervalMs = 10000) {
  const start = performance.now();
  console.log(`⏱️ [timing] ${label} start`);

  //每隔intervalMs秒，执行一次参数内的函数
  const timer = setInterval(() => {
    const costMs = performance.now() - start;
    console.log(`⏱️ [timing] ${label} running ${costMs.toFixed(0)}ms`);
  }, intervalMs);

  // 统一 Promise 包装，确保同步/异步函数都可被监控
  return Promise.resolve()
    .then(fn)
    .then((res) => {
      clearInterval(timer);
      const costMs = performance.now() - start;
      console.log(`⏱️ [timing] ${label} ${costMs.toFixed(1)}ms`);
      return res;
    })
    .catch((err) => {
      clearInterval(timer);
      const costMs = performance.now() - start;
      console.log(`⏱️ [timing] ${label} ${costMs.toFixed(1)}ms error`);
      throw err;
    });
}


// 创建 Embeddings，并为 embedQuery / embedDocuments 注入耗时监控
export function createEmbeddings(options = {}) {
  // 模型与服务地址均可被 options 覆盖
  const embeddings = new OllamaEmbeddings({
    baseUrl: options.ollamaBaseUrl ?? config.ollama.baseUrl,
    model: options.embeddingModel ?? config.ollama.embeddingModel,
  });

  // 避免重复打补丁
  if (!embeddings.__timingPatched) {
    embeddings.__timingPatched = true;

    if (typeof embeddings.embedQuery === "function") {
      const originalEmbedQuery = embeddings.embedQuery.bind(embeddings);
      // 单条向量化耗时
      embeddings.embedQuery = async (...args) =>
        monitor("embeddings.embedQuery", () => originalEmbedQuery(...args));
    }

    if (typeof embeddings.embedDocuments === "function") {
      const originalEmbedDocuments = embeddings.embedDocuments.bind(embeddings);
      // 批量向量化耗时（附带文本数量）
      embeddings.embedDocuments = async (texts, ...rest) => {
        const n = Array.isArray(texts) ? texts.length : 0;
        return monitor(`embeddings.embedDocuments n=${n}`, () => originalEmbedDocuments(texts, ...rest));
      };
    }
  }

  return embeddings;
}

// 创建并返回 Milvus VectorStore 实例
export function loadVectorStore(options = {}) {
  // 每次都创建 embeddings，确保配置可由 options 覆盖
  const embeddings = createEmbeddings(options);
  return Promise.resolve(
    new Milvus(embeddings, {
      // collection/partition 由 options 或 config 指定
      collectionName: options.collectionName ?? config.milvus.collectionName,
      partitionName:
        options.partitionName ?? (config.milvus.partitionName ? config.milvus.partitionName : undefined),
      // 地址、鉴权与字段配置
      url: normalizeMilvusAddress(options.url ?? config.milvus.url),
      username: options.username ?? config.milvus.username,
      password: options.password ?? config.milvus.password,
      ssl: options.ssl ?? config.milvus.ssl,
      textField: options.textField ?? "text",
      vectorField: options.vectorField ?? "vector",
      textFieldMaxLength: options.textFieldMaxLength ?? config.milvus.textFieldMaxLength,
    })
  );
}

/**
 * 检查 Milvus collection 是否存在
 */
function checkCollectionExists(vectorStore, collectionName) {
  // SDK 返回的 value 表示是否存在
  return vectorStore.client
    .hasCollection({ collection_name: collectionName })
    .then((res) => res.value === true)
    .catch(() => false);
}

/**
 * 如果本地已有索引则加载，否则用传入 documents 构建并持久化
 */


export function deleteVectorStore(options = {}) {
  // 对 Milvus 来说，删除向量库=删除 collection
  return loadVectorStore(options).then((vs) => {
    const collectionName = options.collectionName ?? config.milvus.collectionName;
    // milvus2-sdk-node API：dropCollection({ collection_name })
    return vs.client
      .dropCollection({ collection_name: collectionName })
      .then(() => undefined)
      .catch(() => undefined);
  });
}

export function showVectorStore(options = {}) {
  const collectionName = options.collectionName ?? config.milvus.collectionName;
  let vectorStore;

  // 加载 vectorStore 并确认 collection 存在
  return loadVectorStore(options)
    .then((vs) => {
      vectorStore = vs;
      return checkCollectionExists(vs, collectionName);
    })
    .then((exists) => {
      if (!exists) {
        throw new Error(`Milvus collection "${collectionName}" 不存在。`);
      }

      // 首先加载 collection 确保其可被查询
      return vectorStore.client.loadCollectionSync({
        collection_name: collectionName,
      }).then(() => {
        // 然后获取 collection 的 schema 来查找所有字段
        return vectorStore.client.describeCollection({ collection_name: collectionName });
      });
    })
    .then((desc) => {
        if (desc.status && desc.status.error_code !== 'Success') {
            throw new Error(`获取 collection 结构失败: ${desc.status.reason}`);
        }
        // 解析 schema，决定查询输出字段
        const schemaFields = desc.schema?.fields ?? [];
        const primaryField =
          schemaFields.find((f) => f.is_primary_key)?.name ?? vectorStore.primaryField;
        const output_fields = schemaFields
          .map((f) => f.name)
          .filter((name) => name && name !== vectorStore.vectorField);
        const final_output_fields =
          output_fields.length > 0
            ? output_fields
            : [vectorStore.primaryField, vectorStore.textField, "source"].filter(Boolean);
        const expr =
          typeof primaryField === "string" && primaryField.length > 0
            ? `${primaryField} >= 0`
            : "pk >= 0";

        // 查询少量数据用于展示
        return vectorStore.client.query({
            collection_name: collectionName,
            expr,
            output_fields: final_output_fields,
            limit: 5,
        });
    })
    .then((res) => {
      if (res.status && res.status.error_code !== 'Success') {
        throw new Error(`查询失败: ${res.status.reason}`);
      }
      return res.data;
    });
}

// 确保 collection 存在：不存在则创建 schema、索引并加载
async function ensureCollection(vectorStore, documents) {
  // 判断 collection 是否存在
  const hasColResp = await monitor("milvus.hasCollection", () =>
    vectorStore.client.hasCollection({
      collection_name: vectorStore.collectionName,
    })
  );
  if (hasColResp.status?.error_code && hasColResp.status.error_code !== "Success") {
    throw new Error(`Error checking collection: ${JSON.stringify(hasColResp)}`);
  }
  if (hasColResp.value === true) {
    return;
  }

  // 用嵌入模型探测向量维度
  const dimProbe = await monitor("dimensionProbe", () => vectorStore.embeddings.embedQuery("dimension_probe"));
  const dim = Array.isArray(dimProbe) ? dimProbe.length : 0;
  if (!dim) {
    throw new Error("Failed to determine embedding dimension.");
  }

  // 根据首个文档的 metadata 动态生成字段
  const sampleMetadata = documents[0]?.metadata ?? {};
  const metadataFields = Object.entries(sampleMetadata)
    .filter(([key]) => key !== vectorStore.primaryField && key !== vectorStore.partitionKey)
    .map(([key, value]) => {
      const t = typeof value;
      if (t === "number") {
        return {
          name: key,
          description: "Metadata Number field",
          data_type: DataType.Float,
        };
      }
      if (t === "boolean") {
        return {
          name: key,
          description: "Metadata Boolean field",
          data_type: DataType.Bool,
        };
      }
      if (value === null || value === undefined) {
        return null;
      }
      return {
        name: key,
        description: "Metadata String field",
        data_type: DataType.VarChar,
        type_params: {
          max_length: "4096",
        },
      };
    })
    .filter(Boolean);

  // 组合完整 schema：metadata + 主键 + 文本 + 向量
  const fields = [
    ...metadataFields,
    {
      name: vectorStore.primaryField,
      description: "Primary key",
      data_type: DataType.Int64,
      is_primary_key: true,
      autoID: true,
    },
    {
      name: vectorStore.textField,
      description: "Text field",
      data_type: DataType.VarChar,
      type_params: {
        max_length: String(vectorStore.textFieldMaxLength || 65535),
      },
    },
    {
      name: vectorStore.vectorField,
      description: "Vector field",
      data_type: DataType.FloatVector,
      type_params: {
        dim: String(dim),
      },
    },
  ];

  // 创建 collection
  const createRes = await monitor("milvus.createCollection", () =>
    vectorStore.client.createCollection({
      collection_name: vectorStore.collectionName,
      fields,
    })
  );
  if (createRes.error_code && createRes.error_code !== "Success") {
    throw new Error(`Failed to create collection: ${JSON.stringify(createRes)}`);
  }

  // 创建向量索引（HNSW）
  await monitor("milvus.createIndex", () =>
    vectorStore.client.createIndex({
      collection_name: vectorStore.collectionName,
      field_name: vectorStore.vectorField,
      extra_params: {
        index_type: "HNSW",
        metric_type: "L2",
        params: JSON.stringify({ M: 8, efConstruction: 64 }),
      },
    })
  );

  // 加载 collection 以便后续查询
  await monitor("milvus.loadCollectionSync", () =>
    vectorStore.client.loadCollectionSync({
      collection_name: vectorStore.collectionName,
    })
  );
}



/**
 * 构建或加载向量库
 * @param {Array} documents - 要插入的文档数组
 * @param {Object} options - 配置选项（可选）
 * @returns {Promise<Milvus>} 返回 Milvus vectorStore 实例
 */
export async function buildOrLoadVectorStore(documents, options = {}) {
  const collectionName = options.collectionName ?? config.milvus.collectionName;
  
  try {
    // 先尝试加载已有 collection
    const vectorStore = await loadVectorStore(options);
    const exists = await checkCollectionExists(vectorStore, collectionName);

    if (exists) {
      console.log(`✅ Milvus collection "${collectionName}" 已存在，直接使用`);
      await monitor("milvus.loadCollectionSync(existing)", () =>
        vectorStore.client.loadCollectionSync({
          collection_name: collectionName,
        })
      );
      return vectorStore;
    }

    // 不存在时需要 documents 来构建 collection
    if (!documents || documents.length === 0) {
      throw new Error(
        `Milvus collection "${collectionName}" 不存在，且未提供可用于构建的 documents。`
      );
    }

    // 创建 collection 并插入文档
    console.log(`🔄 Milvus collection "${collectionName}" 不存在，正在创建并插入数据...`);
    await monitor("ensureCollection", () => ensureCollection(vectorStore, documents));
    await monitor(`vectorStore.addDocuments n=${documents.length}`, () => vectorStore.addDocuments(documents));
    console.log("✅ Collection 创建并插入成功");
    return vectorStore;

  } catch (err) {
    // 发生严重错误时清理 collection，避免残留不一致状态
    console.error("❌ 创建或插入数据时发生严重错误:", err);
    await deleteVectorStore(options).catch(() => {});
    throw err;
  }
}

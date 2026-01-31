/**
 * 向量库构建 / 加载（本地持久化）
 *
 * - VectorStore: HNSWLib（本地索引，支持 save/load）
 * - Embeddings: OllamaEmbeddings（走本地 Ollama 服务）
 */

import { Milvus } from "@langchain/community/vectorstores/milvus";
import { OllamaEmbeddings } from "@langchain/ollama";
import { DataType } from "@zilliz/milvus2-sdk-node";
import { config } from "./config.js";

export function createEmbeddings(options = {}) {
  return new OllamaEmbeddings({
    baseUrl: options.ollamaBaseUrl ?? config.ollama.baseUrl,
    model: options.embeddingModel ?? config.ollama.embeddingModel,
  });
}

export function loadVectorStore(options = {}) {
  const embeddings = createEmbeddings(options);
  return Promise.resolve(
    new Milvus(embeddings, {
      collectionName: options.collectionName ?? config.milvus.collectionName,
      partitionName:
        options.partitionName ?? (config.milvus.partitionName ? config.milvus.partitionName : undefined),
      url: options.url ?? config.milvus.url,
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

async function ensureCollection(vectorStore, documents) {
  const hasColResp = await vectorStore.client.hasCollection({
    collection_name: vectorStore.collectionName,
  });
  if (hasColResp.status?.error_code && hasColResp.status.error_code !== "Success") {
    throw new Error(`Error checking collection: ${JSON.stringify(hasColResp)}`);
  }
  if (hasColResp.value === true) {
    return;
  }

  const dimProbe = await vectorStore.embeddings.embedQuery("dimension_probe");
  const dim = Array.isArray(dimProbe) ? dimProbe.length : 0;
  if (!dim) {
    throw new Error("Failed to determine embedding dimension.");
  }

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

  const createRes = await vectorStore.client.createCollection({
    collection_name: vectorStore.collectionName,
    fields,
  });
  if (createRes.error_code && createRes.error_code !== "Success") {
    throw new Error(`Failed to create collection: ${JSON.stringify(createRes)}`);
  }

  await vectorStore.client.createIndex({
    collection_name: vectorStore.collectionName,
    field_name: vectorStore.vectorField,
    extra_params: {
      index_type: "HNSW",
      metric_type: "L2",
      params: JSON.stringify({ M: 8, efConstruction: 64 }),
    },
  });

  await vectorStore.client.loadCollectionSync({
    collection_name: vectorStore.collectionName,
  });
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
    const vectorStore = await loadVectorStore(options);
    const exists = await checkCollectionExists(vectorStore, collectionName);

    if (exists) {
      console.log(`✅ Milvus collection "${collectionName}" 已存在，直接使用`);
      await vectorStore.client.loadCollectionSync({
        collection_name: collectionName,
      });
      return vectorStore;
    }

    if (!documents || documents.length === 0) {
      throw new Error(
        `Milvus collection "${collectionName}" 不存在，且未提供可用于构建的 documents。`
      );
    }

    console.log(`🔄 Milvus collection "${collectionName}" 不存在，正在创建并插入数据...`);
    await ensureCollection(vectorStore, documents);
    await vectorStore.addDocuments(documents);
    console.log("✅ Collection 创建并插入成功");
    return vectorStore;

  } catch (err) {
    console.error("❌ 创建或插入数据时发生严重错误:", err);
    await deleteVectorStore(options).catch(() => {});
    throw err;
  }
}

/**
 * 向量库构建 / 加载（本地持久化）
 *
 * - VectorStore: HNSWLib（本地索引，支持 save/load）
 * - Embeddings: OllamaEmbeddings（走本地 Ollama 服务）
 */

import { Milvus } from "@langchain/community/vectorstores/milvus";
import { OllamaEmbeddings } from "@langchain/ollama";
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
      textFieldMaxLength: options.textFieldMaxLength ?? config.milvus.textFieldMaxLength ?? 65535,
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
export function buildOrLoadVectorStore(documents, options = {}) {
  const collectionName = options.collectionName ?? config.milvus.collectionName;
  
  return loadVectorStore(options).then((vs) => {
    // 先检查 collection 是否存在
    return checkCollectionExists(vs, collectionName).then((exists) => {
      if (exists) {
        // Collection 已存在，直接返回
        console.log(`✅ Milvus collection "${collectionName}" 已存在，直接使用`);
        return vs;
      }
      
      // Collection 不存在，需要创建
      if (!documents || documents.length === 0) {
        return Promise.reject(
          new Error(
            `Milvus collection "${collectionName}" 不存在，且未提供可用于构建的 documents。请先加载并切分 .docs 文档。`
          )
        );
      }
      
      // 创建 collection 并插入文档
      console.log(`🔄 Milvus collection "${collectionName}" 不存在，正在创建并插入 ${documents.length} 个文档...`);
      return vs.addDocuments(documents).then(() => {
        console.log(`✅ Milvus collection "${collectionName}" 创建完成`);
        return vs;
      }).catch((error) => {
        // 如果插入失败（可能是旧的 collection 配置不兼容），删除并重试
        console.warn(`⚠️ 插入文档失败，可能是旧的 collection 配置不兼容，正在删除并重建...`);
        return vs.client
          .dropCollection({ collection_name: collectionName })
          .catch(() => {}) // 忽略删除失败（可能 collection 不存在）
          .then(() => {
            // 重新创建 VectorStore（使用新的配置）
            return loadVectorStore(options).then((newVs) => {
              console.log(`🔄 重新创建 collection 并插入 ${documents.length} 个文档...`);
              return newVs.addDocuments(documents).then(() => {
                console.log(`✅ Milvus collection "${collectionName}" 创建完成`);
                return newVs;
              });
            });
          });
      });
    });
  });
}

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

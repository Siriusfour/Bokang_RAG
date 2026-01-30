/**
 * 文档加载 & 切分模块（使用 LangChain 内置工具）
 *
 * - 使用 DirectoryLoader 从 `.docs` 批量加载：txt / md / pdf / docx
 * - 使用 RecursiveCharacterTextSplitter 切分为可向量化的 chunks
 */

import path from "node:path";

import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { DocxLoader } from "@langchain/community/document_loaders/fs/docx";
import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { config } from "./config.js";

/**
 * 从 `.docs` 目录批量加载文档（txt/md/pdf/docx）
 */
export function loadDocuments(options = {}) {
  const docsDir = options.docsDir ?? path.resolve(process.cwd(), config.documents.docsDir);

  const loader = new DirectoryLoader(docsDir, {
    ".txt": (p) => new TextLoader(p),
    ".md": (p) => new TextLoader(p),
    ".pdf": (p) => new PDFLoader(p),
    ".docx": (p) => new DocxLoader(p),
  });

  return loader.load();
}

/**
 * 使用 LangChain 的递归切分器切分文档（更贴近语义边界）
 */
export function splitDocuments(documents, options = {}) {
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: options.chunkSize ?? config.documents.chunkSize,
    chunkOverlap: options.chunkOverlap ?? config.documents.chunkOverlap,
  });

  return splitter.splitDocuments(documents);
}

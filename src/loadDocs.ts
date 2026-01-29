/**
 * 文档加载 & 切分模块（使用 LangChain 内置工具）
 *
 * - 使用 DirectoryLoader 从 `.docs` 批量加载：txt / md / pdf / docx
 * - 使用 RecursiveCharacterTextSplitter 切分为可向量化的 chunks
 */

import path from "node:path";

import { Document } from "@langchain/core/documents";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { DocxLoader } from "@langchain/community/document_loaders/fs/docx";
import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

export interface LoadDocsOptions {
  /**
   * 文档根目录，默认指向项目下的 .docs
   */
  docsDir?: string;
}

export interface SplitOptions {
  chunkSize?: number;
  chunkOverlap?: number;
}

const DEFAULT_DOCS_DIR = path.resolve(process.cwd(), ".docs");

/**
 * 从 `.docs` 目录批量加载文档（txt/md/pdf/docx）
 */
export function loadDocuments(
  options: LoadDocsOptions = {}
): Promise<Document[]> {
  const docsDir = options.docsDir ?? DEFAULT_DOCS_DIR;

  const loader = new DirectoryLoader(docsDir, {
    ".txt": (p: string) => new TextLoader(p),
    ".md": (p: string) => new TextLoader(p),
    ".pdf": (p: string) => new PDFLoader(p),
    ".docx": (p: string) => new DocxLoader(p),
  });

  return loader.load();
}

/**
 * 使用 LangChain 的递归切分器切分文档（更贴近语义边界）
 */
export function splitDocuments(
  documents: Document[],
  options: SplitOptions = {}
): Promise<Document[]> {
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: options.chunkSize ?? 1000,
    chunkOverlap: options.chunkOverlap ?? 200,
  });

  return splitter.splitDocuments(documents);
}


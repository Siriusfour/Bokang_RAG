/**
 * 文档加载 & 切分模块（使用 LangChain 内置工具）
 *
 * - 使用 DirectoryLoader 从 `.docs` 批量加载：txt / md / pdf / docx
 * - 使用 RecursiveCharacterTextSplitter 切分为可向量化的 chunks
 */

import path from "node:path";

import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { DirectoryLoader } from "@langchain/classic/document_loaders/fs/directory";
import { TextLoader } from "@langchain/classic/document_loaders/fs/text";
import { Document } from "@langchain/core/documents";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import mammoth from "mammoth";
import { config } from "./config.js";

function decodeHtmlEntities(value) {
  return value
    .replace(/&nbsp;/g, " ")
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'");
}

function htmlToText(value) {
  if (!value) return "";
  const withBreaks = value
    .replace(/<br\s*\/?>/gi, "\n")
    .replace(/<\/p>/gi, "\n")
    .replace(/<\/h[1-6]>/gi, "\n");
  const stripped = withBreaks.replace(/<[^>]+>/g, "");
  return decodeHtmlEntities(stripped).replace(/\n{2,}/g, "\n").trim();
}

function extractHtmlBlocks(html) {
  const blocks = [];
  const regex = /<(h[1-6]|p)[^>]*>[\s\S]*?<\/\1>/gi;
  let match;
  while ((match = regex.exec(html)) !== null) {
    blocks.push({ tag: match[1].toLowerCase(), html: match[0] });
  }
  return blocks;
}

function detectNumberedHeading(text) {
  const match = /^(\d+(?:\.\d+){1,6})\s+(.+)$/.exec(text);
  if (!match) return null;
  const dots = match[1].split(".").length - 1;
  const level = Math.min(dots + 1, 6);
  return { level, title: `${match[1]} ${match[2]}`.trim() };
}

function buildHeadingSectionsFromHtml(html) {
  const blocks = extractHtmlBlocks(html);
  const sections = [];
  const headingStack = [];
  let current = { title: null, level: null, body: [] };

  const pushCurrent = () => {
    if (!current.title && current.body.length === 0) return;
    const title = current.title ?? "前言";
    const path = current.title ? headingStack.map((h) => h.title).join(" / ") : title;
    const content = [title, ...current.body].join("\n").trim();
    if (!content) return;
    sections.push({
      title,
      level: current.level ?? 0,
      path,
      content,
    });
  };

  for (const block of blocks) {
    const tag = block.tag;
    const inner = block.html
      .replace(new RegExp(`^<${tag}[^>]*>`, "i"), "")
      .replace(new RegExp(`</${tag}>$`, "i"), "");
    const text = htmlToText(inner);
    if (!text) continue;
    if (tag.startsWith("h")) {
      pushCurrent();
      const level = Number(tag.slice(1));
      while (headingStack.length > 0 && headingStack[headingStack.length - 1].level >= level) {
        headingStack.pop();
      }
      headingStack.push({ title: text, level });
      current = { title: text, level, body: [] };
      continue;
    }
    if (tag === "p") {
      const numbered = detectNumberedHeading(text);
      if (numbered) {
        pushCurrent();
        const level = numbered.level;
        while (headingStack.length > 0 && headingStack[headingStack.length - 1].level >= level) {
          headingStack.pop();
        }
        headingStack.push({ title: numbered.title, level });
        current = { title: numbered.title, level, body: [] };
        continue;
      }
      current.body.push(text);
    }
  }

  pushCurrent();

  if (sections.length === 0) {
    const fallback = htmlToText(html);
    if (fallback) {
      sections.push({
        title: "正文",
        level: 0,
        path: "正文",
        content: fallback,
      });
    }
  }

  return sections;
}

class HeadingDocxLoader {
  constructor(filePath) {
    this.filePath = filePath;
  }

  async load() {
    const result = await mammoth.convertToHtml(
      { path: this.filePath },
      {
        styleMap: [
          "p[style-name='Heading 1'] => h1:fresh",
          "p[style-name='Heading 2'] => h2:fresh",
          "p[style-name='Heading 3'] => h3:fresh",
          "p[style-name='Heading 4'] => h4:fresh",
          "p[style-name='Heading 5'] => h5:fresh",
          "p[style-name='Heading 6'] => h6:fresh",
        ],
      }
    );
    const sections = buildHeadingSectionsFromHtml(result?.value ?? "");
    if (sections.length === 0) {
      return [];
    }
    return sections.map((section, index) => {
      return new Document({
        pageContent: section.content,
        metadata: {
          source: this.filePath,
          heading: section.title,
          headingLevel: section.level,
          headingPath: section.path,
          headingIndex: index,
          chunkedByHeading: true,
        },
      });
    });
  }
}

/**
 * 从 `.docs` 目录批量加载文档（txt/md/pdf/docx）
 */
export function loadDocuments(options = {}) {
  const docsDir = options.docsDir ?? path.resolve(process.cwd(), config.documents.docsDir);

  const loader = new DirectoryLoader(docsDir, {
    ".txt": (p) => new TextLoader(p),
    ".md": (p) => new TextLoader(p),
    ".pdf": (p) => new PDFLoader(p),
    ".docx": (p) => new HeadingDocxLoader(p),
  });

  return loader.load();
}

/**
 * 使用 LangChain 的递归切分器切分文档（更贴近语义边界）
 */
export async function splitDocuments(documents, options = {}) {
  const chunkByHeading = options.chunkByHeading !== false;
  const headingDocs = chunkByHeading
    ? documents.filter((doc) => doc?.metadata?.chunkedByHeading === true)
    : [];
  const remainingDocs = chunkByHeading
    ? documents.filter((doc) => doc?.metadata?.chunkedByHeading !== true)
    : documents;
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: options.chunkSize ?? config.documents.chunkSize,
    chunkOverlap: options.chunkOverlap ?? config.documents.chunkOverlap,
  });

  const headingMaxChars = options.headingMaxChars ?? options.chunkSize ?? config.documents.chunkSize;
  const splitHeadingDocsArrays = await Promise.all(headingDocs.map(async (doc) => {
    const content = String(doc?.pageContent ?? "");
    if (!headingMaxChars || content.length <= headingMaxChars) {
      return [doc];
    }
    const parts = await splitter.splitText(content);
    return parts.map((part, index) => {
      return new Document({
        pageContent: part,
        metadata: {
          ...doc.metadata,
          headingChunkIndex: index,
          headingChunkCount: parts.length,
        },
      });
    });
  }));
  const splitHeadingDocs = splitHeadingDocsArrays.flat();

  const splitPromise = remainingDocs.length > 0 ? splitter.splitDocuments(remainingDocs) : Promise.resolve([]);
  const splits = await Promise.resolve(splitPromise);
  const merged = [...splitHeadingDocs, ...splits];
  return merged.map((doc) => {
    const metadata = doc?.metadata ?? {};
    if (metadata.chunkedByHeading === true) {
      return doc;
    }
    return new Document({
      pageContent: doc?.pageContent ?? "",
      metadata: {
        ...metadata,
        heading: metadata.heading ?? "",
        headingLevel: metadata.headingLevel ?? 0,
        headingPath: metadata.headingPath ?? "",
        headingIndex: metadata.headingIndex ?? -1,
        headingChunkIndex: metadata.headingChunkIndex ?? 0,
        headingChunkCount: metadata.headingChunkCount ?? 1,
        chunkedByHeading: false,
      },
    });
  });
}

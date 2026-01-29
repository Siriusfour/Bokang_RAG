# 本地知识库（LangChain JS + Ollama）

## 依赖

- Node.js 18+
- 已安装并启动 Ollama（默认地址：`http://127.0.0.1:11434`）

## 准备文档

把你的文件放到项目根目录的 `.docs/` 下（支持：`txt`/`md`/`pdf`/`docx`）。

## 拉取模型（示例）

```bash
ollama pull nomic-embed-text
ollama pull llama3.1
```

## 环境变量（可选）

在你的环境里设置（PowerShell 示例）：

```powershell
$env:OLLAMA_BASE_URL="http://127.0.0.1:11434"
$env:OLLAMA_EMBEDDING_MODEL="nomic-embed-text"
$env:OLLAMA_CHAT_MODEL="llama3.1"
$env:OLLAMA_TEMPERATURE="0.2"
$env:CHUNK_SIZE="1000"
$env:CHUNK_OVERLAP="200"
$env:TOP_K="4"
```

## 安装与运行

```bash
npm install
npm run dev
```

运行后：
- 输入问题直接问答
- 输入 `rebuild` 删除并重建本地向量索引（默认保存到 `data/vectorstore`）
- 输入 `exit` 退出


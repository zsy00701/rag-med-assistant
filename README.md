# rag-med-assistant

## 1.项目摘要
本项目旨在解决大型语言模型在专业医疗领域知识的准确性和时效性问题，通过构建一套基于检索增强生成 (RAG) 架构的智能问答系统。系统专注于提供肝胆胰外科手术患者术后营养支持和康复指导。通过整合权威医疗指南和文献，本项目确保为患者和初级医护人员提供快速、准确、且基于最新证据的营养管理建议，从而辅助术后康复过程，提高患者依从性和治疗效果。
### 核心目标:
* 提供基于最新医疗证据的专业营养指导。
* 实现 95% 以上的答案忠实度 (Faithfulness)。
* 建立可增量更新的专业知识库。

## 2.技术栈
| 模块 | 技术 / 库 | 描述 |
| :--- | :--- | :--- |
| **编程语言** | Python 3.10+ | 项目主要开发语言。 |
| **框架** | FastAPI | 用于构建高性能的检索和 RAG API 服务。 |
| **RAG 编排** | LangChain / LlamaIndex | 负责整合检索、提示词工程和 LLM 调用的核心逻辑。 |
| **向量数据库** | ChromaDB (MVP) | 轻量级、易于部署的向量存储解决方案。支持本地存储和 Chroma Cloud。 |
| **LLM / Embedding** | Deepseek / Llama-3 (或 OpenAI API) | 核心生成模型和文本转向量模型。 |
| **环境管理** | Poetry / virtualenv | 依赖和虚拟环境管理。 |

## 3.快速开始

### 3.1 环境要求
- Python 3.10 或更高版本
- OpenAI API Key（用于 LLM 和 Embedding）

### 3.2 安装依赖

项目依赖包括：
- `chromadb` - 向量数据库（支持本地和云端）
- `langchain` - RAG 框架
- `langchain-openai` - OpenAI 集成
- `fastapi` / `uvicorn` - Web API 框架
- `python-dotenv` - 环境变量管理

#### 方式一：使用 uv（推荐）
```bash
# 安装 uv（如果还没有安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 安装项目依赖
uv sync
```

#### 方式二：使用 pip
```bash
pip install -r requirements.txt
```

**注意**：`chromadb` 包已包含在依赖中，安装后即可使用本地数据库或 Chroma Cloud。

### 3.3 配置环境变量

#### 方式一：使用本地向量数据库（默认，推荐新手）

创建 `.env` 文件：

```bash
# 必需：OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here
```

系统会自动使用本地向量数据库（存储在 `./chroma_db_medical` 目录）。

#### 方式二：使用 Chroma Cloud（推荐生产环境）

1. **注册 Chroma Cloud 账号**
   - 访问 [Chroma Cloud](https://www.trychroma.com/)
   - 注册账号并创建项目
   - 获取 API Key、Tenant ID 等信息

2. **创建 `.env` 文件**：

```bash
# 必需：OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here

# Chroma Cloud 配置（可选，如果设置则使用云端数据库）
CHROMA_API_KEY=your_chroma_api_key_here
CHROMA_TENANT=your_tenant_id
CHROMA_DATABASE=your_database_name
CHROMA_COLLECTION=medical_qa
```

**注意**：
- 如果不设置 `CHROMA_API_KEY`，系统会自动使用本地向量数据库
- 如果设置了 `CHROMA_API_KEY` 但连接失败，系统会自动回退到本地数据库
- 详细配置说明请参考 `CHROMA_CLOUD_SETUP.md`

### 3.4 数据导入

首次使用前，需要将医疗问答数据导入向量数据库：

```bash
python src/ingest.py
```

该脚本会：
1. 读取 `QA_V0.md` 文件（格式：`Q：... A：...`）
2. 解析问答对
3. 生成向量嵌入
4. 存储到 ChromaDB（本地或云端）

导入成功后，你会看到类似输出：
```
✅ 成功解析出 XX 条问答数据。
🎉 入库成功！数据已保存到本地: ./chroma_db_medical
```

### 3.5 启动服务

#### 方式一：启动 API 服务（推荐）

```bash
python src/api.py
```

或者使用 uvicorn 直接启动：

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

服务启动后，访问：
- API 文档：http://localhost:8000/docs
- 健康检查：http://localhost:8000

#### 方式二：命令行交互模式

```bash
python src/rag.py
```

启动后可以直接在命令行中提问，输入 `q` 或 `quit` 退出。

### 3.6 使用 Web 界面

1. 确保 API 服务已启动（见 3.5 方式一）
2. 在浏览器中打开 `frontend/index.html` 文件
3. 在界面中输入问题，点击"发送"按钮

**注意**：如果 API 服务运行在不同端口，需要修改 `frontend/index.html` 中的 `API_URL` 变量（第 367 行）。

## 4.API 使用示例

### 4.1 发送问题

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"question": "术后第一天可以吃什么？"}'
```

### 4.2 响应格式

```json
{
  "answer": "根据术后营养管理指南...",
  "status": "success"
}
```

## 5.项目结构

```
rag-med-assistant/
├── src/
│   ├── api.py          # FastAPI 服务入口
│   ├── rag.py          # RAG 链构建和命令行交互
│   └── ingest.py       # 数据导入脚本
├── frontend/
│   └── index.html      # Web 前端界面
├── chroma_db_medical/  # 本地向量数据库（自动生成）
├── QA_V0.md           # 医疗问答数据源
├── requirements.txt    # Python 依赖
├── pyproject.toml      # 项目配置（uv/poetry）
└── README.md          # 本文件
```

## 6.常见问题

### Q: 提示"向量数据库未找到"？
A: 请先运行 `python src/ingest.py` 导入数据。

### Q: 提示"未设置 OPENAI_API_KEY"？
A: 请设置环境变量 `OPENAI_API_KEY`，或在 `.env` 文件中配置。

### Q: 如何更新知识库？
A: 修改 `QA_V0.md` 文件后，重新运行 `python src/ingest.py`。注意：这会覆盖现有数据，如需增量更新，请修改 `ingest.py` 脚本。

### Q: 如何使用 Chroma Cloud？
A: 设置 `CHROMA_API_KEY` 等环境变量后，系统会自动使用云端数据库。详见 `CHROMA_CLOUD_SETUP.md`。

## 7.开发说明

### 运行测试
```bash
pytest
```

### 代码检查
```bash
flake8 src/
```

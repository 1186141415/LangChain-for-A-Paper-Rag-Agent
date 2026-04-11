# LangChain for A Paper RAG Agent

一个基于 **LangChain** 实现的论文问答 / 检索增强生成（RAG）最小实践项目。  
项目面向论文场景，支持将论文文本切分、向量化、检索，并结合大模型完成基于上下文的问答。

这个项目的目标不是堆砌复杂功能，而是搭建一个清晰、可运行、可扩展的 **Paper RAG baseline**，并为后续的 **Agent / LangGraph / Web 部署**打基础。

---

## 1. Project Overview

在论文阅读场景中，单纯依赖大模型往往会出现以下问题：

- 无法准确引用论文内容
- 回答容易脱离原文上下文
- 对长文档支持较弱
- 难以针对特定论文进行稳定问答

因此，本项目通过引入 **RAG（Retrieval-Augmented Generation）** 机制，让模型在回答问题前，先从论文内容中检索相关片段，再基于检索到的上下文生成回答。

当前项目定位为：

- **LangChain 版最小 RAG 实践**
- **论文问答系统的基础原型**
- **后续 Agent / LangGraph 扩展的起点**
- **可用于 GitHub / 博客 / 简历展示的小型工程项目**

---

## 2. Main Features

当前版本计划或已支持的能力包括：

- 论文文本加载
- 文本切分（chunking）
- 向量化与向量索引构建
- 基于相似度检索相关上下文
- 基于检索上下文生成回答
- 面向论文场景的问答流程封装

后续可扩展方向：

- 多论文对比问答
- 会话记忆 / Session 管理
- Tool Calling
- Router / Agent 调度
- LangGraph 工作流
- Django / FastAPI Web 接入

---

## 3. Tech Stack

本项目主要使用以下技术：

- **Python**
- **LangChain**
- **FAISS / Chroma**（按你的实现选择）
- **Embedding Model**
- **LLM API**
- **PyCharm / Git / GitHub**

你可以根据自己的实现替换具体组件，例如：

- Embedding：`text-embedding-3-small`
- LLM：`deepseek-chat` / `gpt-4o-mini` / other OpenAI-compatible models
- Vector Store：`FAISS`

---

## 4. Project Structure

当前项目建议的基础结构如下：

```text
LangChain-for-A-Paper-Rag-Agent/
├── data/                     # 论文文本 / 原始资料
├── app.py                    # 主程序入口
├── requirements.txt          # 项目依赖
├── README.md                 # 项目说明文档
├── .gitignore                # Git 忽略文件
└── .env                      # 环境变量（不提交）
```

后续如果项目继续扩展，可以进一步拆分为：

```text
LangChain-for-A-Paper-Rag-Agent/
├── data/
├── src/
│   ├── loader.py
│   ├── splitter.py
│   ├── vector_store.py
│   ├── rag_chain.py
│   └── utils.py
├── app.py
├── requirements.txt
├── README.md
└── .env
```

---

## 5. Workflow

本项目的核心流程如下：

1. **加载论文文本**
2. **对文本进行切分**
3. **将文本块向量化**
4. **建立向量索引**
5. **接收用户问题**
6. **从向量库中检索相关文本块**
7. **将检索结果作为上下文传给大模型**
8. **生成最终回答**

可抽象为：

```text
User Question
    ↓
Retriever
    ↓
Relevant Chunks
    ↓
LLM with Context
    ↓
Final Answer
```

---

## 6. Why LangChain

本项目选择 LangChain 的原因不是为了单纯“使用框架”，而是希望通过框架实践，理解：

- RAG 的标准化构建方式
- 文档加载、切分、检索、生成之间的衔接关系
- 框架到底帮我们封装了哪些能力
- 手写 RAG 与框架实现之间的对应关系

因此，这个项目既可以作为：

- **LangChain 入门实践**
- 也可以作为 **手写 RAG 到框架化 RAG 的对照实验**

---

## 7. Quick Start

### 7.1 Clone the Repository

```bash
git clone https://github.com/your-username/LangChain-for-A-Paper-Rag-Agent.git
cd LangChain-for-A-Paper-Rag-Agent
```

### 7.2 Create Virtual Environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux / macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 7.3 Install Dependencies

```bash
pip install -r requirements.txt
```

### 7.4 Configure Environment Variables

在项目根目录创建 `.env` 文件，例如：

```env
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=your_api_base_url
MODEL_NAME=your_model_name
EMBEDDING_MODEL=your_embedding_model
```

如果你使用的是 OpenAI-compatible 接口，可以按实际服务修改。

### 7.5 Prepare Data

将论文文本或处理后的数据放入 `data/` 目录中。

例如：

```text
data/
└── paper1.txt
```

### 7.6 Run the Project

```bash
python app.py
```

---

## 8. Example Use Case

一个典型问题示例：

```text
What is the main contribution of this paper?
```

系统流程：

- 检索论文中与“main contribution”最相关的文本块
- 将检索结果作为上下文
- 由大模型生成回答

另一个示例：

```text
What is the difference between the proposed method and previous work?
```

这类问题尤其适合通过 RAG 来减少模型“脱离原文发挥”的情况。

---

## 9. Current Position in My Project Roadmap

这个项目并不是终点，而是整个论文分析 / AI 工程项目中的一个重要阶段。

当前它处于这样的角色：

- 作为 **LangChain 版最小 RAG 闭环**
- 用于补齐 **企业岗位中高频出现的框架关键词**
- 作为后续扩展 **Agent / LangGraph / Web Demo** 的基础版本

换句话说：

> 当前重点是先把一个可运行、可解释、可展示的 LangChain RAG 做出来，  
> 然后再逐步扩展到更复杂的 Agent 和工作流能力。

---

## 10. Future Work

后续计划扩展的方向包括：

- [ ] 支持 PDF 直接加载
- [ ] 支持多论文检索与对比
- [ ] 增加会话管理
- [ ] 引入 Tool Calling
- [ ] 引入 Router / Agent 调度
- [ ] 用 LangGraph 实现多步骤工作流
- [ ] 接入 Django / FastAPI 做成可展示的 Web 应用
- [ ] 增加日志、异常处理和配置管理的框架版实现

---

## 11. What This Project Tries to Demonstrate

这个项目希望体现的不是“会调一个框架 API”，而是：

- 理解 RAG 的核心流程
- 能用框架快速实现最小可用闭环
- 知道框架封装背后的工程逻辑
- 能把项目做成可展示、可讲解、可扩展的工程原型

对于 AI 工程 / RAG / Agent 方向的学习和求职来说，这比单纯堆功能更重要。

---

## 12. Notes

- 本项目当前强调 **最小闭环**，不追求一次性实现所有复杂功能
- 适合作为个人学习项目、博客素材和求职展示项目
- 后续随着功能扩展，README 会继续更新

---

## 13. License

This project is for learning, experimentation, and engineering practice.
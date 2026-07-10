# 个性化资源生成与学习多智能体系统

第十五届中国软件杯大学生软件设计大赛 · A3 赛题
出题企业：科大讯飞股份有限公司

---

## 项目简介

本系统面向高等教育场景，通过 12 个 LangGraph Agent 协同为学生自动生成个性化学习资源，并根据学习行为动态更新学生画像和学习路径。提交包内置**《动手学深度学习》**课程知识库作为演示数据集。

核心能力：
- 对话式学习画像构建（8 个维度，自然语言采集，无需填表）
- 7 种个性化资源自动生成：知识讲解文档、思维导图、练习题、代码案例、要点总结、**p5.js 教学动画**、知识图谱
- 基于知识图谱的学习路径规划与资源推送
- RAG 增强生成，内容溯源至课程知识库，防止幻觉
- 个性化学习计划表生成（基于知识点掌握度 + 遗忘曲线自动排程）
- 学习小助手（番茄钟专注、激励语、微型对话、复习提醒）
- 邮件服务（注册验证、学习报告推送）
- 语音输入（智能对话支持语音转文字）

---

## 技术架构

```
Aurora UI 前端
        ↕ REST API
FastAPI 后端
  ├── LangGraph 多智能体编排
  ├── RAG 检索层（PostgreSQL pgvector + HNSW 索引）
  ├── 邮件服务
  └── 数据层（SQLAlchemy 2.0 async + PostgreSQL + Alembic 迁移）
```

**主要依赖**

| 层次 | 技术 |
|------|------|
| 前端 | HTML/CSS/JS、Aurora UI 设计系统、p5.js (动画) |
| 后端 | FastAPI、Uvicorn |
| Agent 编排 | LangGraph |
| LLM | OpenAI 兼容大模型服务 |
| Embedding | OpenAI 兼容 Embedding 服务 |
| 向量检索 | PostgreSQL + pgvector (HNSW 索引) |
| 关系库 | PostgreSQL + asyncpg |
| 邮件 | aiosmtplib + Jinja2 模板 |

---

## 运行环境依赖

| 依赖 | 版本要求 | 用途 |
|------|---------|------|
| **Python** | 3.10+ | 运行后端 |
| **PostgreSQL** | 16/17 + **pgvector** 扩展 | 关系数据 + 向量检索（提交包用 Docker 免安装） |
| **Docker Desktop** | 任意较新版本 | 一键启动数据库 |
| **OpenAI 兼容 LLM 服务的 API Key** | — | LLM 生成 + 文本向量化（provider 可在 `configs/config.yaml` 一键切换） |

Python 依赖统一由 `requirements.txt` 管理，核心依赖如下（完整清单见 `requirements.txt`）：

| 用途 | 依赖 |
|------|------|
| Web | FastAPI + Uvicorn |
| Agent 编排 | LangGraph + LangChain |
| 异步数据层 | SQLAlchemy 2.0 + asyncpg + Alembic |
| 向量客户端 | pgvector |
| 文档解析 | pymupdf4llm + mammoth |
| 邮件 | aiosmtplib |
| 认证 | bcrypt + PyJWT |
| 中文分词 | jieba |
| 日志 | loguru |

---

## 快速启动（Docker 一键部署）

提交包内置预建数据库快照 `submission_seed.sql`（含《动手学深度学习》知识库的 4561 个向量分块 + 演示账号 + 学生画像 + 知识图谱 + 学习路径），配合 `docker-compose.yml` 免去手动建库、装 pgvector、跑迁移、等待入库的全部步骤。

### 1. 启动数据库

```bash
docker compose up -d
```

首次启动会自动拉取 `pgvector/pgvector:pg17` 镜像并导入 `submission_seed.sql`（需等待半分钟至一分钟）。`docker compose ps` 显示 `healthy` 即就绪。

### 2. 配置环境变量

```bash
cp .env.example .env
```

`.env.example` 中的 `DATABASE_URL` 已与 `docker-compose.yml` 对齐，**无需修改**。只需填写：

```bash
LLM_API_KEY=<所用 LLM 服务的 API Key>   # LLM + Embedding 共用（默认 provider，见 configs/config.yaml）
JWT_SECRET=<任意随机字符串> 
```

邮件服务为可选（`SMTP_*`），不配置时自动降级为本地文件模式（HTML 存入 `debug_emails/`）。

### 3. 安装 Python 依赖并启动

```bash
pip install -r requirements.txt
uvicorn backend.main:app --port 8000
```

由于数据库快照中提供了用于测试验证的知识库，后端会检测到向量库非空，**跳过自动索引**。

### 4. 访问系统

浏览器打开 `http://localhost:8000/app`，使用演示账号登录：

```
用户名：demo
密  码：demo1234
```

登录后仪表盘、学生画像、知识图谱、学习路径均已预置演示数据。可直接进入「资源生成」选择知识点（如"卷积神经网络"）现场生成学习文档 / 思维导图 / 练习题 / 教学动画，并观察 RAG 检索溯源至《动手学深度学习》原文。

---

## 手动部署

不使用 Docker 快照时，可自行准备 PostgreSQL：

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境

```bash
cp .env.example .env
```

必需环境变量：

```bash
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/softbei
LLM_API_KEY=your_api_key      # 所选 LLM Provider 的 API Key（OpenAI 兼容接口）
JWT_SECRET=your_jwt_secret_here
```

可选环境变量（邮件服务，不配置则降级为本地文件模式）：

```bash
SMTP_HOST=smtp.example.com
SMTP_PORT=587
SMTP_USERNAME=your_email
SMTP_PASSWORD=your_password
SMTP_FROM=noreply@example.com
```

### 3. 初始化数据库

```bash
# 确保 PostgreSQL 已启用 pgvector 扩展
# CREATE EXTENSION IF NOT EXISTS vector;
alembic upgrade head
```

### 4. 启动服务

```bash
uvicorn backend.main:app --reload --port 8000
```

启动后访问 `http://localhost:8000/app`。**首次启动时若向量库为空**，系统会自动递归索引 `knowledge_base/` 目录下的全部课程知识库（如 `knowledge_base/深度学习/`），子目录名即作为课程标签写入检索溯源元数据。首次索引需向 Embedding 服务发起数千次向量化请求，约需十余分钟，请耐心等待日志输出完成。

> 详细的评委导入与测试说明见 `评委导入与测试说明.md`。

---

## 项目结构

```
softbei/
├── backend/
│   ├── main.py              # FastAPI 入口
│   ├── agents/
│   │   ├── graph.py         # LangGraph 状态机
│   │   ├── profile_agent.py # 画像提取 + 完整性路由
│   │   ├── planner_agent.py # 意图分析，决定资源类型和知识点
│   │   ├── doc_agent.py     # 知识讲解文档生成
│   │   ├── mindmap_agent.py # 思维导图生成
│   │   ├── quiz_agent.py    # 练习题生成
│   │   ├── code_agent.py    # 代码案例生成
│   │   ├── summary_agent.py # 要点总结生成
│   │   ├── anim_agent.py    # p5.js 教学动画生成
│   │   ├── kg_agent.py      # 知识图谱构建
│   │   ├── safety_agent.py  # 内容安全校验
│   │   ├── clarify_agent.py # 追问澄清
│   │   └── recommend_agent.py # 后续学习推荐
│   ├── rag/
│   │   ├── loader.py        # PDF/DOCX/Markdown/TXT 解析
│   │   ├── indexer.py       # 向量化入库
│   │   └── retriever.py     # 混合检索 + 引用格式化
│   ├── services/
│   │   ├── llm.py           # LLM 统一调用层
│   │   ├── profile.py       # 画像 CRUD
│   │   ├── generation.py    # 资源生成任务调度
│   │   ├── resource.py      # 资源管理
│   │   ├── chat_history.py  # 多轮对话历史管理
│   │   └── study_plan/      # 学习计划表生成模块
│   │       ├── collector.py     # 知识点 + 掌握度收集
│   │       ├── sequencer.py     # LLM 排序
│   │       ├── scheduler.py     # 时间槽分配
│   │       ├── resource_linker.py # 关联已有资源
│   │       └── service.py       # 对外服务入口
│   ├── email/
│   │   ├── sender.py        # 异步 SMTP 发送器
│   │   ├── templates.py     # 邮件内容模板
│   │   └── utils.py         # 验证码生成等工具
│   ├── evaluation/          # RAG 四层评估体系
│   ├── db/
│   │   ├── models.py        # 20 张 ORM 表
│   │   ├── crud.py          # 通用异步 CRUD
│   │   └── database.py      # 异步连接池
│   ├── templates/email/     # HTML 邮件模板
│   └── models/schemas.py    # Pydantic v2 模型
├── frontend/
│   ├── index.html           # 主页
│   ├── auth.html            # 登录注册
│   ├── chat.html            # 智能对话
│   ├── profile.html         # 学生画像
│   ├── generate.html        # 资源生成
│   ├── pathway.html         # 知识图谱与学习路径
│   ├── library.html         # 资源库浏览
│   ├── evaluate.html        # 测验与评估
│   ├── forgot-password.html # 忘记密码
│   ├── reset-password.html  # 重置密码
│   ├── verify-email.html    # 邮箱验证
│   └── assets/
│       ├── aurora.css       # Aurora UI 设计系统
│       ├── aurora-bridge.css # Aurora 桥接样式
│       ├── assistant.js/css # 学习小助手
│       ├── guide.js         # 新用户引导
│       ├── anim-runtime.js  # p5.js 动画沙箱运行时
│       ├── nav.js           # 导航控制
│       ├── api.js           # API 调用层
│       ├── tracker.js       # 页面停留时长追踪
│       └── ...              # sidebar, command, dialog, toast 等
├── knowledge_base/
│   └── 深度学习/            # 示例课程知识库（《动手学深度学习》教材）
├── configs/
│   ├── config.yaml          # 运行时配置
│   └── prompts.yaml         # Agent 系统提示词
├── migrations/              # Alembic 数据库迁移
└── tests/
    ├── test_video_search.py # 视频搜索测试
    ├── test_study_plan.py   # 学习计划表测试
    └── test_email.py        # 邮件服务测试
```

---

## Agent 流水线

每次用户发送消息，LangGraph 按以下拓扑执行：

```
profile_agent
  ├─ 画像不足 → 生成追问 → END   （多轮对话累积画像）
  └─ 画像足够 → planner_agent
                ↙  ↙  ↙  ↙  ↘  ↘
           doc mindmap quiz code anim summary
                ↘  ↘  ↘  ↘  ↙  ↙
                 safety_agent
                      ↓
               recommend_agent → END
```

特殊路由：
- `clarify_agent` — 直接 → END（无需安全审计）
- `kg_agent` — 跳过 safety，直接 → `recommend_agent`

`profile_agent` 在每轮对话中持续提取和累积画像字段，画像达到最低要求（有学习目标或知识基础信息）后才放行到资源生成流程。

---

## 学生画像功能

- 8 维度画像：学习目标、知识基础、学习风格、时间安排等
- 学习效果评估：答题正确率自动更新掌握度
- 学习行为分析：近 30 天时长/次数统计
- 能力雷达图 + 遗忘曲线可视化
- 各知识点掌握程度：综合答题正确率 + 时间衰减
- 最近学习动态：支持跳转到对应资源

---

## 运行测试

```bash
# 运行全部测试
pytest tests/ -v

# 视频搜索测试
pytest tests/test_video_search.py -v

# 学习计划表测试
pytest tests/test_study_plan.py -v

# RAG 黄金数据集评估（需要数据库 + LLM）
python -m backend.evaluation.golden_dataset --run
```

---

## AI 工具使用说明

**系统运行所依赖的 AI 服务**（均通过 OpenAI 兼容协议接入，可在 `configs/config.yaml` 一键切换）：

- **大语言模型 (LLM)**：驱动全部 Agent 的内容生成、画像提取与意图分析
- **文本向量化 (Embedding)**：用于 RAG 检索，模型与维度可配置

内置支持的 LLM Provider（改 `llm.provider` 即可切换，各家可配独立 API Key）：

| provider | 说明 |
|----------|------|
| `spark` | 讯飞星火 |
| `deepseek` | DeepSeek |
| `qwen` | 通义千问 |
| `openai` | OpenAI 及任意兼容其协议的服务 |

**开发阶段使用的 AI 编码工具：**

- Copilot：辅助代码开发、架构设计
- 豆包：辅助代码开发、架构设计、提示词优化
- Figma Make：制作演示材料

---

## 提交物清单

- [x] 演示 PPT
- [x] 完整源码 + 知识库数据集 + 配置文件
- [x] 智能体演示视频（≤7 分钟）
- [x] 配套文档（需求分析 + 技术开发说明）
- [x] AI Coding 工具使用说明

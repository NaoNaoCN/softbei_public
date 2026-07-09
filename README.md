# 个性化资源生成与学习多智能体系统

第十五届中国软件杯大学生软件设计大赛 · A3 赛题
出题企业：科大讯飞股份有限公司

---

## 项目简介

本系统面向高等教育场景，以**人工智能导论**课程为切入点，通过 12 个 LangGraph Agent 协同为学生自动生成个性化学习资源，并根据学习行为动态更新学生画像和学习路径。

核心能力：
- 对话式学习画像构建（8 个维度，自然语言采集，无需填表）
- 6 种个性化资源自动生成：知识讲解文档、思维导图、练习题、代码案例、要点总结、**p5.js 教学动画**
- 基于知识图谱的学习路径规划与资源推送
- RAG 增强生成，内容溯源至课程知识库，防止幻觉
- 个性化学习计划表生成（基于知识点掌握度 + 遗忘曲线自动排程）
- 学习小助手（番茄钟专注、激励语、微型对话、复习提醒）
- 邮件服务（注册验证、密码重置、学习报告推送）
- 语音输入（智能对话支持语音转文字）

---

## 技术架构

```
Aurora UI 前端（HTML/CSS/JS，12 页面）
        ↕ REST API
FastAPI 后端
  ├── LangGraph 多智能体编排（12 个 Agent）
  ├── RAG 检索层（PostgreSQL pgvector + HNSW 索引）
  ├── 邮件服务（aiosmtplib 异步发送）
  └── 数据层（SQLAlchemy 2.0 async + PostgreSQL + Alembic 迁移）
```

**主要依赖**

| 层次 | 技术 |
|------|------|
| 前端 | HTML/CSS/JS、Aurora UI 设计系统、p5.js (动画) |
| 后端 | FastAPI、Uvicorn |
| Agent 编排 | LangGraph |
| LLM | 通义千问 qwen3.6-plus (DashScope API，OpenAI 兼容) |
| Embedding | DashScope text-embedding-v4 (1024维) |
| 向量检索 | PostgreSQL + pgvector (HNSW 索引，余弦相似度) |
| 关系库 | PostgreSQL + asyncpg |
| 邮件 | aiosmtplib + Jinja2 模板 |

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境

复制 `.env.example` 为 `.env` 并填写实际值：

```bash
cp .env.example .env
```

必需环境变量：

```bash
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/softbei
LLM_API_KEY=your_dashscope_api_key
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

启动后访问 `http://localhost:8000/app` 即可使用。首次启动时，若知识库为空，系统会自动索引 `knowledge_base/ai_intro/` 目录。

---

## 项目结构

```
softbei/
├── backend/
│   ├── main.py              # FastAPI 入口
│   ├── agents/
│   │   ├── graph.py         # LangGraph 状态机（12 节点）
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
│   │   ├── loader.py        # PDF/DOCX/Markdown/TXT 解析（pymupdf4llm/mammoth）
│   │   ├── indexer.py       # 向量化入库（DashScope embedding，父子切割）
│   │   └── retriever.py     # 混合检索（向量+关键词 RRF 融合）+ 引用格式化
│   ├── services/
│   │   ├── llm.py           # LLM 统一调用层（多 provider + 重试）
│   │   ├── profile.py       # 画像 CRUD
│   │   ├── generation.py    # 资源生成任务调度
│   │   ├── resource.py      # 资源管理
│   │   ├── chat_history.py  # 多轮对话历史管理
│   │   └── study_plan/      # 学习计划表生成模块
│   │       ├── collector.py     # 知识点 + 掌握度收集
│   │       ├── sequencer.py     # LLM 排序（先修关系）
│   │       ├── scheduler.py     # 时间槽分配
│   │       ├── resource_linker.py # 关联已有资源
│   │       └── service.py       # 对外服务入口
│   ├── email/
│   │   ├── sender.py        # 异步 SMTP 发送器（自动重试）
│   │   ├── templates.py     # 邮件内容模板
│   │   └── utils.py         # 验证码生成等工具
│   ├── evaluation/          # RAG 四层评估体系
│   ├── db/
│   │   ├── models.py        # 20 张 ORM 表
│   │   ├── crud.py          # 通用异步 CRUD
│   │   └── database.py      # 异步连接池
│   ├── templates/email/     # HTML 邮件模板
│   └── models/schemas.py    # 所有 Pydantic v2 模型
├── frontend/
│   ├── index.html           # 主页（Aurora UI）
│   ├── auth.html            # 登录注册
│   ├── chat.html            # 智能对话（支持语音输入）
│   ├── profile.html         # 学生画像（雷达图/遗忘曲线/行为统计）
│   ├── generate.html        # 资源生成
│   ├── pathway.html         # 知识图谱 + 学习路径
│   ├── library.html         # 资源库浏览（含动画预览）
│   ├── evaluate.html        # 测验与评估
│   ├── forgot-password.html # 忘记密码
│   ├── reset-password.html  # 重置密码
│   ├── verify-email.html    # 邮箱验证
│   └── assets/
│       ├── aurora.css       # Aurora UI 设计系统
│       ├── aurora-bridge.css # Aurora 桥接样式
│       ├── assistant.js/css # 学习小助手（番茄钟/激励/对话）
│       ├── guide.js         # 新用户 11 步引导
│       ├── anim-runtime.js  # p5.js 动画沙箱运行时
│       ├── nav.js           # 导航控制
│       ├── api.js           # API 调用层
│       ├── tracker.js       # 页面停留时长追踪
│       └── ...              # sidebar, command, dialog, toast 等
├── knowledge_base/
│   └── ai_intro/            # 人工智能导论课程知识库
├── configs/
│   ├── config.yaml          # 运行时配置（支持 ${ENV_VAR} 语法）
│   └── prompts.yaml         # 所有 Agent 系统提示词
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

本项目开发过程中使用了以下 AI 相关工具：

- **通义千问大模型 API (DashScope)**：系统主力 LLM，用于所有 Agent 的内容生成、画像提取和意图分析
- **DashScope text-embedding-v4**：文本向量化，用于 RAG 检索
- **Claude Code**：辅助代码开发、架构设计、提示词优化

支持的 LLM Provider（通过配置切换）：
- qwen (通义千问，当前默认)
- spark (讯飞星火)
- deepseek
- openai

---

## 提交物清单

- [ ] 演示 PPT
- [ ] 完整源码 + 知识库数据集 + 配置文件
- [ ] 智能体演示视频（≤7 分钟）
- [ ] 配套文档（需求分析 + 技术开发说明）
- [ ] AI Coding 工具使用说明

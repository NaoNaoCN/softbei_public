# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Context

第十五届中国软件杯 A3 赛题 — 个性化资源生成与学习多智能体系统。面向高等教育场景，通过 12 个 LangGraph Agent 协同为学生自动生成个性化学习资源（含动画演示）。

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run database migrations (first time or after schema changes)
alembic upgrade head

# Run backend (frontend at http://localhost:8000/app)
# On startup, if the knowledge base is empty, the configured KB dir is auto-indexed.
uvicorn backend.main:app --reload --port 8000

# Reindex on demand: clear the document_chunk table, then restart (auto-index runs),
# or call index_directory() from backend.rag.indexer (no standalone CLI entrypoint).

# Run all tests
pytest tests/ -v

# Run a specific test file (current tests cover video search)
pytest tests/test_video_search.py -v

# Run a single test function
pytest tests/test_video_search.py::TestExtractSearchKeywords::test_extracts_technical_term -v

# Run RAG golden evaluation (requires DB + LLM)
python -m backend.evaluation.golden_dataset --run
```

## Architecture

**Stack:** FastAPI (async) + HTML/CSS/JS frontend + LangGraph agents + PostgreSQL with **pgvector** extension (in-database cosine search via `<=>`, HNSW index) + SQLAlchemy 2.0 (async ORM)

**LLM & Config:** Provider/model configured in `configs/config.yaml` via `${ENV_VAR}` substitution. Currently uses `qwen3.6-plus-2026-04-02` (provider `qwen`) via DashScope. Multi-provider support (spark/deepseek/qwen/openai) in `backend/services/llm.py`. Agent system prompts in `configs/prompts.yaml`. Config is a module-level singleton: `from backend.config import config`.

**Required env vars** (see `.env.example`): `LLM_API_KEY`, `JWT_SECRET`, `DATABASE_URL` (PostgreSQL, e.g. `postgresql+asyncpg://user:pass@localhost:5432/softbei`). **Optional:** `SMTP_HOST`, `SMTP_PORT`, `SMTP_USERNAME`, `SMTP_PASSWORD`, `SMTP_FROM` (邮件服务，不配置则降级为本地文件模式).

### Agent Pipeline (LangGraph)

`backend/agents/graph.py` defines a 12-node StateGraph. All agents share `AgentState` (Pydantic `BaseModel` in `backend/models/schemas.py`). The `db` session is injected via `config={"configurable": {"db": db}}`. Routing is **conditional, not parallel fan-out** — each run dispatches to a single generation agent based on `intent_type` + `resource_type`.

1. `profile_agent` — extracts/accumulates student profile. Routes to END (ask follow-up) or `planner_agent` (profile sufficient).
2. `planner_agent` — analyzes intent (`route_by_resource_type`), routes to one of the generation agents, `kg_agent`, `clarify_agent`, or `recommend_agent` (fallback).
3. `clarify_agent` — asks a follow-up question, then → END (skips safety).
4. `doc_agent`, `mindmap_agent`, `quiz_agent`, `code_agent`, `summary_agent`, `anim_agent` — generation agents, each → `safety_agent`.
5. `kg_agent` — knowledge-graph node, → `recommend_agent` (skips safety).
6. `safety_agent` — content safety check → `recommend_agent`.
7. `recommend_agent` → END.

`graph.invoke()` / `stream_invoke()` load the student profile and multi-turn `chat_history` into the initial state, and clear the RAG retrieval cache before each run. RAG evaluation data is collected post-run and an async LLM-as-Judge may be triggered by sampling.

Resource generation is triggered via `POST /generate` (single) / `POST /generate/batch` / `POST /generate/smart`, runs as a background task in `backend/services/generation.py`, and persists results to `ResourceMeta` + `QuizItem` tables.

### Database Layer

- **PostgreSQL** via `asyncpg` driver. Schema managed by **Alembic** (`migrations/`).
- ORM models in `backend/db/models.py`: `User`, `StudentProfile`, `ProfileHistory`, `ChatSession`, `ChatMessage`, `DocumentChunk` (pgvector store), `KGNode`, `KGEdge`, `ResourceMeta`, `GenerationBatch`, `GenerationTask`, `KGBuildTask`, `QuizItem`, `QuizAttempt`, `LearningPath`, `LearningPathItem`, `LearningRecord`, `StudyPlan`, `StudyPlanItem`, `EmailVerification`
- Generic async CRUD in `backend/db/crud.py`: `insert`, `select`, `select_one`, `update_by_id`, `delete_by_id`, `count`. Supports relation loading via `loadRelations` param.
- Chat messages stored in static `chat_message` table.
- Connection pool: `pool_size=10 + max_overflow=20`, `pool_pre_ping=True`.

### RAG Pipeline

`backend/rag/loader.py` → `indexer.py` → `retriever.py`. Loader parses PDF/DOCX/Markdown/TXT into `TextChunk` objects (PDF→Markdown via `pymupdf4llm`, DOCX→Markdown via `mammoth`, preserving heading/table/list structure). Indexer vectorizes with DashScope `text-embedding-v4` (1024-dim; local BGE-M3 path deprecated, `embedding.use_spark: true`) and writes to the PostgreSQL `document_chunk` table (pgvector column + `to_tsvector` full-text column). Vector search runs **in-database** via the `<=>` cosine operator over an **HNSW** index (`hnsw.ef_search` tunable per query). Retriever does hybrid search (vector + jieba keyword via `ts_rank`) merged with RRF fusion, keyword-overlap re-ranking, diversity-aware reordering, and citation formatting.

Key features:
- **Parent-child chunking**: child chunks for precise retrieval, parent chunks provide full context
- **Hybrid retrieval**: vector semantic search + jieba keyword search, merged via RRF
- **Query rewrite**: decontextualization + profile-aware expansion + optional multi-query
- **Diversity-aware ordering**: round-robin by section to avoid top-k all from same sub-topic

### RAG Evaluation System

Four-layer evaluation architecture in `backend/evaluation/`:

| Layer | File | What | Cost |
|-------|------|------|------|
| L1 Health Check | `health_check.py` | Vector DB connectivity, index stats | <5ms |
| L2 Retrieval | `collector.py` + `judge.py` | P@5, Recall@5, MRR, NDCG@5, Hit Rate | LLM Judge |
| L3 Generation | `judge.py` | Faithfulness, Completeness, Citation Accuracy, Hallucination | LLM Judge |
| L4 Golden Eval | `golden_dataset.py` | Full-pipeline eval on hand-labeled queries | LLM Judge |

**Golden eval workflow:**
1. `golden_queries.yaml` — 15 hand-crafted queries across concept/algorithm/comparison/practice categories, each with `expected_aspects` for completeness scoring
2. `golden_dataset.py` — runs queries through RAG pipeline → generates answers → judges faithfulness/completeness
3. Report written to `logs/golden_eval_*.md`

Reference docs in `docs/xqt/rag/`: 优化效果报告.md, 生成层优化方案.md, RAG评估系统.md, RAG评估指标说明.md, 系统提示词书写原则.md

### Prompt Engineering Conventions

All Agent system prompts live in `configs/prompts.yaml`. Template variables: config-level (`{min_recommendations}`) resolved by `config.py`; runtime (`{context}`, `{kp_name}`) filled by agents via `.format()`.

**Hierarchy for rules** (from `docs/xqt/rag/系统提示词书写原则.md`):

| Priority | Keyword | When to use |
|----------|---------|-------------|
| Highest | **NEVER** | Irreversible harm: fabrication, hallucination, false citations |
| High | **IMPORTANT** | Must-do quality gates: cite sources, verify against references |
| Medium | Do NOT | Default prohibition, exceptions allowed |
| Low | Avoid | Preference, can be overridden |

**Standard prompt structure** every generation agent follows:
```
# Role — what the agent IS and IS NOT
# Rules — NEVER/IMPORTANT hierarchy, each with a reason
# Pre-generation Check — mental checklist before output (not included in output)
# Output — format specification
```

**Key conventions:**
- Always use `role: "system"` for generation prompts (higher LLM adherence than `role: "user"`)
- Every prohibition must explain *why* — prevents the LLM from rationalizing violations
- `[n]` citation markers mandatory in all generated content; `💡 补充` prefix for knowledge beyond references
- Empty context handling: declare "暂无参考资料" rather than silently fabricating
- Anti-fabrication is the top priority — faithfulness beats completeness

### Frontend

HTML/CSS/JS frontend in the `frontend/` directory, served via FastAPI StaticFiles at `/app`. **Aurora UI** design system (`aurora.css` + `aurora-bridge.css`). Pages: `index`, `auth`, `chat`, `profile`, `generate`, `pathway`, `library`, `evaluate`, `forgot-password`, `reset-password`, `verify-email`. API layer in `frontend/assets/api.js`; other assets: `sidebar.js`, `command.js`, `dialog.js`, `toast.js`, `tracker.js` (page dwell-time tracking), `button.js`, `shortcut.js`, `nav.js`, `guide.js` (11-step onboarding), `assistant.js` + `assistant.css` (学习小助手：番茄钟/激励/微型对话), `anim-runtime.js` (p5.js 动画沙箱运行时). Auth: JWT stored in localStorage, user_id passed as query param. Voice input supported in chat.

### Email Service

`backend/email/` module: async SMTP sender (`aiosmtplib` + `tenacity` retry), Jinja2 HTML templates in `backend/templates/email/`. Features: email verification, password reset, learning report. SMTP 未配置时降级为本地文件模式（保存 HTML 到 `debug_emails/`）。

### Study Plan Service

`backend/services/study_plan/` module: generates personalized learning schedules. Pipeline: `collector.py` (gather KPs + mastery data) → `sequencer.py` (LLM-powered prerequisite ordering) → `scheduler.py` (time-slot allocation) → `resource_linker.py` (link existing resources to plan items). ORM: `StudyPlan` + `StudyPlanItem` tables. Config in `configs/config.yaml` under `study_plan:` section.

## Key Conventions

- **Pydantic v2** for all schemas (`backend/models/schemas.py`)
- **pytest-asyncio** with `asyncio_mode = auto` — no `@pytest.mark.asyncio` needed
- All DB operations are async (`async with get_session() as session`)
- Agent pattern: receive `AgentState` (Pydantic model) → call LLM → return an updated copy via `state.model_copy(update={...})`
- **Config is a module-level singleton**: `from backend.config import config`. Never instantiate a second Config object.
- **Prompts are externalized**: never hardcode system prompts in agent `.py` files — all prompts live in `configs/prompts.yaml` and are accessed via `config.prompts["agents"][agent_name]["system_prompt"]`.

## Naming Conventions — ORM ↔ Schema Alignment

**`backend/db/models.py` is the single source of truth.** Schema field names must exactly match ORM field names. Never invent aliases in schemas; never use `model_validator` for field name mapping. See the quick reference below.

### Quick reference

| Category | Correct | Forbidden |
|----------|---------|-----------|
| Creation timestamp | `created_at` | `submitted_at`, `recorded_at`, `added_at` |
| Update timestamp | `updated_at` | `modified_at`, `last_updated` |
| Primary key in schema | `id` | `task_id`, `record_id`, `path_id` |
| Error text | `error_message` | `error_msg`, `error`, `err` |
| Entity title | `title` (match ORM) | renaming to `name` in schema |
| Node type field | `node_type` (match ORM) | renaming to `type` in schema |

**When adding a new model**, verify:
1. All timestamps use `created_at` / `updated_at`
2. Every schema field name matches the ORM column name exactly
3. No `model_validator` used for field renaming
4. `main.py` manual schema construction references ORM field names directly
5. Frontend `.get("field_name")` keys match actual API response fields

## Logging Conventions

**Framework:** loguru (`from loguru import logger`). Configuration in `backend/logging_config.py`.

### Log levels

| Level | When to use | Example |
|-------|------------|---------|
| `TRACE` | Extreme detail (function args, step-by-step return values) | `logger.trace("query_vector shape={}", vec.shape)` |
| `DEBUG` | Developer debugging info | `logger.debug("[DocAgent] LLM prompt length={}", len(prompt))` |
| `INFO` | Key business process milestones | `logger.info("[DocAgent] doc generated, kp={}, len={}", kp, n)` |
| `SUCCESS` | Operation completed successfully | `logger.success("[Generation] batch {} done, {}s", id, t)` |
| `WARNING` | Recoverable errors, degradation | `logger.warning("[LLM] retry {} due to {}", n, reason)` |
| `ERROR` | Unrecoverable errors needing attention | `logger.error("[Database] connection pool exhausted")` |
| `CRITICAL` | System-level failures | `logger.critical("[App] startup failed, port {} in use", p)` |

### Message format

Use **lazy evaluation** with `{}` placeholders. Do NOT use f-strings:

```python
# ✅ Correct — deferred formatting, runs only if level is reached
logger.info("[DocAgent] kp={}, draft_len={}", kp_name, len(draft))

# ❌ Wrong — f-string always evaluated
logger.info(f"[DocAgent] kp={kp_name}")
```

### Agent prefix

All agent log messages use a `[AgentName]` prefix for grep-ability.

### Exception logging

Use `logger.exception()` (auto-includes traceback) instead of `logger.error()` for exceptions:

```python
try:
    ...
except Exception as e:
    logger.exception("[AgentName] operation failed: {}", e)
```

### trace_id

A `trace_id` is automatically injected into every log record's `extra` via the patcher in `logging_config.py`. Do NOT manually add `[trace_id]` in log messages — it already appears in the format string.

"""
backend/agents/graph.py
LangGraph 主状态机：定义节点、边（含条件路由）并编译图。
"""

from __future__ import annotations

from typing import Any
from sqlalchemy.ext.asyncio import AsyncSession

from langgraph.graph import END, StateGraph

from backend.agents import (
    clarify_agent,
    code_agent,
    anim_agent,
    doc_agent,
    kg_agent,
    mindmap_agent,
    planner_agent,
    profile_agent,
    quiz_agent,
    recommend_agent,
    safety_agent,
    summary_agent,
)
from backend.models.schemas import AgentState
from backend.services.profile import get_profile
from backend.services.chat_history import load_chat_history

# ----------------------------------------------------------
# 数据库会话注入辅助
# ----------------------------------------------------------

async def _run_with_db(node_func, state: AgentState, db: AsyncSession) -> AgentState:
    """
    通用包装器：如果 node_func 签名需要 db，则传递。
    LangGraph 节点函数签名为 (state,) 或 (state, config)。
    """
    import inspect
    sig = inspect.signature(node_func)
    params = list(sig.parameters.keys())
    if "db" in params:
        return await node_func(state, db)
    return await node_func(state)


# ----------------------------------------------------------
# 图构建
# ----------------------------------------------------------

def build_graph() -> StateGraph:
    """
    构建并返回编译后的 LangGraph 状态机。

    节点拓扑（条件路由，非并行）：

    START → profile_agent
              ├─ (画像不足) → END
              └─ (画像足够) → planner_agent
                              │ (先判断 intent_type)
                              ├─ intent="clarify" → clarify_agent → END
                              │
                              │ (intent="generate", 按 resource_type 路由)
                              ├─ doc_agent ─────┐
                              ├─ mindmap_agent ─┤
                              ├─ quiz_agent ────┼─→ safety_agent → recommend_agent → END
                              ├─ code_agent ────┤
                              ├─ summary_agent ─┘
                              ├─ kg_agent ──────────────────────→ recommend_agent → END
                              └─ recommend_agent → END           ← 兜底路由
    """
    graph = StateGraph(AgentState)

    # -- 注册节点 --
    graph.add_node("profile_agent", profile_agent.run)
    graph.add_node("planner_agent", planner_agent.run)
    graph.add_node("doc_agent", doc_agent.run)
    graph.add_node("mindmap_agent", mindmap_agent.run)
    graph.add_node("quiz_agent", quiz_agent.run)
    graph.add_node("code_agent", code_agent.run)
    graph.add_node("anim_agent", anim_agent.run)
    graph.add_node("summary_agent", summary_agent.run)
    graph.add_node("safety_agent", safety_agent.run)
    graph.add_node("recommend_agent", recommend_agent.run)
    graph.add_node("kg_agent", kg_agent.run)
    graph.add_node("clarify_agent", clarify_agent.run)

    # -- 起始节点 --
    graph.set_entry_point("profile_agent")

    # profile → 条件路由（画像不足则直接 END，足够则进 planner）
    graph.add_conditional_edges(
        "profile_agent",
        profile_agent.route_after_profile,
        {
            "planner_agent": "planner_agent",
            END: END,
        },
    )

    # planner → 条件路由（按 intent_type + resource_type）
    graph.add_conditional_edges(
        "planner_agent",
        planner_agent.route_by_resource_type,
        {
            "doc_agent": "doc_agent",
            "mindmap_agent": "mindmap_agent",
            "quiz_agent": "quiz_agent",
            "code_agent": "code_agent",
            "anim_agent": "anim_agent",
            "summary_agent": "summary_agent",
            "kg_agent": "kg_agent",
            "recommend_agent": "recommend_agent",
            "clarify_agent": "clarify_agent",
        },
    )

    # 各生成 Agent → safety_agent
    for agent_name in ["doc_agent", "mindmap_agent", "quiz_agent", "code_agent", "anim_agent", "summary_agent"]:
        graph.add_edge(agent_name, "safety_agent")

    # safety → recommend → END
    graph.add_edge("safety_agent", "recommend_agent")
    graph.add_edge("kg_agent", "recommend_agent")  # KG 跳过 safety，直接到 recommend
    graph.add_edge("recommend_agent", END)
    graph.add_edge("clarify_agent", END)  # clarify 直接结束，无需 safety

    return graph.compile()


# 模块级全局图实例（FastAPI 启动时调用 build_graph() 初始化）
_compiled_graph = None


def get_graph() -> StateGraph:
    """返回已编译的图，若未初始化则抛出 RuntimeError。"""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


async def invoke(user_id: int, session_id: int, message: str, db: AsyncSession) -> AgentState:
    """
    执行一次完整的图推理，返回最终状态。

    :param user_id:   用户 ID（Snowflake BIGINT）
    :param session_id: 会话 ID（Snowflake BIGINT）
    :param message:   用户输入
    :param db:        数据库会话
    :return:           最终 AgentState
    """
    # 每次图推理开始前清除 RAG 检索缓存
    from backend.agents.utils import clear_retrieval_cache
    clear_retrieval_cache()

    existing_profile = await get_profile(user_id, db)

    # 加载多轮对话历史
    chat_history = await load_chat_history(session_id, db)

    initial_state = AgentState(
        user_id=user_id,
        session_id=session_id,
        user_message=message,
        profile=existing_profile,
        chat_history=chat_history,
    )

    result = await get_graph().ainvoke(
        initial_state,
        config={"configurable": {"db": db}},
    )
    final_state = AgentState(**result)

    # -- RAG 评估采集：记录生成结果 --
    _collect_generation_eval(final_state, session_id)

    # -- 异步 LLM-as-Judge 评估（采样触发）--
    _maybe_trigger_async_judge(final_state, session_id)

    return final_state


async def stream_invoke(user_id: int, session_id: int, message: str, db: AsyncSession):
    """
    流式执行图推理，逐步 yield AgentState 快照。
    供 FastAPI StreamingResponse 或 Streamlit 实时显示使用。
    """
    # 每次图推理开始前清除 RAG 检索缓存
    from backend.agents.utils import clear_retrieval_cache
    clear_retrieval_cache()

    existing_profile = await get_profile(user_id, db)

    # 加载多轮对话历史
    chat_history = await load_chat_history(session_id, db)

    initial_state = AgentState(
        user_id=user_id,
        session_id=session_id,
        user_message=message,
        profile=existing_profile,
        chat_history=chat_history,
    )
    async for event in get_graph().astream(
        initial_state,
        config={"configurable": {"db": db}},
    ):
        yield event


# ----------------------------------------------------------
# RAG 评估辅助
# ----------------------------------------------------------

def _collect_generation_eval(state: AgentState, session_id: int) -> None:
    """从 AgentState 中采集生成评估数据。"""
    try:
        from backend.config import config
        if not config.evaluation.enabled:
            return

        from backend.evaluation.collector import collector

        agent_type = state.resource_type.value if state.resource_type else ""
        draft_content = state.draft_content or ""
        safety_issues = state.metadata.get("safety_issues", []) if state.metadata else []

        # 读取各 Agent 记录的生成耗时（由 agent 写入 state.metadata）
        gen_latency = state.metadata.get("generation_latency_ms", 0.0) if state.metadata else 0.0

        # 读取 A/B 实验分组
        experiment_group = state.metadata.get("experiment_group") if state.metadata else None

        collector.record_generation(
            agent_type=agent_type,
            draft_length=len(draft_content),
            generation_latency_ms=gen_latency,
            safety_passed=state.safety_passed,
            safety_issues_count=len(safety_issues),
            safety_issues=safety_issues,
            experiment_group=experiment_group,
        )
    except Exception:
        pass  # 评估采集失败不应影响主流程


def _maybe_trigger_async_judge(state: AgentState, session_id: int) -> None:
    """按采样率决定是否触发异步 LLM-as-Judge 评估。"""
    import asyncio
    import re

    try:
        from backend.config import config
        if not config.evaluation.enabled:
            return

        from backend.evaluation.collector import collector
        from backend.evaluation.judge import get_judge
        from backend.rag.retriever import RetrievedChunk

        session_id_str = str(session_id)

        # 采样决策
        if not collector.decide_sample(session_id_str):
            return

        # 检查是否有足够的评估素材
        draft = state.draft_content or ""
        retrieved = state.retrieved_docs or []
        kp_name = state.kp_id or ""
        query = state.user_message or ""

        if not draft or not retrieved:
            return

        # 构建 RetrievedChunk 列表（从缓存中获取 chunk 元数据）
        retrieval_record = collector._current_retrieval
        if retrieval_record is None:
            return

        chunks = []
        for i, text in enumerate(retrieved):
            chunk = RetrievedChunk(
                chunk_id=retrieval_record.chunk_ids[i] if i < len(retrieval_record.chunk_ids) else f"chunk_{i}",
                text=text,
                score=retrieval_record.scores[i] if i < len(retrieval_record.scores) else 0.0,
                doc_id=retrieval_record.doc_ids[i] if i < len(retrieval_record.doc_ids) else "",
                source="",
            )
            chunks.append(chunk)

        async def _run_judge():
            try:
                judge = get_judge()
                experiment_group = state.metadata.get("experiment_group") if state.metadata else None
                safety_issues = state.metadata.get("safety_issues", []) if state.metadata else []
                result = await judge.evaluate_full(
                    query=query,
                    kp_name=kp_name,
                    retrieved_chunks=chunks,
                    generated_content=draft,
                    experiment_group=experiment_group,
                    safety_issues=safety_issues,
                )
                # 将评估结果写回 collector 的当前生成记录
                gen_record = collector._current_generation
                if gen_record:
                    gen_record.faithfulness_score = result.get("faithfulness_score", 0.0)
                    gen_record.hallucination_rate_val = result.get("hallucination_rate", 0.0)
                    gen_record.completeness_score = result.get("completeness_score", 0.0)
                    gen_record.concept_coverage = result.get("completeness_score", 0.0)
                    gen_record.relevance_labels = result.get("relevance_labels", [])
                    gen_record.faithfulness_statements = result.get("faithfulness_statements", [])
                    gen_record.completeness_aspects = result.get("completeness_aspects", [])
                collector.flush()

                from loguru import logger
                logger.info(
                    f"[Eval] async judge complete: faithfulness={result.get('faithfulness_score', 0):.2f}, "
                    f"completeness={result.get('completeness_score', 0):.2f}"
                )

                # 自动更新日报文件
                try:
                    from backend.evaluation.reporter import RAGReporter
                    reporter = RAGReporter()
                    reporter.save_daily_report(collector.get_recent_records(500))
                except Exception:
                    pass
            except Exception as e:
                from loguru import logger
                logger.warning(f"[Eval] async judge failed: {e}")

        asyncio.create_task(_run_judge())

    except Exception:
        pass  # Judge 触发失败不应影响主流程

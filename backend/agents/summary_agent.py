"""
backend/agents/summary_agent.py
SummaryAgent：生成知识点精简总结（适合复习的要点提炼）。
"""

from __future__ import annotations

from loguru import logger

from backend.config import config as app_config
from backend.models.schemas import AgentState
from backend.agents.utils import resolve_kp_name, retrieve_context, format_reference_list
from backend.services.llm import chat_completion
from langchain_core.runnables import RunnableConfig

from backend.config import prompts as _prompts

SYSTEM_PROMPT = _prompts.get("agents.summary.system_prompt")


async def run(state: AgentState, config: RunnableConfig = None) -> AgentState:
    """
    SummaryAgent 节点入口。

    职责：
    1. 检索相关文档
    2. 调用 LLM 生成复习总结 Markdown
    3. 写入 state.draft_content
    """
    kp_name = await resolve_kp_name(state, config)
    logger.info("[SummaryAgent] kp_name=%s", kp_name)

    # 检索相关文档（return_sources：拿到真实来源清单用于文末追加参考资料）
    context, retrieved_texts, sources = await retrieve_context(
        state, "SummaryAgent", config, return_sources=True
    )

    # 更新 retrieved_docs
    state = state.model_copy(update={"retrieved_docs": retrieved_texts})

    # 构造 prompt
    prompt = SYSTEM_PROMPT.format(context=context, kp_name=kp_name)

    try:
        draft = await chat_completion(
            [{"role": "system", "content": prompt}],
            temperature=app_config.agents.summary.temperature,
            max_tokens=app_config.agents.summary.max_tokens,
        )
        # 后处理：追加真实来源清单（编号与正文 [n] 对齐）
        draft += format_reference_list(sources)
        logger.info("[SummaryAgent] 总结生成成功，draft_len=%d", len(draft))
        state = state.model_copy(update={"draft_content": draft})
    except Exception as e:
        logger.error("[SummaryAgent] 生成失败: %s", e)
        state = state.model_copy(update={"draft_content": f"总结生成失败：{e}"})

    return state

"""
backend/agents/anim_agent.py
AnimAgent：基于 p5.js 生成教学动画 sketch（针对 anim-runtime 助手库的 defineAnimation 调用）。
"""

from __future__ import annotations

from loguru import logger

from backend.config import config as app_config
from backend.models.schemas import AgentState
from backend.agents.utils import resolve_kp_name, retrieve_context
from backend.services import profile as profile_svc
from backend.services.llm import chat_completion
from langchain_core.runnables import RunnableConfig

from backend.config import prompts as _prompts

SYSTEM_PROMPT = _prompts.get("agents.anim.system_prompt")


async def run(state: AgentState, config: RunnableConfig = None) -> AgentState:
    """
    AnimAgent 节点入口。

    职责：
    1. 检索相关文档
    2. 调用 LLM 生成 p5.js 动画 sketch（defineAnimation 调用）
    3. 写入 state.draft_content
    """
    kp_name = await resolve_kp_name(state, config)

    # 检索相关文档
    context, retrieved_texts = await retrieve_context(state, "AnimAgent")

    # 更新 retrieved_docs
    state = state.model_copy(update={"retrieved_docs": retrieved_texts})

    # 构建画像上下文
    profile_summary = ""
    if state.profile:
        try:
            profile_summary = await profile_svc.build_profile_context(state.profile)
        except Exception:
            profile_summary = "（暂无画像信息）"
    else:
        profile_summary = "（暂无画像信息）"

    # 构造 prompt
    prompt = SYSTEM_PROMPT.format(
        context=context,
        kp_name=kp_name,
        profile_summary=profile_summary,
    )

    try:
        draft = await chat_completion(
            [{"role": "system", "content": prompt}],
            temperature=app_config.agents.anim.temperature,
            max_tokens=app_config.agents.anim.max_tokens,
        )
        logger.info(
            "[anim_agent] draft_len={} has_fence={} preview={:.200}",
            len(draft), "```" in draft, draft,
        )
        state = state.model_copy(update={"draft_content": draft})
    except Exception as e:
        state = state.model_copy(update={"draft_content": f"动画生成失败：{e}"})

    return state

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
    """AnimAgent 节点入口。"""
    kp_name = await resolve_kp_name(state, config)

    context, retrieved_texts = await retrieve_context(state, "AnimAgent")

    state = state.model_copy(update={"retrieved_docs": retrieved_texts})

    profile_summary = ""
    if state.profile:
        try:
            profile_summary = await profile_svc.build_profile_context(state.profile)
        except Exception:
            profile_summary = "（暂无画像信息）"
    else:
        profile_summary = "（暂无画像信息）"

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

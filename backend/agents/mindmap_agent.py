from __future__ import annotations

import json

from loguru import logger

from backend.config import config as app_config
from backend.models.schemas import AgentState
from backend.agents.utils import resolve_kp_name, retrieve_context
from backend.services.llm import chat_completion
from langchain_core.runnables import RunnableConfig

from backend.config import prompts as _prompts

SYSTEM_PROMPT = _prompts.get("agents.mindmap.system_prompt")


async def run(state: AgentState, config: RunnableConfig = None) -> AgentState:
    """MindmapAgent 节点入口。"""
    kp_name = await resolve_kp_name(state, config)
    logger.info("[MindmapAgent] kp_name=%s", kp_name)

    context, retrieved_texts = await retrieve_context(state, "MindmapAgent", config)

    state = state.model_copy(update={"retrieved_docs": retrieved_texts})

    prompt = SYSTEM_PROMPT.format(
        context=context, kp_name=kp_name,
        max_depth=app_config.generation.mindmap_max_depth,
        max_children=app_config.generation.mindmap_max_children,
    )

    try:
        raw = await chat_completion(
            [{"role": "system", "content": prompt}],
            temperature=app_config.agents.mindmap.temperature,
            max_tokens=app_config.agents.mindmap.max_tokens,
        )

        try:
            json.loads(raw)
        except json.JSONDecodeError:
            # 如果不是合法 JSON，尝试提取 JSON 部分
            import re
            match = re.search(r"\{[\s\S]*\}", raw)
            if match:
                raw = match.group(0)
                json.loads(raw)  # 再验证一次

        logger.info("[MindmapAgent] 思维导图生成成功，json_len=%d", len(raw))
        state = state.model_copy(update={"draft_content": raw})
    except Exception as e:
        logger.error("[MindmapAgent] 生成失败: %s", e)
        state = state.model_copy(update={"draft_content": f"思维导图生成失败：{e}"})

    return state

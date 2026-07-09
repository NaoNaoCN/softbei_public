"""
backend/agents/clarify_agent.py
ClarifyAgent：针对用户追问/澄清请求，基于对话历史给出简短、针对性的回答。
不重新生成完整文档，直接对话式回复。
"""

from __future__ import annotations

import asyncio

from backend.config import config as app_config
from backend.models.schemas import AgentState
from backend.services.llm import chat_completion
from backend.services.video_search import (
    search_videos,
    inject_video_citations,
    extract_search_keywords,
    extract_topic_from_history,
)
from langchain_core.runnables import RunnableConfig


from backend.config import prompts as _prompts

SYSTEM_PROMPT = _prompts.get("agents.clarify.system_prompt")


async def run(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    ClarifyAgent 节点入口。

    职责：
    1. 基于 chat_history 和 user_message 生成针对性回答
    2. 写入 state.final_content
    3. 直接到 END，跳过 safety_agent
    """
    from backend.services import profile as profile_svc

    # 构建画像上下文
    profile_ctx = ""
    if state.profile:
        profile_ctx = await profile_svc.build_profile_context(state.profile)

    prompt = SYSTEM_PROMPT.format(profile_ctx=profile_ctx if profile_ctx else "暂无")

    messages = [
        {"role": "system", "content": prompt},
    ]
    # 注入对话历史
    messages.extend(state.chat_history)
    messages.append({"role": "user", "content": state.user_message})

    try:
        # 构建带上下文的视频搜索词：历史主题 + 当前提问关键词
        topic = extract_topic_from_history(state.chat_history)
        user_kw = extract_search_keywords(state.user_message)

        from loguru import logger
        logger.debug("[ClarifyAgent] 对话主题: {}, 用户关键词: {}", topic, user_kw)
        # 拼接后限制总词数，避免搜索词过长
        combined_parts = (topic.split() + user_kw.split())[:5]
        video_query = " ".join(combined_parts)

        response, videos = await asyncio.gather(
            chat_completion(messages, temperature=app_config.agents.clarify.temperature),
            search_videos(video_query, skip_extraction=True),
        )
        # 后处理：注入视频引用
        if videos:
            response = inject_video_citations(response, videos)
        state = state.model_copy(update={
            "final_content": response,
            "metadata": {**state.metadata, "video_refs": [v.model_dump() for v in videos]},
        })
    except Exception as e:
        from loguru import logger
        logger.warning("[ClarifyAgent] LLM 调用失败: {}", str(e))
        state = state.model_copy(update={
            "final_content": "抱歉，我暂时无法回答这个问题，请稍后再试。",
            "error": str(e),
        })

    return state

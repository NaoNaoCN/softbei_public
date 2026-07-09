"""
backend/services/chat_history.py
多轮对话历史管理：加载、截断、token 估算。
"""

from __future__ import annotations

from typing import Optional

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import config


def estimate_tokens(text: str) -> int:
    """
    粗略估算 token 数。
    中文约 1.5 字/token，英文约 4 字符/token，取混合估算。
    """
    cn_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    other_chars = len(text) - cn_chars
    return int(cn_chars / config.chat.token_estimation.cn_chars_per_token + other_chars / config.chat.token_estimation.en_chars_per_token)


def truncate_history(
    history: list[dict[str, str]],
    max_turns: int | None = None,
    max_tokens: int | None = None,
) -> list[dict[str, str]]:
    """
    截断对话历史，确保不超过 token 预算。

    策略：
    1. 先按 max_turns 截取最近 N 轮
    2. 从最近往前累加 token，超出预算时截断
    3. 保证返回的历史从 user 消息开始（不会出现孤立的 assistant 消息）
    """
    _max_turns = max_turns if max_turns is not None else config.chat.max_turns
    _max_tokens = max_tokens if max_tokens is not None else config.chat.history_max_tokens

    if not history:
        return []

    # Step 1: 按轮数截断（保留最近 max_turns 轮）
    truncated = history[-(_max_turns * 2):]

    # Step 2: 按 token 预算从后往前保留
    total_tokens = 0
    keep_from = 0
    for i in range(len(truncated) - 1, -1, -1):
        msg_tokens = estimate_tokens(truncated[i].get("content", ""))
        if total_tokens + msg_tokens > _max_tokens:
            keep_from = i + 1
            break
        total_tokens += msg_tokens

    result = truncated[keep_from:]

    # Step 3: 确保从 user 消息开始
    if result and result[0].get("role") == "assistant":
        result = result[1:]

    return result


async def load_chat_history(
    session_id: int,
    db: AsyncSession,
    max_turns: int | None = None,
    max_tokens: int | None = None,
) -> list[dict[str, str]]:
    """
    从 ChatMessage 表加载历史消息并截断。

    :param session_id: 会话 ID（Snowflake BIGINT）
    :param db: 数据库会话
    :param max_turns: 最大保留轮数
    :param max_tokens: 历史 token 预算
    :return: 截断后的历史消息列表
    """
    from backend.db.crud import select as db_select
    from backend.db.models import ChatMessage

    try:
        _max_turns = max_turns if max_turns is not None else config.chat.max_turns
        # DB 层取最近 N×4 条消息作为截断缓冲，避免加载全量历史
        db_limit = _max_turns * 4

        messages = await db_select(
            db, ChatMessage,
            filters={"session_id": session_id},
            order_by=ChatMessage.created_at.desc(),
            limit=db_limit,
        )
        if not messages:
            return []

        # 恢复时间正序供 truncate_history 处理
        history = [{"role": m.role, "content": m.content} for m in reversed(messages)]
        return truncate_history(history, max_turns, max_tokens)

    except Exception as e:
        logger.warning(f"加载对话历史失败: {e}")
        return []

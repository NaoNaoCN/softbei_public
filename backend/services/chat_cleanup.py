"""聊天会话过期清理服务：定期清理过期的 ChatSession 及关联 ChatMessage。"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

from loguru import logger
from sqlalchemy import text

from backend.config import config
from backend.db.database import get_engine


async def cleanup_expired_sessions() -> None:
    """
    清理过期的聊天会话及关联消息。
    ChatMessage 通过外键 CASCADE 自动删除，此处主要清理 ChatSession。
    """
    engine = get_engine()
    expiry_date = datetime.now() - timedelta(days=config.chat.session_expiry_days)

    async with engine.begin() as conn:
        result = await conn.execute(
            text("""
            SELECT COUNT(*) FROM chat_message
            WHERE session_id IN (
                SELECT id FROM chat_session WHERE created_at < :expiry_date
            )
            """),
            {"expiry_date": expiry_date},
        )
        msg_count = result.scalar() or 0

        # 删除过期会话（CASCADE 自动删除关联消息）
        result = await conn.execute(
            text("DELETE FROM chat_session WHERE created_at < :expiry_date"),
            {"expiry_date": expiry_date},
        )
        session_count = result.rowcount or 0

        if session_count > 0:
            logger.info(
                f"[ChatCleanup] 清理完成: 删除 {session_count} 个过期会话, "
                f"{msg_count} 条关联消息"
            )
        else:
            logger.debug("[ChatCleanup] 无过期会话需要清理")


async def start_cleanup_task() -> None:
    """启动后台清理任务，每 24 小时执行一次。"""
    interval = config.chat.cleanup_interval_hours * 3600
    logger.info(f"[ChatCleanup] 启动聊天会话清理后台任务（每{config.chat.cleanup_interval_hours}小时执行一次）")

    while True:
        try:
            await asyncio.sleep(interval)
            await cleanup_expired_sessions()
        except asyncio.CancelledError:
            logger.info("[ChatCleanup] 会话清理任务已取消")
            break
        except Exception as e:
            logger.error(f"[ChatCleanup] 会话清理任务出错: {e}")

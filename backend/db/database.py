"""
backend/db/database.py
PostgreSQL 数据库连接池与基础 CRUD 助手（异步 SQLAlchemy 2.x）。
Schema 管理由 Alembic 负责，此处不再调用 create_all。
"""

from __future__ import annotations

import logging as stdlib_logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, AsyncIterator

from loguru import logger
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from backend.config import config


# ----------------------------------------------------------
# stdlib logging → loguru 桥接
# ----------------------------------------------------------

class _LoguruHandler(stdlib_logging.Handler):
    """将 stdlib logging 记录桥接到 loguru，确保 SQLAlchemy 等库的日志统一输出。"""

    def emit(self, record: stdlib_logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        logger.opt(depth=6, exception=record.exc_info).log(
            level, record.getMessage(),
        )


def _setup_db_logging() -> None:
    """将 SQLAlchemy 引擎日志桥接到 loguru。"""
    sa_logger = stdlib_logging.getLogger("sqlalchemy.engine")
    sa_logger.handlers = []
    sa_logger.addHandler(_LoguruHandler())
    sa_logger.propagate = False

# ----------------------------------------------------------
# ORM Base
# ----------------------------------------------------------

class Base(DeclarativeBase):
    """所有 ORM 模型的基类"""
    pass


# ----------------------------------------------------------
# Engine & Session factory（模块级单例，应用启动时初始化）
# ----------------------------------------------------------

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """返回当前引擎实例，未初始化则抛出 RuntimeError。"""
    if _engine is None:
        raise RuntimeError("Database engine not initialized. Call init_db() first.")
    return _engine


async def init_db() -> None:
    """
    创建 PostgreSQL 异步引擎、建立连接池。
    应在 FastAPI lifespan 的 startup 阶段调用。

    Schema 管理由 Alembic 负责，此处不调用 create_all。
    """
    global _engine, _session_factory
    db_cfg = config.database

    connect_args = {
        "timeout": db_cfg.pool_timeout,
        "command_timeout": db_cfg.command_timeout,
    }

    # 桥接 SQLAlchemy 日志到 loguru
    _setup_db_logging()

    _engine = create_async_engine(
        db_cfg.url,
        echo=db_cfg.echo,
        pool_size=db_cfg.pool_size,
        max_overflow=db_cfg.max_overflow,
        pool_timeout=db_cfg.pool_timeout,
        pool_recycle=db_cfg.pool_recycle,
        pool_pre_ping=True,
        connect_args=connect_args,
    )

    # 记录连接池事件
    @event.listens_for(_engine.sync_engine, "connect")
    def _on_connect(dbapi_connection, connection_record):
        logger.debug("[Database] 新连接建立: pool_size={}", db_cfg.pool_size)

    @event.listens_for(_engine.sync_engine, "close")
    def _on_close(dbapi_connection, connection_record):
        logger.debug("[Database] 连接关闭")

    _session_factory = async_sessionmaker(
        bind=_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    logger.info("[Database] 连接池初始化完成: pool_size={}, max_overflow={}",
                db_cfg.pool_size, db_cfg.max_overflow)


async def close_db() -> None:
    """释放连接池，在 FastAPI lifespan 的 shutdown 阶段调用。"""
    global _engine
    if _engine:
        logger.info("[Database] 关闭连接池...")
        await _engine.dispose()
        _engine = None
        logger.info("[Database] 连接池已释放")


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI Depends 依赖项，提供请求作用域的数据库会话。"""
    if _session_factory is None:
        raise RuntimeError("Database not initialized.")
    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


@asynccontextmanager
async def get_session_ctx() -> AsyncIterator[AsyncSession]:
    """返回 async context manager 形式的数据库会话，用于非 FastAPI DI 场景
    （如评估系统、CLI 脚本、后台任务等）。

    用法::

        async with get_session_ctx() as session:
            ...
    """
    if _session_factory is None:
        raise RuntimeError("Database not initialized.")
    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def health_check() -> bool:
    """简单的数据库连通性检查，返回 True 表示正常。"""
    try:
        async with get_engine().connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False

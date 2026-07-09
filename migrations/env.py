"""
Alembic 迁移环境配置（异步 PostgreSQL）。
"""
import asyncio
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.ext.asyncio import create_async_engine

from alembic import context

# Alembic Config 对象
config = context.config

# 日志配置
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# 导入 ORM Base，用于 autogenerate
from backend.db.database import Base  # noqa: E402

# 确保所有模型已注册到 Base.metadata
from backend.db import models  # noqa: E402, F401

target_metadata = Base.metadata

# 从应用配置读取数据库 URL
from backend.config import config as app_config  # noqa: E402


def get_url() -> str:
    return app_config.database.url


def run_migrations_offline() -> None:
    """离线模式：生成 SQL 脚本而不连接数据库。"""
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection):
    """同步回调：在异步连接上执行迁移。"""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """在线模式：通过异步引擎连接数据库并执行迁移。"""
    connectable = create_async_engine(
        get_url(),
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())

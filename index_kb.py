"""
index_kb.py
一次性知识库入库脚本（提交快照准备用）。

复刻 main.py lifespan 的自动索引逻辑，但作为独立脚本运行：
递归索引 config.storage.knowledge_base_dir 下的全部课程文档，
调用 Embedding API 构建向量，写入 document_chunk 表。

用法（数据库由 .env 中 DATABASE_URL 决定）：
    python index_kb.py
"""

from __future__ import annotations

import asyncio

from loguru import logger

from backend.config import config
from backend.db.database import init_db, close_db
from backend.db.vector import get_collection
from backend.rag.indexer import index_directory


async def main() -> None:
    await init_db()
    try:
        col = get_collection()
        existing = await col.count()
        logger.info(f"[index_kb] 现有向量块: {existing}")

        kb_dir = config.storage.knowledge_base_dir
        logger.info(f"[index_kb] 开始索引目录: {kb_dir}")
        indexed = await index_directory(kb_dir)

        total = await col.count()
        logger.info(f"[index_kb] DONE_INDEXING indexed={indexed} total_now={total}")
    finally:
        await close_db()


if __name__ == "__main__":
    asyncio.run(main())

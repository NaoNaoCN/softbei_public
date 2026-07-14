"""文档导入服务：保存文件、转换为 Markdown、解析内容、索引到向量库。

支持格式：PDF / DOCX / DOC / Markdown / TXT
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional

from loguru import logger

from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import config
from backend.db.crud import insert
from backend.db.models import ResourceMeta
from backend.rag import loader, indexer as rag_indexer
from backend.services.llm import check_embedding_health
from backend.utils.snowflake import generate_id


UPLOAD_DIR = Path(__file__).parent.parent.parent / config.storage.upload_dir
UPLOAD_DIR.mkdir(exist_ok=True)

SUPPORTED_SUFFIXES = set(config.storage.supported_extensions)


async def import_document(
    file_path: str,
    user_id: int,
    title: Optional[str] = None,
    db: Optional[AsyncSession] = None,
) -> dict:
    """
    导入文档：保存文件 → 转换为 Markdown → 解析为文本块 → 索引到向量库 → 创建资源记录。

    :param file_path:  临时保存的文件路径
    :param user_id:    上传用户 ID
    :param title:      自定义文档标题，默认使用文件名
    :param db:         数据库会话（可选，无会话时仅解析和索引）
    :return:           {"doc_id": str, "chunks": int, "resource_id": uuid}
    """
    t_start = time.perf_counter()
    path = Path(file_path)
    doc_id = f"doc_{hex(generate_id())[2:2 + config.storage.doc_id_hex_length]}"
    doc_title = title or path.stem
    file_size_bytes = path.stat().st_size

    # 预检 Embedding API 连通性，避免解析后才发现 API 不可用
    if not await check_embedding_health():
        raise RuntimeError(
            "Embedding API 连接失败，无法索引文档。请检查网络连接和 LLM_API_KEY 配置。"
        )

    t_parse = time.perf_counter()
    chunks = loader.load_file(str(path), doc_id=doc_id)
    parse_ms = (time.perf_counter() - t_parse) * 1000
    logger.info(f"[import_document] 解析完成，生成 {len(chunks)} 个文本块 ({parse_ms:.0f}ms)")

    t_index = time.perf_counter()
    indexed_count = 0
    if chunks:
        indexed_count = await rag_indexer.index_chunks(chunks, user_id=str(user_id))
    index_ms = (time.perf_counter() - t_index) * 1000
    logger.info(f"[import_document] 索引完成，共索引 {indexed_count} 个文本块 ({index_ms:.0f}ms)")

    t_db = time.perf_counter()
    resource_id = None
    if db is not None:
        resource = await insert(
            db, ResourceMeta,
            data={
                "user_id": user_id,
                "kp_id": doc_id,
                "resource_type": "doc",
                "title": doc_title,
                "content": f"已导入文档：{path.name}，共 {len(chunks)} 个文本块",
            },
        )
        resource_id = resource.id
    db_ms = (time.perf_counter() - t_db) * 1000
    if resource_id:
        logger.info(f"[import_document] 资源记录创建完成，ID={resource_id} ({db_ms:.0f}ms)")

    total_ms = (time.perf_counter() - t_start) * 1000

    logger.info(
        f"[Metrics] import_document | "
        f"file={path.name} "
        f"format={path.suffix.lower()} "
        f"file_size_bytes={file_size_bytes} "
        f"chunks={len(chunks)} "
        f"indexed={indexed_count} "
        f"parse_ms={parse_ms:.0f} "
        f"index_ms={index_ms:.0f} "
        f"db_ms={db_ms:.0f} "
        f"total_ms={total_ms:.0f} "
        f"throughput_kb_s={file_size_bytes / 1024 / (total_ms / 1000):.1f}"
    )

    return {
        "doc_id": doc_id,
        "title": doc_title,
        "file_name": path.name,
        "chunks": len(chunks),
        "indexed": indexed_count,
        "resource_id": str(resource_id) if resource_id else None,
        "timings_ms": {
            "parse": round(parse_ms),
            "index": round(index_ms),
            "db": round(db_ms),
            "total": round(total_ms),
        },
    }


async def import_document_with_progress(
    file_path: str,
    user_id: int,
    title: Optional[str] = None,
    db: Optional[AsyncSession] = None,
    progress_callback: Optional[Callable[[str, int], None]] = None,
) -> dict:
    """
    带进度回调的文档导入。

    :param progress_callback: (stage: str, percent: int) 在各阶段调用
    """
    def _cb(stage: str, pct: int):
        if progress_callback is not None:
            progress_callback(stage, pct)

    t_start = time.perf_counter()
    path = Path(file_path)
    doc_id = f"doc_{hex(generate_id())[2:2 + config.storage.doc_id_hex_length]}"
    doc_title = title or path.stem
    file_size_bytes = path.stat().st_size
    logger.info(f"[import_document_with_progress] received title={title!r}, final doc_title={doc_title!r}")

    _cb("saving", 5)

    # 预检 Embedding API 连通性，避免解析后才发现 API 不可用
    if not await check_embedding_health():
        raise RuntimeError(
            "Embedding API 连接失败，无法索引文档。请检查网络连接和 LLM_API_KEY 配置。"
        )

    t_parse = time.perf_counter()
    chunks = loader.load_file(str(path), doc_id=doc_id)
    parse_ms = (time.perf_counter() - t_parse) * 1000
    logger.info(f"[import_document_with_progress] 解析完成，生成 {len(chunks)} 个文本块 ({parse_ms:.0f}ms)")
    _cb("parsing", 20)

    t_index = time.perf_counter()
    indexed_count = 0
    if chunks:
        total_batches_ref = [1]

        def _index_progress(batch_num: int, total_batches: int):
            total_batches_ref[0] = total_batches
            pct = 20 + int(batch_num / total_batches * 70)
            _cb("indexing", min(pct, 90))

        _cb("indexing", 20)
        indexed_count = await rag_indexer.index_chunks(
            chunks, progress_callback=_index_progress, user_id=str(user_id)
        )
    index_ms = (time.perf_counter() - t_index) * 1000
    logger.info(f"[import_document_with_progress] 索引完成，共索引 {indexed_count} 个文本块 ({index_ms:.0f}ms)")

    t_db = time.perf_counter()
    resource_id = None
    if db is not None:
        resource = await insert(
            db, ResourceMeta,
            data={
                "user_id": user_id,
                "kp_id": doc_id,
                "resource_type": "doc",
                "title": doc_title,
                "content": f"已导入文档：{path.name}，共 {len(chunks)} 个文本块",
            },
        )
        resource_id = resource.id
    db_ms = (time.perf_counter() - t_db) * 1000
    if resource_id:
        logger.info(f"[import_document_with_progress] 资源记录创建完成，ID={resource_id} ({db_ms:.0f}ms)")
    _cb("saving_record", 95)

    _cb("done", 100)

    total_ms = (time.perf_counter() - t_start) * 1000

    logger.info(
        f"[Metrics] import_document_with_progress | "
        f"file={path.name} "
        f"format={path.suffix.lower()} "
        f"file_size_bytes={file_size_bytes} "
        f"chunks={len(chunks)} "
        f"indexed={indexed_count} "
        f"parse_ms={parse_ms:.0f} "
        f"index_ms={index_ms:.0f} "
        f"db_ms={db_ms:.0f} "
        f"total_ms={total_ms:.0f} "
        f"throughput_kb_s={file_size_bytes / 1024 / (total_ms / 1000):.1f}"
    )

    return {
        "doc_id": doc_id,
        "title": doc_title,
        "file_name": path.name,
        "chunks": len(chunks),
        "indexed": indexed_count,
        "resource_id": str(resource_id) if resource_id else None,
        "timings_ms": {
            "parse": round(parse_ms),
            "index": round(index_ms),
            "db": round(db_ms),
            "total": round(total_ms),
        },
    }


def save_uploaded_file(content: bytes, original_name: str) -> str:
    """
    将上传的文件内容保存到 upload 目录。

    :param content:        文件字节内容
    :param original_name:  原始文件名
    :return:               保存后的文件路径
    """
    suffix = Path(original_name).suffix.lower()
    supported = set(config.storage.supported_extensions)
    if suffix not in supported:
        raise ValueError(f"不支持的文件格式：{suffix}，支持：{', '.join(sorted(supported))}")

    unique_name = f"{hex(generate_id())[2:2 + config.storage.doc_id_hex_length]}_{original_name}"
    dest = UPLOAD_DIR / unique_name
    dest.write_bytes(content)
    return str(dest)

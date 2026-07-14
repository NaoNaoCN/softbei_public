"""向量索引构建器：将 TextChunk 列表嵌入并写入向量库。"""

from __future__ import annotations

from typing import Callable, Optional

from loguru import logger

from backend.config import config
from backend.db.vector import (
    upsert_documents,
    delete_by_doc_id,
)
from backend.rag.loader import TextChunk
from backend.services.llm import get_embeddings_batch


def _tokenize_for_tsvector(text: str) -> str:
    """
    使用 jieba 精确模式对文本分词，返回空格分隔的 token 字符串，
    供 PostgreSQL to_tsvector('simple', ...) 使用。
    """
    import jieba

    if not text or len(text.strip()) < 2:
        return text.strip()

    words = jieba.lcut(text, cut_all=False)
    tokens = [w.strip() for w in words if len(w.strip()) >= 1]
    return " ".join(tokens)


async def index_chunks(
    chunks: list[TextChunk],
    collection_name: Optional[str] = None,
    batch_size: int = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    user_id: Optional[str] = None,
) -> int:
    """
    全量索引文本块到向量库。

    对于每个 doc_id，先删除旧数据再写入新数据（全量替换策略）。

    :param chunks:             TextChunk 列表（来自 loader）
    :param collection_name:    目标集合名，None 使用默认集合
    :param batch_size:         每批嵌入请求的大小（上限受 API 限制）
    :param progress_callback:  可选回调 (batch_num, total_batches)，每批完成后调用
    :param user_id:            上传用户 ID，写入 metadata 用于账户隔离
    :return:                   成功 upsert 的 chunk 数量
    """
    if not chunks:
        return 0

    if batch_size is None:
        batch_size = config.embedding.index_batch_size
    effective_batch_size = min(batch_size, config.embedding.api_max_batch_size)

    # 全量替换：先删除每个 doc_id 的旧数据
    unique_doc_ids = set(c.doc_id for c in chunks if c.doc_id)
    for doc_id in unique_doc_ids:
        await delete_by_doc_id(doc_id, collection_name=collection_name)

    parents = [c for c in chunks if c.is_parent]
    children = [c for c in chunks if not c.is_parent]

    logger.info(
        f"[Indexer] 全量索引：{len(chunks)} 个 chunk "
        f"（父块 {len(parents)}，子块 {len(children)}），"
        f"涉及 {len(unique_doc_ids)} 个文档"
    )

    if parents:
        logger.info(f"[Indexer] 写入 {len(parents)} 个父块（无嵌入）...")
        for i in range(0, len(parents), effective_batch_size):
            batch = parents[i : i + effective_batch_size]
            await upsert_documents(
                ids=[c.chunk_id for c in batch],
                documents=[c.text for c in batch],
                embeddings=[None] * len(batch),  # 父块不嵌入
                metadatas=[
                    {
                        "doc_id": c.doc_id,
                        "source": c.source_path,
                        "page": str(c.page or ""),
                        "section": c.section or "",
                        "user_id": user_id or "",
                        "parent_chunk_id": c.parent_chunk_id or "",
                        "is_parent": True,
                        "text_search": _tokenize_for_tsvector(c.text),
                        **c.metadata,
                    }
                    for c in batch
                ],
                collection_name=collection_name,
            )

    total = 0
    batches = list(range(0, len(children), effective_batch_size))
    total_batches = len(batches)

    for batch_num, i in enumerate(batches, start=1):
        batch = children[i : i + effective_batch_size]
        if not batch:
            continue
        logger.info(
            f"[Indexer] 正在 embedding 第 {i+1}-{i+len(batch)}/{len(children)} 块..."
        )
        embeddings = await _embed_batch([c.text for c in batch])

        # 校验嵌入结果：跳过空向量（Embedding API 失败时不应继续写入）
        valid_pairs = [(c, emb) for c, emb in zip(batch, embeddings) if emb and len(emb) > 0]
        if not valid_pairs:
            logger.error(
                f"[Indexer] Embedding API 返回空向量，跳过批次 "
                f"({i + 1}-{i + len(batch)}/{len(children)})"
            )
            continue
        if len(valid_pairs) < len(batch):
            logger.warning(
                f"[Indexer] {len(batch) - len(valid_pairs)}/{len(batch)} 个 chunk 嵌入为空，已过滤"
            )
        valid_batch = [c for c, _ in valid_pairs]
        valid_embeddings = [emb for _, emb in valid_pairs]

        await upsert_documents(
            ids=[c.chunk_id for c in valid_batch],
            documents=[c.text for c in valid_batch],
            embeddings=valid_embeddings,
            metadatas=[
                {
                    "doc_id": c.doc_id,
                    "source": c.source_path,
                    "page": str(c.page or ""),
                    "section": c.section or "",
                    "user_id": user_id or "",
                    "parent_chunk_id": c.parent_chunk_id or "",
                    "is_parent": False,
                    "text_search": _tokenize_for_tsvector(c.text),
                    **c.metadata,
                }
                for c in valid_batch
            ],
            collection_name=collection_name,
        )
        total += len(valid_batch)
        if progress_callback is not None:
            progress_callback(batch_num, total_batches)

    logger.info(f"[Indexer] 全量索引完成：{total} 个 chunk 已写入")
    return total


async def index_file(
    file_path: str,
    collection_name: Optional[str] = None,
) -> int:
    """一键加载并索引单个文件。"""
    from backend.rag.loader import load_file
    chunks = load_file(file_path)
    return await index_chunks(chunks, collection_name=collection_name)


async def index_directory(
    dir_path: str,
    collection_name: Optional[str] = None,
) -> int:
    """递归扫描目录并全量索引。"""
    from backend.rag.loader import load_directory
    chunks = load_directory(dir_path)
    return await index_chunks(chunks, collection_name=collection_name)


async def _embed_batch(texts: list[str]) -> list[list[float]]:
    """批量嵌入文本，使用 API 批量接口一次发送多条。"""
    if not texts:
        return []
    return await get_embeddings_batch(texts)

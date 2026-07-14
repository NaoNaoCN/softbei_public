"""向量存储（基于 pgvector 扩展，向量检索在数据库内完成）。"""

from __future__ import annotations

from typing import Optional

from loguru import logger
from sqlalchemy import text

from backend.config import config
from backend.db.database import get_engine

COLLECTION_NAME: str = config.vector_db.collection

# HNSW ef_search：控制检索精度与速度的平衡，由 config.vector_db.hnsw_ef_search 控制


def _format_vector(embedding: list[float]) -> str:
    """将 list[float] 转为 pgvector 可解析的字符串（如 "[-0.02, 0.01, ...]"）。"""
    return "[" + ",".join(str(x) for x in embedding) + "]"


class _CollectionProxy:
    """向量集合的异步代理。"""

    def __init__(self, name: str = "knowledge_base"):
        self.collection_name = name

    async def count(self) -> int:
        """返回该集合中的文档块数量。"""
        engine = get_engine()
        async with engine.connect() as conn:
            result = await conn.execute(
                text("SELECT COUNT(*) FROM document_chunk WHERE collection_name = :cn"),
                {"cn": self.collection_name},
            )
            return result.scalar() or 0

    async def get(self, where: Optional[dict] = None, limit: Optional[int] = None,
                  include: Optional[list[str]] = None) -> dict:
        """按条件获取文档块。"""
        engine = get_engine()
        conditions = ["collection_name = :cn"]
        params: dict = {"cn": self.collection_name}

        if where:
            where_clause, where_params = _build_where_clause(where)
            if where_clause:
                conditions.append(where_clause)
                params.update(where_params)

        sql = f"SELECT chunk_id, text, doc_id, embedding, source, page, section, user_id, metadata FROM document_chunk WHERE {' AND '.join(conditions)}"
        if limit is not None:
            limit_val = int(limit)
            sql += f" LIMIT {limit_val}"

        async with engine.connect() as conn:
            result = await conn.execute(text(sql), params)
            rows = result.fetchall()

        ids_list = []
        documents_list = []
        metadatas_list = []
        for row in rows:
            ids_list.append(row[0])
            documents_list.append(row[1])
            meta = {
                "doc_id": row[2] or "",
                "source": row[4] or "",
                "page": int(row[5]) if row[5] is not None else None,
                "section": row[6] or "",
                "user_id": row[7] or "",
            }
            raw_metadata = row[8] if len(row) > 8 else None
            if raw_metadata and isinstance(raw_metadata, dict):
                meta.update(raw_metadata)
            metadatas_list.append(meta)

        return {
            "ids": ids_list,
            "documents": documents_list,
            "metadatas": metadatas_list,
        }


def init_vector_db() -> None:
    """初始化向量库。pgvector 方案下表由 Alembic 管理，此处仅验证连通性。"""
    logger.info("[VectorDB] 使用 pgvector 向量存储 (cosine distance via <=>) ")


_collection_proxy: Optional[_CollectionProxy] = None


def get_collection() -> _CollectionProxy:
    """返回默认知识库集合代理。"""
    global _collection_proxy
    if _collection_proxy is None:
        _collection_proxy = _CollectionProxy(COLLECTION_NAME)
    return _collection_proxy


def _convert_metadata_to_columns(meta: dict) -> dict:
    """将 metadata dict 转换为列值。"""
    import json

    # 提取 extra metadata（排除固定列字段 + 父子切割字段 + tsvector 字段）
    extra_meta = {
        k: v for k, v in meta.items()
        if k not in ("source", "page", "section", "user_id", "doc_id", "chunk_id",
                      "parent_chunk_id", "is_parent", "text_search")
    }
    return {
        "source": meta.get("source", ""),
        "page": int(meta["page"]) if meta.get("page") and str(meta["page"]).isdigit() else None,
        "section": meta.get("section", ""),
        "user_id": meta.get("user_id", ""),
        "parent_chunk_id": meta.get("parent_chunk_id", ""),
        "is_parent": bool(meta.get("is_parent", False)),
        "text_search": meta.get("text_search", ""),
        "metadata_": json.dumps(extra_meta, ensure_ascii=False) if extra_meta else None,
    }


async def upsert_documents(
    ids: list[str],
    documents: list[str],
    embeddings: list[list[float]],
    metadatas: Optional[list[dict]] = None,
    collection_name: Optional[str] = None,
) -> None:
    """将文档及其向量批量写入向量库（pgvector，多行 VALUES 一次 INSERT）。"""
    from backend.utils.snowflake import generate_id

    if not ids:
        return

    col = collection_name or COLLECTION_NAME
    meta_list = metadatas or [{}] * len(ids)
    engine = get_engine()

    columns = [
        "id", "chunk_id", "doc_id", "collection_name", "text",
        "embedding", "source", "page", "section", "user_id",
        "parent_chunk_id", "is_parent", "text_search", "metadata", "created_at",
    ]
    value_rows: list[str] = []
    params: dict = {}

    for i, (chunk_id, doc_text, emb, meta) in enumerate(
        zip(ids, documents, embeddings, meta_list)
    ):
        cols = _convert_metadata_to_columns(meta)
        doc_id = meta.get("doc_id", "")
        # 将 list[float] 转为 pgvector 字符串格式（如 "[-0.02, 0.01, ...]"），
        # 配合 SQL 中 ::vector 转型，避免依赖 asyncpg codec 注册。
        emb_str = _format_vector(emb) if (emb and len(emb) > 0) else None
        row_placeholders = ", ".join([
            f":id_{i}", f":chunk_id_{i}", f":doc_id_{i}", f":col_{i}",
            f":text_{i}",
            f"CAST(:emb_{i} AS vector)" if emb_str else "NULL",
            f":source_{i}", f":page_{i}",
            f":section_{i}", f":user_id_{i}",
            f":parent_chunk_id_{i}", f":is_parent_{i}",
            f"to_tsvector('simple', :text_search_{i})" if cols.get("text_search") else "NULL",
            f":metadata__{i}", "NOW()",
        ])
        value_rows.append(f"({row_placeholders})")
        params.update({
            f"id_{i}": generate_id(),
            f"chunk_id_{i}": chunk_id,
            f"doc_id_{i}": doc_id,
            f"col_{i}": col,
            f"text_{i}": doc_text,
            f"source_{i}": cols["source"],
            f"page_{i}": cols["page"],
            f"section_{i}": cols["section"],
            f"user_id_{i}": cols["user_id"],
            f"parent_chunk_id_{i}": cols["parent_chunk_id"],
            f"is_parent_{i}": cols["is_parent"],
            f"metadata__{i}": cols["metadata_"],
        })
        text_search_val = cols.get("text_search", "")
        if text_search_val:
            params[f"text_search_{i}"] = text_search_val
        if emb_str:
            params[f"emb_{i}"] = emb_str

    sql = f"""
        INSERT INTO document_chunk ({', '.join(columns)})
        VALUES {', '.join(value_rows)}
        ON CONFLICT (chunk_id) DO UPDATE SET
            text = EXCLUDED.text,
            embedding = EXCLUDED.embedding,
            source = EXCLUDED.source,
            page = EXCLUDED.page,
            section = EXCLUDED.section,
            user_id = EXCLUDED.user_id,
            parent_chunk_id = EXCLUDED.parent_chunk_id,
            is_parent = EXCLUDED.is_parent,
            text_search = EXCLUDED.text_search,
            metadata = EXCLUDED.metadata
    """

    async with engine.begin() as conn:
        await conn.execute(text(sql), params)


def _build_where_clause(where: Optional[dict]) -> tuple[str, dict]:
    """将 where 条件转换为 SQL WHERE 子句。

    支持：
    - {"user_id": "xxx"} → user_id = 'xxx'
    - {"$or": [{"user_id": "a"}, {"user_id": ""}]} → (user_id = 'a' OR user_id = '')
    - {"$and": [...]} → (cond1 AND cond2)
    """
    if not where:
        return "", {}

    params: dict = {}
    counter = [0]

    def _convert(condition: dict) -> str:
        parts = []
        for key, value in condition.items():
            if key == "$or":
                or_parts = []
                for item in value:
                    or_parts.append(_convert(item))
                parts.append(f"({' OR '.join(or_parts)})")
            elif key == "$and":
                and_parts = []
                for item in value:
                    and_parts.append(_convert(item))
                parts.append(f"({' AND '.join(and_parts)})")
            else:
                pname = f"wp_{counter[0]}"
                counter[0] += 1
                params[pname] = value
                parts.append(f"{key} = :{pname}")
        return " AND ".join(parts)

    clause = _convert(where)
    return clause, params


async def query_documents(
    query_embedding: list[float],
    n_results: int = 5,
    where: Optional[dict] = None,
    collection_name: Optional[str] = None,
) -> dict:
    """
    pgvector 向量检索。
    使用 <=> 余弦距离运算符，检索在数据库内完成，仅返回 top-N。
    """
    col = collection_name or COLLECTION_NAME
    engine = get_engine()

    conditions = ["collection_name = :cn", "is_parent = FALSE"]
    # 只检索子块；父块没有 embedding，不参与向量检索
    emb_str = _format_vector(query_embedding)
    params: dict = {"cn": col, "embedding": emb_str}

    where_clause, where_params = _build_where_clause(where)
    if where_clause:
        conditions.append(where_clause)
        params.update(where_params)

    sql = f"""
        SELECT
            chunk_id,
            text,
            doc_id,
            embedding <=> CAST(:embedding AS vector) AS distance,
            source,
            page,
            section,
            user_id,
            metadata,
            parent_chunk_id,
            is_parent
        FROM document_chunk
        WHERE {' AND '.join(conditions)}
        ORDER BY embedding <=> CAST(:embedding AS vector)
        LIMIT :n_results
    """

    async with engine.connect() as conn:
        # 设置 HNSW ef_search 控制检索精度/速度平衡
        await conn.execute(text(f"SET LOCAL hnsw.ef_search = {config.vector_db.hnsw_ef_search}"))
        result = await conn.execute(text(sql), {**params, "n_results": n_results})
        rows = result.fetchall()

    ids_list = []
    documents_list = []
    distances_list = []
    metadatas_list = []

    for row in rows:
        chunk_id = row[0]
        ids_list.append(chunk_id)
        documents_list.append(row[1])
        distances_list.append(float(row[3]) if row[3] is not None else 1.0)
        meta = {
            "doc_id": row[2] or "",
            "source": row[4] or "",
            "page": int(row[5]) if row[5] is not None else None,
            "section": row[6] or "",
            "user_id": row[7] or "",
        }
        raw_metadata = row[8] if len(row) > 8 else None
        if raw_metadata and isinstance(raw_metadata, dict):
            meta.update(raw_metadata)
        if len(row) > 9 and row[9]:
            meta["parent_chunk_id"] = row[9]
        if len(row) > 10:
            meta["is_parent"] = row[10]
        metadatas_list.append(meta)

    return {
        "ids": [ids_list],
        "documents": [documents_list],
        "distances": [distances_list],
        "metadatas": [metadatas_list],
    }


async def delete_documents(ids: list[str], collection_name: Optional[str] = None) -> None:
    """按 chunk_id 列表删除向量库中的文档。"""
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM document_chunk WHERE chunk_id = ANY(:ids)"),
            {"ids": ids},
        )


async def delete_by_doc_id(doc_id: str, collection_name: Optional[str] = None) -> None:
    """删除指定 doc_id 的所有向量块。"""
    engine = get_engine()
    async with engine.begin() as conn:
        result = await conn.execute(
            text("DELETE FROM document_chunk WHERE doc_id = :doc_id"),
            {"doc_id": doc_id},
        )
        deleted = result.rowcount
        logger.info(f"[VectorDB] 删除 doc_id={doc_id} 的 {deleted} 个向量块")


async def get_parent_texts(
    parent_ids: list[str],
    collection_name: Optional[str] = None,
) -> dict[str, str]:
    """
    批量获取父块文本。用于检索后父块回填。

    :param parent_ids:      父块 chunk_id 列表
    :param collection_name: 可选集合过滤
    :return:                {chunk_id: text} 映射（不存在的 key 不在结果中）
    """
    if not parent_ids:
        return {}

    engine = get_engine()
    col = collection_name or COLLECTION_NAME
    async with engine.connect() as conn:
        result = await conn.execute(
            text(
                "SELECT chunk_id, text FROM document_chunk "
                "WHERE chunk_id = ANY(:ids) AND is_parent = TRUE AND collection_name = :cn"
            ),
            {"ids": parent_ids, "cn": col},
        )
        rows = result.fetchall()
    return {row[0]: row[1] for row in rows}


async def get_documents_by_doc_id(doc_id: str, collection_name: Optional[str] = None) -> dict:
    """按 doc_id 获取所有文本块及其元数据。"""
    engine = get_engine()
    async with engine.connect() as conn:
        result = await conn.execute(
            text("SELECT text, source, page, section, user_id, metadata FROM document_chunk WHERE doc_id = :doc_id ORDER BY created_at"),
            {"doc_id": doc_id},
        )
        rows = result.fetchall()

    documents = []
    metadatas = []
    for row in rows:
        documents.append(row[0])
        meta = {
            "source": row[1] or "",
            "page": str(row[2]) if row[2] else "",
            "section": row[3] or "",
            "user_id": row[4] or "",
        }
        raw_metadata = row[5] if len(row) > 5 else None
        if raw_metadata and isinstance(raw_metadata, dict):
            meta.update(raw_metadata)
        metadatas.append(meta)

    return {"documents": documents, "metadatas": metadatas}


async def query_keyword(
    keywords: list[str],
    n_results: int = 5,
    where: dict | None = None,
    collection_name: str | None = None,
) -> dict:
    """
    关键词全文检索：基于 PostgreSQL tsvector/tsquery + jieba 分词。

    查询关键词（已由上层 jieba 分词并过滤停用词）以 & (AND) 连接构成
    tsquery，使用 ts_rank() 作为相关性评分，按 rank DESC 排序返回 top-N。

    只检索 text_search IS NOT NULL 的行（legacy 数据未回填 tsvector 的被排除）。

    :param keywords:        jieba 分词后的关键词列表
    :param n_results:       返回条数
    :param where:           额外过滤条件
    :param collection_name: 集合名
    :return:                与 query_documents() 相同格式的结果 dict
    """
    if not keywords:
        return {"ids": [[]], "documents": [[]], "distances": [[]], "metadatas": [[]]}

    col = collection_name or COLLECTION_NAME
    engine = get_engine()

    conditions = [
        "collection_name = :cn",
        "is_parent = FALSE",
        "text_search IS NOT NULL",
    ]
    params: dict = {"cn": col}

    # 构建 tsquery：用 & (AND) 连接所有关键词，'simple' 配置不做 stemming/stopword
    tsquery_parts: list[str] = []
    for kw in keywords:
        safe_kw = kw.replace("\\", "\\\\").replace("'", "\\'")
        tsquery_parts.append(safe_kw)
    tsquery_str = " & ".join(tsquery_parts)
    params["tsquery"] = tsquery_str

    conditions.append("text_search @@ to_tsquery('simple', :tsquery)")

    where_clause, where_params = _build_where_clause(where)
    if where_clause:
        conditions.append(where_clause)
        params.update(where_params)

    sql = f"""
        SELECT
            chunk_id,
            text,
            doc_id,
            ts_rank(text_search, to_tsquery('simple', :tsquery)) AS rank,
            source,
            page,
            section,
            user_id,
            metadata,
            parent_chunk_id,
            is_parent
        FROM document_chunk
        WHERE {' AND '.join(conditions)}
        ORDER BY rank DESC
        LIMIT :n_results
    """

    async with engine.connect() as conn:
        result = await conn.execute(text(sql), {**params, "n_results": n_results})
        rows = result.fetchall()

    ids_list = []
    documents_list = []
    distances_list = []
    metadatas_list = []

    if not rows:
        return {"ids": [[]], "documents": [[]], "distances": [[]], "metadatas": [[]]}

    for row in rows:
        ids_list.append(row[0])
        documents_list.append(row[1])
        # ts_rank → "距离"（越低越好）：rank 越高，距离越小
        rank = float(row[3]) if row[3] is not None else 0.0
        distances_list.append(1.0 - rank)
        meta = {
            "doc_id": row[2] or "",
            "source": row[4] or "",
            "page": int(row[5]) if row[5] is not None else None,
            "section": row[6] or "",
            "user_id": row[7] or "",
        }
        raw_metadata = row[8] if len(row) > 8 else None
        if raw_metadata and isinstance(raw_metadata, dict):
            meta.update(raw_metadata)
        if len(row) > 9 and row[9]:
            meta["parent_chunk_id"] = row[9]
        if len(row) > 10:
            meta["is_parent"] = row[10]
        metadatas_list.append(meta)

    return {
        "ids": [ids_list],
        "documents": [documents_list],
        "distances": [distances_list],
        "metadatas": [metadatas_list],
    }


async def health_check() -> bool:
    """检查向量库是否可用。"""
    try:
        engine = get_engine()
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1 FROM document_chunk LIMIT 0"))
        return True
    except Exception:
        return False




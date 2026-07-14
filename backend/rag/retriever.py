"""RAG 检索器：给定用户问题，返回相关文本块及其来源引用。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

from backend.db.vector import query_documents, query_keyword
from backend.services.llm import get_embedding

# 数据结构

@dataclass
class RetrievedChunk:
    """检索到的单个文本块及相关信息。"""
    chunk_id: str
    text: str
    score: float        # 相似度得分（越高越相关）
    doc_id: str
    source: str         # 原始文件路径
    page: Optional[int] = None
    section: Optional[str] = None
    metadata: dict = field(default_factory=dict)  # 扩展元数据（JSONB，如 language、chunk_type 等）


@dataclass
class CitationSource:
    """format_context 实际展示的单条参考资料的来源信息。

    index 与 context 字符串中的 [n] 编号严格对应（经多样化排序 + token 截断后定稿），
    供生成 Agent 在文末程序化追加「参考资料」清单，保证正文编号与清单一一对齐。
    """
    index: int                    # 引用编号 n（从 1 开始）
    source: str                   # 原始文件名/路径
    page: Optional[int] = None
    section: Optional[str] = None


# 公开接口

async def retrieve(
    query: str,
    n_results: int | None = None,
    score_threshold: float | None = None,
    where: Optional[dict] = None,
    collection_name: Optional[str] = None,
    user_id: Optional[str] = None,
) -> list[RetrievedChunk]:
    """
    语义检索：将 query 嵌入后查询向量库，过滤低相似度结果。
    向量库未初始化或为空时返回空列表（优雅降级）。
    """
    try:
        from backend.db.vector import get_collection
        col = get_collection()
        doc_count = await col.count()
        if doc_count == 0:
            logger.warning("[RAG] 向量库为空（0 条文档），RAG 降级为纯 LLM 生成。请运行 python -m backend.rag.indexer 导入文档。")
            return []
        logger.info(f"[RAG] 向量库就绪，共 {doc_count} 条文档，开始检索: query={query[:60]!r}")
    except Exception as e:
        logger.warning(f"[RAG] 向量库未初始化或不可用: {e}，RAG 降级为纯 LLM 生成。")
        return []

    from backend.config import config as _cfg
    _n_results = n_results if n_results is not None else _cfg.rag.n_results
    _score_threshold = score_threshold if score_threshold is not None else _cfg.rag.score_threshold

    embedding = await get_embedding(query)
    if not embedding:
        logger.warning("[RAG] Embedding 返回空向量，无法执行语义检索。请检查 embedding 模型/API 配置。")
        return []

    # 构建用户隔离过滤条件
    effective_where = where
    if user_id:
        user_filter = {"$or": [
            {"user_id": user_id},
            {"user_id": ""},
        ]}
        if effective_where:
            effective_where = {"$and": [effective_where, user_filter]}
        else:
            effective_where = user_filter

    # 预取更多候选，用于后续 re-rank 精排
    prefetch_count = max(_n_results * _cfg.rag.prefetch_multiplier, _cfg.rag.prefetch_min)
    raw = await query_documents(
        query_embedding=embedding,
        n_results=prefetch_count,
        where=effective_where,
        collection_name=collection_name,
    )
    chunks = _parse_results(raw, _score_threshold)

    # Re-rank: 对 cosine 结果做关键词重叠加权重排
    if chunks:
        chunks = _rerank_by_keyword_overlap(query, chunks)

        # 父块回填：子块 → 父块映射 + 去重（parent_chunking 启用时生效）
        chunks = await _resolve_parent_chunks(chunks)
        chunks = chunks[:_n_results]

    if not chunks:
        logger.info(f"[RAG] 检索无结果（threshold={_score_threshold}），query={query[:60]!r}，将由 LLM 纯生成")
    else:
        logger.info(f"[RAG] 检索到 {len(chunks)} 条相关文档，最高分={chunks[0].score:.3f}，最低分={chunks[-1].score:.3f}")
    return chunks


async def retrieve_by_kp(
    kp_name: str,
    n_results: int | None = None,
    collection_name: Optional[str] = None,
    user_id: Optional[str] = None,
) -> list[RetrievedChunk]:
    """
    按知识点名称检索相关文档片段。
    使用多角度查询扩展以提升检索覆盖率和精度（固定模板方案，Query Rewrite 未启用时使用）。
    """
    query = f"知识点：{kp_name}；定义：{kp_name}；{kp_name}的核心概念与原理"
    return await retrieve(
        query=query,
        n_results=n_results,
        collection_name=collection_name,
        user_id=user_id,
    )


async def retrieve_with_queries(
    queries: list[str],
    n_results: int | None = None,
    score_threshold: float | None = None,
    collection_name: Optional[str] = None,
    user_id: Optional[str] = None,
) -> list[list[RetrievedChunk]]:
    """
    使用多条查询分别检索，返回各查询的结果列表（供 RRF 融合使用）。

    :param queries:        查询字符串列表
    :param n_results:      每条查询的返回条数
    :param score_threshold: 最低相似度阈值
    :param collection_name: 集合名
    :param user_id:        用户 ID（用于隔离）
    :return:               每条查询的 RetrievedChunk 列表
    """
    import asyncio

    async def _fetch_one(query: str) -> list[RetrievedChunk]:
        try:
            return await retrieve(
                query=query,
                n_results=n_results,
                score_threshold=score_threshold,
                collection_name=collection_name,
                user_id=user_id,
            )
        except Exception as e:
            logger.warning(f"[RAG] 子查询检索失败: {query[:40]!r}: {e}")
            return []

    tasks = [_fetch_one(q) for q in queries]
    results = await asyncio.gather(*tasks)
    return list(results)


# 路径 A：关键词召回（jieba 分词 + PostgreSQL ILIKE）

# jieba 通用停用词表
_KEYWORD_STOP_WORDS: set[str] = {
    "的", "了", "是", "在", "和", "就", "都", "而", "及", "与",
    "着", "或", "一个", "没有", "我们", "你们", "他们", "它们",
    "这个", "那个", "哪些", "哪", "什么", "怎么", "如何", "为什么",
    "可以", "能够", "应该", "需要", "会", "将", "要", "也", "还",
    "更", "最", "很", "非常", "比较", "不", "之", "其", "以", "从",
    "到", "对", "把", "被", "让", "向", "由", "于", "因", "为",
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "can", "shall",
    "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "it", "its", "this", "that", "these", "those", "or", "and",
    "but", "not", "no", "if", "so", "as", "than", "then", "just",
}


def _tokenize_keywords(query: str) -> list[str]:
    """使用 jieba 精确模式分词，过滤停用词和单字。"""
    import jieba

    from backend.config import config

    words = jieba.lcut(query, cut_all=False)
    keywords: list[str] = []
    for w in words:
        w = w.strip()
        if not w or len(w) < config.rag.keyword_min_length:
            continue
        if w.lower() in _KEYWORD_STOP_WORDS:
            continue
        keywords.append(w)
    return keywords


async def retrieve_keyword(
    query: str,
    n_results: int | None = None,
    where: Optional[dict] = None,
    collection_name: Optional[str] = None,
    user_id: Optional[str] = None,
) -> list[RetrievedChunk]:
    """
    关键词检索：使用 jieba 分词后在 document_chunk.text 中做 ILIKE 匹配。

    与向量检索互补——向量擅长语义泛化，关键词擅长精确命中专有名词和代码片段。

    :param query:           用户查询字符串
    :param n_results:       返回条数
    :param where:           额外过滤条件
    :param collection_name: 集合名
    :param user_id:         用户 ID（用于隔离）
    :return:                RetrievedChunk 列表（按匹配关键词数量降序）
    """
    from backend.config import config as _cfg

    _n_results = n_results if n_results is not None else _cfg.rag.n_results

    keywords = _tokenize_keywords(query)
    if not keywords:
        logger.info("[RAG] jieba 分词后无有效关键词，关键词召回返回空")
        return []

    logger.info(f"[RAG] 关键词召回: jieba 分词 → {len(keywords)} 个关键词: {keywords[:10]}")

    # 构建用户隔离过滤条件
    effective_where = where
    if user_id:
        user_filter = {"$or": [{"user_id": user_id}, {"user_id": ""}]}
        if effective_where:
            effective_where = {"$and": [effective_where, user_filter]}
        else:
            effective_where = user_filter

    raw = await query_keyword(
        keywords=keywords,
        n_results=_n_results,
        where=effective_where,
        collection_name=collection_name,
    )

    # 关键词检索分数阈值由 config 控制
    chunks = _parse_results(raw, score_threshold=_cfg.rag.keyword_score_threshold)
    logger.info(f"[RAG] 关键词召回: {len(chunks)} 条结果")
    return chunks


# 混合检索（多路召回 + RRF 融合）

def _rrf_fusion_cross_path(
    path_results: dict[str, list[RetrievedChunk]],
    k: int = 60,
    weights: Optional[dict[str, float]] = None,
) -> list[RetrievedChunk]:
    """
    跨召回路径的 RRF（Reciprocal Rank Fusion）合并。

    各路分数量纲天然不可比（余弦相似度 vs 关键词匹配数），RRF 只看排名，
    避免了归一化问题。

    :param path_results: {"vector": [...], "keyword": [...]}
    :param k:            RRF 平滑常数
    :param weights:      各路权重，如 {"vector": 1.0, "keyword": 1.0}
    :return:             合并去重后的 RetrievedChunk 列表（按 RRF 分降序）
    """
    if weights is None:
        weights = {}

    fused: dict[str, tuple[RetrievedChunk, float]] = {}

    for path_name, ranked_list in path_results.items():
        w = weights.get(path_name, 1.0)
        for rank, chunk in enumerate(ranked_list, 1):
            rrf_score = w / (k + rank)
            if chunk.chunk_id in fused:
                existing_chunk, existing_score = fused[chunk.chunk_id]
                # 保留原始分数更高的 chunk 对象，累加 RRF 分
                if chunk.score > existing_chunk.score:
                    fused[chunk.chunk_id] = (chunk, existing_score + rrf_score)
                else:
                    fused[chunk.chunk_id] = (existing_chunk, existing_score + rrf_score)
            else:
                fused[chunk.chunk_id] = (chunk, rrf_score)

    sorted_results = sorted(fused.values(), key=lambda x: x[1], reverse=True)
    return [chunk for chunk, _ in sorted_results]


async def retrieve_hybrid(
    query: str,
    n_results: int | None = None,
    score_threshold: float | None = None,
    where: Optional[dict] = None,
    collection_name: Optional[str] = None,
    user_id: Optional[str] = None,
) -> list[RetrievedChunk]:
    """
    混合检索：并行执行向量召回 + 关键词召回，RRF 合并后返回 Top-K。

    通过 config.rag.hybrid.paths 控制启用的召回路，
    通过 config.rag.hybrid.enabled 控制总开关。

    预取 3x 候选供 RRF + re-rank 精排使用。

    :param query:           用户查询字符串
    :param n_results:       最终返回条数
    :param score_threshold: 向量召回的余弦相似度阈值
    :param where:           额外过滤条件
    :param collection_name: 集合名
    :param user_id:         用户 ID（用于隔离）
    :return:                合并去重后的 RetrievedChunk 列表
    """
    import asyncio

    from backend.config import config as _cfg

    _n_results = n_results if n_results is not None else _cfg.rag.n_results
    _score_threshold = score_threshold if score_threshold is not None else _cfg.rag.score_threshold
    hybrid_cfg = _cfg.rag.hybrid
    paths = hybrid_cfg.paths

    prefetch_count = max(_n_results * _cfg.rag.prefetch_multiplier, _cfg.rag.prefetch_min)

    async def _vector_path() -> tuple[str, list[RetrievedChunk]]:
        if "vector" not in paths:
            return ("vector", [])
        try:
            chunks = await retrieve(
                query=query,
                n_results=prefetch_count,
                score_threshold=_score_threshold,
                where=where,
                collection_name=collection_name,
                user_id=user_id,
            )
            return ("vector", chunks)
        except Exception as e:
            logger.warning(f"[RAG] 向量召回路异常: {e}")
            return ("vector", [])

    async def _keyword_path() -> tuple[str, list[RetrievedChunk]]:
        if "keyword" not in paths:
            return ("keyword", [])
        try:
            chunks = await retrieve_keyword(
                query=query,
                n_results=prefetch_count,
                where=where,
                collection_name=collection_name,
                user_id=user_id,
            )
            return ("keyword", chunks)
        except Exception as e:
            logger.warning(f"[RAG] 关键词召回路异常: {e}")
            return ("keyword", [])

    tasks = [_vector_path(), _keyword_path()]
    path_results: dict[str, list[RetrievedChunk]] = {}
    results = await asyncio.gather(*tasks)
    for path_name, chunks in results:
        path_results[path_name] = chunks

    counts = ", ".join(f"{k}={len(v)}" for k, v in path_results.items())
    logger.info(f"[RAG] 混合检索各路召回: {counts}")

    merged = _rrf_fusion_cross_path(
        path_results,
        k=hybrid_cfg.rrf_k,
        weights={
            "vector": hybrid_cfg.vector_weight,
            "keyword": hybrid_cfg.keyword_weight,
        },
    )

    if merged:
        merged = _rerank_by_keyword_overlap(query, merged)
        merged = await _resolve_parent_chunks(merged)
        merged = merged[:_n_results]

    if not merged:
        logger.info(f"[RAG] 混合检索无结果，query={query[:60]!r}")
    else:
        logger.info(
            f"[RAG] 混合检索完成: {len(merged)} 条, "
            f"最高分={merged[0].score:.3f}, 最低分={merged[-1].score:.3f}"
        )
    return merged


# 上下文格式化


def format_context(chunks: list[RetrievedChunk], max_tokens: int | None = None) -> str:
    """
    将检索结果格式化为 LLM prompt 上下文字符串，附带来源引用编号。
    超过 max_tokens 估算时截断。

    增强特性：
    - 多样感知排序：优先展示不同节/子主题的 chunk，避免 top-k 全是同一角度
    - Citation 安全：截断时标注实际展示条数与总检索条数

    格式示例：
    [1] （来源：chapter_01.pdf, 第2页）
    梯度下降是一种...

    [2] （来源：notes.md, 第一章）
    反向传播算法...
    """
    context, _ = format_context_with_sources(chunks, max_tokens)
    return context


def format_context_with_sources(
    chunks: list[RetrievedChunk], max_tokens: int | None = None
) -> tuple[str, list[CitationSource]]:
    """
    与 format_context 相同的格式化逻辑，但额外返回实际展示的来源清单。

    返回的 CitationSource.index 与 context 字符串中的 [n] 编号严格对应
    （经多样化排序 + token 截断后定稿）。供生成 Agent 在文末程序化追加
    「参考资料」清单，保证正文 [n] 标记与清单条目一一对齐、不出现悬空编号。

    :return: (context 字符串, CitationSource 列表)
    """
    from backend.config import config as _cfg
    _max_tokens = max_tokens if max_tokens is not None else _cfg.rag.context_max_tokens

    if not chunks:
        logger.warning("[RAG] format_context 收到空 chunks，LLM 将在无参考资料的情况下生成内容。")
        return "（暂无参考资料）", []

    # 按分数排序后，重新排列：优先保留不同 section 的 chunk
    # 避免 top-k 的 5 条全来自同一节的"定义"部分
    diverse_chunks = _diversify_order(chunks)

    parts: list[str] = []
    sources: list[CitationSource] = []
    estimated_tokens = 0
    shown_count = 0
    for i, chunk in enumerate(diverse_chunks):
        source_info = f"来源：{chunk.source}"
        if chunk.page:
            source_info += f"，第 {chunk.page} 页"
        if chunk.section:
            source_info += f"，{chunk.section}"
        # 附加扩展元数据（若有）
        extra_info_parts = []
        if chunk.metadata.get("chunk_type"):
            type_labels = {"definition": "定义", "theorem": "定理", "example": "示例",
                          "exercise": "习题", "summary": "总结"}
            ct = chunk.metadata["chunk_type"]
            extra_info_parts.append(type_labels.get(ct, ct))
        if chunk.metadata.get("language"):
            lang_labels = {"zh": "中文", "en": "英文", "mixed": "中英混合"}
            lang = chunk.metadata["language"]
            extra_info_parts.append(lang_labels.get(lang, lang))
        if chunk.metadata.get("difficulty"):
            diff_labels = {"beginner": "入门", "intermediate": "进阶", "advanced": "高级"}
            diff = chunk.metadata["difficulty"]
            extra_info_parts.append(diff_labels.get(diff, diff))
        if extra_info_parts:
            source_info += " [" + ", ".join(extra_info_parts) + "]"
        entry = f"[{shown_count + 1}] （{source_info}）\n{chunk.text}"
        entry_tokens = _estimate_tokens(entry)
        if estimated_tokens + entry_tokens > _max_tokens:
            break
        parts.append(entry)
        sources.append(CitationSource(
            index=shown_count + 1,
            source=chunk.source,
            page=chunk.page,
            section=chunk.section,
        ))
        estimated_tokens += entry_tokens
        shown_count += 1

    result = "\n\n".join(parts)

    # 如果截断了 chunk，在末尾显式告知 LLM 实际展示范围
    total = len(diverse_chunks)
    if shown_count < total:
        result += (
            f"\n\n[注意：以上仅展示了前 {shown_count} 条参考资料，"
            f"共检索到 {total} 条。请勿引用 [{shown_count + 1}] 及更大的编号。]"
        )
        logger.info(
            f"[RAG] format_context 截断: shown={shown_count}/{total}, "
            f"estimated_tokens={estimated_tokens}/{_max_tokens}"
        )

    return result, sources


def _diversify_order(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """
    多样感知排序：在保持相关性优先的前提下，将来自不同 section 的 chunk
    交错排列，避免 top-k 全部集中在一个子主题。
    """
    if len(chunks) <= 2:
        return list(chunks)

    # 按 section 分组（保留每组内的分数排序）
    sections: dict[str, list[RetrievedChunk]] = {}
    no_section: list[RetrievedChunk] = []
    for c in chunks:
        sec = c.section.strip() if c.section else ""
        if sec:
            sections.setdefault(sec, []).append(c)
        else:
            no_section.append(c)

    # 如果没有足够多的不同 section，回退到原始排序
    if len(sections) < 2:
        return list(chunks)

    # Round-robin：轮流从每个 section 取一个 chunk
    result: list[RetrievedChunk] = []
    section_keys = list(sections.keys())
    idx = [0] * len(section_keys)

    while len(result) < len(chunks):
        added = False
        for i, key in enumerate(section_keys):
            if idx[i] < len(sections[key]):
                result.append(sections[key][idx[i]])
                idx[i] += 1
                added = True
        if not added:
            break

    result.extend(no_section)

    return result


# 内部辅助

def _estimate_tokens(text: str) -> int:
    """按语言比例估算 token 数。"""
    import re

    from backend.config import config
    cn_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    en_chars = len(text) - cn_chars
    return int(cn_chars / config.chat.token_estimation.cn_chars_per_token + en_chars / config.chat.token_estimation.en_chars_per_token)


async def _resolve_parent_chunks(
    chunks: list[RetrievedChunk],
) -> list[RetrievedChunk]:
    """
    将子块检索结果映射到父块文本，按父块去重。

    对于有 parent_chunk_id 的子块，查询父块文本并以父块文本替代子块文本，
    同时保留子块的检索分数。同一父块的多个子块只返回分数最高的那一个。

    对于无 parent_chunk_id 的子块（旧数据或 parent_chunking 未启用），
    直接保留原始子块。

    :param chunks: re-rank 后的子块列表（已按分数降序）
    :return:       父块回填 + 去重后的列表
    """
    parent_ids: list[str] = []
    child_to_parent: dict[str, str] = {}  # child_chunk_id → parent_chunk_id
    seen_parents: set[str] = set()

    for c in chunks:
        pid = c.metadata.get("parent_chunk_id", "")
        if pid:
            child_to_parent[c.chunk_id] = pid

    if not child_to_parent:
        return chunks  # 无父子关系，直接返回

    parent_ids = list(set(child_to_parent.values()))
    parent_texts = await _get_parent_texts_batch(parent_ids)

    resolved: list[RetrievedChunk] = []
    for c in chunks:
        pid = child_to_parent.get(c.chunk_id, "")
        if pid:
            if pid in seen_parents:
                continue  # 去重：同一父块只保留第一个（分数最高）
            seen_parents.add(pid)

            parent_text = parent_texts.get(pid)
            if parent_text:
                # 用父块文本替代子块文本，保留子块的分数和来源信息
                resolved.append(RetrievedChunk(
                    chunk_id=pid,
                    text=parent_text,
                    score=c.score,
                    doc_id=c.doc_id,
                    source=c.source,
                    page=c.page,
                    section=c.section,
                    metadata={
                        **c.metadata,
                        "from_child_chunk": c.chunk_id,
                    },
                ))
                continue

        # 无父块或父块未找到 → 保留原始子块
        resolved.append(c)

    resolved.sort(key=lambda c: c.score, reverse=True)
    return resolved


async def _get_parent_texts_batch(parent_ids: list[str]) -> dict[str, str]:
    try:
        from backend.db.vector import get_parent_texts
        return await get_parent_texts(parent_ids)
    except Exception:
        return {}


def _rerank_by_keyword_overlap(query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """
    轻量级 re-rank：基于查询关键词与文档文本的重叠度对原始分数加权。

    使用 jieba 分词（与关键词召回路保持一致），过滤停用词和单字后，
    统计每个 chunk 命中 query 关键词的数量，按重叠比例加分。

    不引入额外模型依赖，计算成本极低。
    """
    from backend.config import config

    keywords = _tokenize_keywords(query)

    if not keywords:
        return sorted(chunks, key=lambda c: c.score, reverse=True)

    for chunk in chunks:
        # 保存原始分数（便于后续分析）
        if "cosine_score" not in chunk.metadata:
            chunk.metadata["cosine_score"] = chunk.score

        text_lower = chunk.text.lower()
        overlap = sum(1 for kw in keywords if kw.lower() in text_lower)
        # 关键词重叠加分，上限由 config.rag.re_rank_keyword_boost 控制
        boost = min(overlap / max(len(keywords), 1), 1.0) * config.rag.re_rank_keyword_boost
        chunk.score = round(chunk.score + boost, 4)

    return sorted(chunks, key=lambda c: c.score, reverse=True)


def _parse_results(raw: dict, score_threshold: float) -> list[RetrievedChunk]:
    """将 QueryResult 转换为 RetrievedChunk 列表并过滤。"""
    chunks: list[RetrievedChunk] = []
    ids = (raw.get("ids") or [[]])[0]
    documents = (raw.get("documents") or [[]])[0]
    distances = (raw.get("distances") or [[]])[0]
    metadatas = (raw.get("metadatas") or [[]])[0]
    for cid, doc, dist, meta in zip(ids, documents, distances, metadatas):
        # cosine distance → similarity: score = 1 - distance
        score = 1.0 - float(dist)
        if score < score_threshold:
            continue
        # 提取固定字段以外的扩展元数据
        fixed_keys = {"doc_id", "source", "page", "section", "user_id"}
        extra_meta = {k: v for k, v in meta.items() if k not in fixed_keys}
        chunks.append(
            RetrievedChunk(
                chunk_id=cid,
                text=doc,
                score=score,
                doc_id=meta.get("doc_id", ""),
                source=meta.get("source", ""),
                page=int(meta["page"]) if meta.get("page") else None,
                section=meta.get("section") or None,
                metadata=extra_meta,
            )
        )
    return sorted(chunks, key=lambda c: c.score, reverse=True)

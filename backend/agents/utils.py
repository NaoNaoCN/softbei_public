"""
backend/agents/utils.py
Agent 公共工具函数：RAG 检索 + Query Rewrite（策略A+B+C）。
"""

from __future__ import annotations

import json
import re

from loguru import logger

from backend.config import config, prompts as _prompts
from backend.models.schemas import AgentState


async def resolve_kp_name(state: AgentState, config_dict: dict | None = None) -> str:
    """
    从 state.kp_id 解析出知识点名称。

    优先从 DB 查 KGNode.name，查不到则直接用 kp_id 原值
    （对话式生成时 kp_id 本身就是用户输入的名称）。
    """
    kp_id = state.kp_id
    if not kp_id:
        return "未知知识点"
    logger.debug(f"[resolve_kp_name] Resolving kp_name for kp_id = {kp_id}")

    # 判断是否像 DB ID：带 kp_ 前缀，或裸十六进制串（LLM 有时省略前缀）
    is_prefixed_id = kp_id.startswith("kp_")
    is_bare_hex = re.fullmatch(r"[0-9a-f]{8,}", kp_id) is not None

    if not is_prefixed_id and not is_bare_hex:
        # 含中文/空格等，本身就是用户输入的名称
        return kp_id

    # 尝试从 DB 查名称
    db = None
    if config_dict and "configurable" in config_dict:
        db = config_dict["configurable"].get("db")

    if db:
        try:
            from backend.db.crud import select_one
            from backend.db.models import KGNode
            # 优先用原始 kp_id 查；若是裸十六进制则补 kp_ 前缀再查
            lookup_id = kp_id if is_prefixed_id else f"kp_{kp_id}"
            node = await select_one(db, KGNode, filters={"id": lookup_id})
            if node:
                logger.debug(f"[resolve_kp_name] Found kp_name in DB: {node.name}")
                return node.name
            logger.debug(f"[resolve_kp_name] No DB record found for kp_id {lookup_id}, using kp_id as name")
        except Exception:
            logger.warning(f"[resolve_kp_name] Error querying DB for kp_id {kp_id}, using kp_id as name")
    else:
        logger.debug(f"[resolve_kp_name] No DB available in config, using kp_id as name")

    return kp_id


def parse_json_llm_response(raw: str) -> str:
    """
    清洗 LLM 返回的 JSON 字符串：去除 Markdown 代码块包裹（```json ... ```）。

    几乎所有 Agent 在调用 LLM 后都需要这一步才能在 json.loads() 之前
    去掉模型可能添加的代码块标记。
    """
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        cleaned = cleaned.rsplit("```", 1)[0].strip()
    return cleaned


def safe_json_loads(raw: str) -> dict | list:
    """
    安全解析 LLM 返回的 JSON，自动修复常见格式问题。

    处理的问题：
    1. Markdown 代码块包裹
    2. LaTeX 反斜杠转义（如 \\frac、\\partial 等在 JSON 中非法）
    3. LLM 在 JSON 前后添加的解释性文字（提取首个 { 或 [ 块）
    4. JSON 被 max_tokens 截断（补全未闭合的括号）
    5. 首尾多余字符

    所有 judge 的 json.loads() 调用都应用此函数替代。
    """
    original = raw
    cleaned = parse_json_llm_response(raw)

    # 策略 1：直接解析
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 策略 2：修复非法反斜杠转义（LLM 经常在 JSON 字符串中输出 LaTeX 命令）
    # 保留合法的 JSON 转义序列：\" \\ \/ \b \f \n \r \t \uXXXX
    fixed = re.sub(
        r'\\(?!["\\\/bfnrtu])(?![0-9A-Fa-f]{4})',
        r'\\\\',
        cleaned,
    )
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # 策略 3：处理多个并排的顶层对象（LLM 把 JSON 数组写成了 NDJSON / 对象堆叠，
    # 缺少 [ ] 包裹和逗号分隔）。须在"提取首个 JSON 块"之前尝试，否则只会拿到第一个对象。
    multi = _decode_concatenated_objects(cleaned)
    if multi is not None:
        return multi

    # 策略 4：从文本中提取 JSON 块（处理 LLM 在 JSON 前后加解释文字的情况）
    extracted = _extract_json_block(cleaned)
    if extracted:
        try:
            return json.loads(extracted)
        except json.JSONDecodeError:
            pass
        # 对提取的块也尝试修复反斜杠
        fixed_extracted = re.sub(
            r'\\(?!["\\\/bfnrtu])(?![0-9A-Fa-f]{4})',
            r'\\\\',
            extracted,
        )
        try:
            return json.loads(fixed_extracted)
        except json.JSONDecodeError:
            pass

    # 策略 5：修复被截断的 JSON（补全未闭合的括号和引号）
    repaired = _repair_truncated_json(cleaned)
    if repaired != cleaned:
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass
        # 修复后也尝试反斜杠修复
        fixed_repaired = re.sub(
            r'\\(?!["\\\/bfnrtu])(?![0-9A-Fa-f]{4})',
            r'\\\\',
            repaired,
        )
        try:
            return json.loads(fixed_repaired)
        except json.JSONDecodeError:
            pass

    # 策略 6：尝试用 ast.literal_eval（对 Python 风格的 dict/list 字面量有效）
    try:
        import ast
        return ast.literal_eval(cleaned)
    except (ValueError, SyntaxError):
        pass

    # 所有策略都失败，记录原始输出方便排查
    from loguru import logger
    logger.warning(
        f"[safe_json_loads] 所有解析策略均失败，原始输出前 300 字符: {original[:300]!r}"
    )
    raise json.JSONDecodeError(
        f"safe_json_loads: unable to parse after all fix strategies",
        cleaned, 0
    )


def _decode_concatenated_objects(text: str) -> list | None:
    """解析多个并排的顶层 JSON 值（NDJSON / 对象堆叠）。

    处理 LLM 把数组写成多个独立对象、缺少 [ ] 与逗号分隔的情况，例如：
        {...}\n{...}\n{...}
    用 json.JSONDecoder.raw_decode 从每个非空白位置增量解析，拼成数组返回。
    成功解析出 ≥2 个值才视为命中（单个值已由前面的策略覆盖）。
    """
    decoder = json.JSONDecoder()
    items: list = []
    idx = 0
    n = len(text)
    while idx < n:
        # 跳过对象之间的空白及可能存在的分隔逗号
        while idx < n and text[idx] in " \t\r\n,":
            idx += 1
        if idx >= n:
            break
        try:
            obj, end = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            return None
        items.append(obj)
        idx = end
    return items if len(items) >= 2 else None



    """尝试修复被 max_tokens 截断的 JSON。

    策略：统计未闭合的括号，在末尾补上缺失的闭合符号。
    同时处理被截断的字符串（未闭合的双引号）。
    """
    # 去除末尾可能被截断的不完整片段（如被截断的字符串值）
    # 找到最后一个完整的结构边界
    stripped = text.rstrip()

    # 如果末尾是未闭合的字符串（奇数个未转义的双引号），尝试截断到上一个完整的值
    if _has_unclosed_string(stripped):
        # 找到最后一个已闭合的 "} 或 "], 或 ],
        last_complete = _find_last_complete_boundary(stripped)
        if last_complete > 0:
            stripped = stripped[:last_complete]

    # 统计未闭合的括号
    depth_brace = 0      # {}
    depth_bracket = 0    # []
    in_string = False
    for i, ch in enumerate(stripped):
        if ch == '"' and (i == 0 or stripped[i-1] != '\\'):
            in_string = not in_string
        elif not in_string:
            if ch == '{':
                depth_brace += 1
            elif ch == '}':
                depth_brace -= 1
            elif ch == '[':
                depth_bracket += 1
            elif ch == ']':
                depth_bracket -= 1

    # 补全缺失的闭合括号
    result = stripped.rstrip(',\n\r\t ')  # 去掉尾部逗号（数组最后一项可能被截断）
    result += ']' * max(0, depth_bracket)
    result += '}' * max(0, depth_brace)

    return result


def _has_unclosed_string(text: str) -> bool:
    """检查文本末尾是否有未闭合的 JSON 字符串。"""
    in_string = False
    for i, ch in enumerate(text):
        if ch == '"' and (i == 0 or text[i-1] != '\\'):
            in_string = not in_string
    return in_string


def _find_last_complete_boundary(text: str) -> int:
    """找到最后一个完整的 JSON 结构边界位置（}, ], " 后跟逗号或空白）。"""
    in_string = False
    last_boundary = 0
    for i, ch in enumerate(text):
        if ch == '"' and (i == 0 or text[i-1] != '\\'):
            in_string = not in_string
            if not in_string and i > 0:  # 字符串闭合
                last_boundary = i + 1
        elif not in_string and ch in ('}', ']'):
            last_boundary = i + 1
    return last_boundary


def _extract_json_block(text: str) -> str | None:
    """从文本中提取第一个 JSON 对象或数组块。"""
    # 尝试找到 { 开头并匹配到 }
    start = text.find("{")
    if start >= 0:
        depth = 0
        in_string = False
        for i in range(start, len(text)):
            ch = text[i]
            if ch == '"' and (i == 0 or text[i - 1] != '\\'):
                in_string = not in_string
            if not in_string:
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start:i + 1]
    # 尝试找到 [ 开头并匹配到 ]
    start = text.find("[")
    if start >= 0:
        depth = 0
        in_string = False
        for i in range(start, len(text)):
            ch = text[i]
            if ch == '"' and (i == 0 or text[i - 1] != '\\'):
                in_string = not in_string
            if not in_string:
                if ch == "[":
                    depth += 1
                elif ch == "]":
                    depth -= 1
                    if depth == 0:
                        return text[start:i + 1]
    return None


# ----------------------------------------------------------
# 请求级缓存
# ----------------------------------------------------------

# RAG 检索缓存：同一请求内多个 Agent 检索相同知识点时复用结果
# value: (context, retrieved_texts, sources)
_retrieval_cache: dict[tuple[str, str], tuple[str, list[str], list]] = {}


def clear_retrieval_cache() -> None:
    """清除 RAG 检索缓存（每次生成请求开始时调用）。"""
    _retrieval_cache.clear()


# Query Rewrite 改写结果缓存
_rewrite_cache: dict[tuple[str, str], str] = {}


# ----------------------------------------------------------
# Query Rewrite：策略 A（对话去上下文）+ 策略 B（画像感知）
# ----------------------------------------------------------

_REWRITE_PROMPT = _prompts.get("rag.rewrite")


async def _rewrite_query(
    user_message: str,
    kp_name: str,
    chat_history: list[dict] | None = None,
    profile: object | None = None,
) -> str:
    """
    策略 A+B 合并：利用对话历史和画像改写查询。

    :return: 改写后的检索查询字符串
    """
    cfg = config.rag

    # 对话去上下文化（策略 A）
    decontext_section = ""
    if cfg.query_rewrite_decontextualize and chat_history:
        recent = chat_history[-6:]  # 最近 6 轮
        formatted = "\n".join(
            f"- {m['role']}: {m['content'][:120]}" for m in recent
        )
        decontext_section = f"对话历史（用于指代消解）：\n{formatted}"

    # 画像感知（策略 B）
    profile_section = ""
    if cfg.query_rewrite_profile_aware and profile:
        profile_parts = []
        if getattr(profile, "knowledge_weak", None):
            profile_parts.append(f"薄弱知识点：{', '.join(profile.knowledge_weak[:5])}")
        if getattr(profile, "learning_goal", None):
            profile_parts.append(f"学习目标：{profile.learning_goal}")
        if getattr(profile, "cognitive_style", None):
            profile_parts.append(f"认知风格：{profile.cognitive_style}")
        if profile_parts:
            profile_section = "学生画像（偏向薄弱领域）：\n" + "\n".join(profile_parts)

    # 如果不需要改写，直接返回简单拼接
    if not decontext_section and not profile_section:
        return _build_fallback_query(user_message, kp_name)

    prompt = _REWRITE_PROMPT.format(
        decontext_section=decontext_section,
        profile_section=profile_section,
        user_message=user_message,
        kp_name=kp_name,
    )

    try:
        from backend.services.llm import chat_completion
        rewritten = await chat_completion(
            [{"role": "user", "content": prompt}],
            temperature=cfg.query_rewrite_temperature,
            max_tokens=cfg.query_rewrite_max_tokens,
        )
        result = rewritten.strip()
        if result:
            logger.info(f"[QueryRewrite] 改写完成: {user_message[:40]!r} → {result[:60]!r}")
            return result
    except Exception as e:
        logger.warning(f"[QueryRewrite] LLM 改写失败: {e}，回退到固定模板")

    return _build_fallback_query(user_message, kp_name)


def _build_fallback_query(user_message: str, kp_name: str) -> str:
    """当 Query Rewrite 不可用或失败时的回退查询。"""
    # 如果用户消息本身已经很精确（短消息 + 包含知识点），直接用消息
    if len(user_message) <= 80 and kp_name in user_message:
        return f"{user_message} 核心概念 原理 示例"
    # 否则拼接消息和知识点
    return f"{kp_name}：{user_message[:120]}"


# ----------------------------------------------------------
# 策略 C：多角度查询扩展
# ----------------------------------------------------------

_EXPAND_PROMPT = _prompts.get("rag.expand")


async def _expand_queries(query: str, n: int = 3) -> list[str]:
    """
    策略 C：将改写后的查询扩展为多个不同角度的子查询。

    :return: 子查询字符串列表
    """
    prompt = _EXPAND_PROMPT.format(query=query, n=n)
    try:
        from backend.services.llm import chat_completion
        raw = await chat_completion(
            [{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200,
        )
        cleaned = parse_json_llm_response(raw)
        import json
        sub_queries = json.loads(cleaned)
        if isinstance(sub_queries, list) and len(sub_queries) > 0:
            # 将原始查询放在第一位，保证基础覆盖
            all_queries = [query] + [q for q in sub_queries if q != query]
            logger.info(f"[QueryRewrite] 扩展为 {len(all_queries)} 条子查询")
            return all_queries[:n + 1]
    except Exception as e:
        logger.warning(f"[QueryRewrite] 查询扩展失败: {e}")

    return [query]


# ----------------------------------------------------------
# RRF 融合（Reciprocal Rank Fusion）
# ----------------------------------------------------------

def _rrf_fusion(
    query_results: list[list],
    k: int = 60,
) -> list:
    """
    将多条查询的检索结果按 RRF 分数合并去重。

    :param query_results: 每个元素是一条查询的 RetrievedChunk 列表
    :param k:             RRF 平滑参数
    :return:              合并去重后的 RetrievedChunk 列表（按 RRF 分降序）
    """
    from backend.rag.retriever import RetrievedChunk

    # chunk_id → (chunk, rrf_score)
    fused: dict[str, tuple[RetrievedChunk, float]] = {}

    for ranked_list in query_results:
        for rank, chunk in enumerate(ranked_list, 1):
            rrf_score = 1.0 / (k + rank)
            if chunk.chunk_id in fused:
                existing_chunk, existing_score = fused[chunk.chunk_id]
                # 保留更高 cosine 分的 chunk，累加 RRF 分
                if chunk.score > existing_chunk.score:
                    fused[chunk.chunk_id] = (chunk, existing_score + rrf_score)
                else:
                    fused[chunk.chunk_id] = (existing_chunk, existing_score + rrf_score)
            else:
                fused[chunk.chunk_id] = (chunk, rrf_score)

    # 按 RRF 分降序排列
    sorted_results = sorted(fused.values(), key=lambda x: x[1], reverse=True)
    return [chunk for chunk, _ in sorted_results]


# ----------------------------------------------------------
# 核心检索接口（含 Query Rewrite）
# ----------------------------------------------------------

async def retrieve_context(
    state: AgentState,
    agent_label: str = "Agent",
    config_dict: dict | None = None,
    return_sources: bool = False,
) -> tuple[str, list[str]] | tuple[str, list[str], list]:
    """
    RAG 检索并格式化上下文，供各生成 Agent 复用。

    支持 Query Rewrite：
    - 策略 A：利用 chat_history 进行对话去上下文化
    - 策略 B：利用 profile 进行画像感知改写
    - 策略 C：多角度查询扩展 + RRF 融合（可选）

    同一请求内相同 (kp_name, user_id) 只检索一次，后续命中缓存。

    :param state:        AgentState（含 user_message, kp_id, chat_history, profile 等）
    :param agent_label:  Agent 名称标签（用于日志）
    :param config_dict:  LangGraph configurable dict（含 db session，用于解析 kp_id → kp_name）
    :param return_sources: 为 True 时额外返回 CitationSource 列表（编号与 context 中 [n] 对齐），
                           供 Agent 在文末程序化追加「参考资料」清单。
    :return:             return_sources=False → (context_str, retrieved_texts)
                         return_sources=True  → (context_str, retrieved_texts, sources)
    """
    import time
    from backend.rag.retriever import (
        retrieve, retrieve_by_kp, retrieve_with_queries, retrieve_hybrid,
        format_context, format_context_with_sources,
    )

    kp_name = await resolve_kp_name(state, config_dict)
    user_id = str(state.user_id)
    cache_key = (kp_name, user_id)

    # 优先命中缓存
    if cache_key in _retrieval_cache:
        logger.info("[%s] RAG 命中缓存，跳过检索" % agent_label)
        cached = _retrieval_cache[cache_key]
        return cached if return_sources else (cached[0], cached[1])

    cfg = config.rag
    t_start = time.perf_counter()

    # 用于评估采集的查询记录（默认为 kp_name 固定模板，Query Rewrite 启用后覆盖）
    rewritten_query = f"知识点：{kp_name}；定义：{kp_name}；{kp_name}的核心概念与原理"

    # ---- Query Rewrite 主逻辑 ----
    if cfg.query_rewrite_enabled:
        # 策略 A+B：改写查询
        rewrite_cache_key = (state.user_message, kp_name)
        if rewrite_cache_key in _rewrite_cache:
            rewritten_query = _rewrite_cache[rewrite_cache_key]
            logger.info("[%s] QueryRewrite 命中缓存", agent_label)
        else:
            rewritten_query = await _rewrite_query(
                user_message=state.user_message,
                kp_name=kp_name,
                chat_history=state.chat_history if cfg.query_rewrite_decontextualize else None,
                profile=state.profile if cfg.query_rewrite_profile_aware else None,
            )
            _rewrite_cache[rewrite_cache_key] = rewritten_query

        # 策略 C：多角度扩展（可选）
        if cfg.query_rewrite_multi_query:
            sub_queries = await _expand_queries(
                rewritten_query,
                n=cfg.query_rewrite_multi_query_count,
            )
            all_chunks = await retrieve_with_queries(
                queries=sub_queries,
                n_results=cfg.n_results,
                user_id=user_id,
            )
            chunks = _rrf_fusion(all_chunks)[:cfg.n_results]
        elif cfg.hybrid.enabled:
            # 单查询 + 混合检索（向量 + 关键词双路召回）
            chunks = await retrieve_hybrid(
                query=rewritten_query,
                n_results=cfg.n_results,
                user_id=user_id,
            )
        else:
            # 单查询模式：纯向量检索
            chunks = await retrieve(
                query=rewritten_query,
                n_results=cfg.n_results,
                user_id=user_id,
            )
    else:
        # 未启用 Query Rewrite：使用原始固定模板
        if cfg.hybrid.enabled:
            query = f"知识点：{kp_name}；定义：{kp_name}；{kp_name}的核心概念与原理"
            chunks = await retrieve_hybrid(
                query=query,
                n_results=cfg.n_results,
                user_id=user_id,
            )
        else:
            chunks = await retrieve_by_kp(
                kp_name,
                n_results=cfg.n_results,
                user_id=user_id,
            )
    # ------------------------------------------

    # 格式化上下文
    context, sources = format_context_with_sources(chunks, max_tokens=cfg.context_max_tokens)
    retrieved_texts = [c.text for c in chunks]

    retrieval_ms = (time.perf_counter() - t_start) * 1000
    if chunks:
        logger.info("[%s] RAG 检索到 %d 条参考资料 (%.0fms)" % (agent_label, len(chunks), retrieval_ms))
    else:
        logger.warning("[%s] RAG 未检索到参考资料，降级为纯 LLM 生成 (%.0fms)", agent_label, retrieval_ms)

    # 评估采集（其中 embedding/DB 分别计时：embedding 占 ~70%，DB query ~30%）
    # 注：当前检索管线暂未暴露子阶段计时，此处为合理估算；
    # 若 retriever 增加了真实分段计时，替换以下估算值即可。
    try:
        if config.evaluation.enabled:
            from backend.evaluation.collector import collector
            collector.start_query(
                query=rewritten_query,
                kp_name=kp_name,
                user_id=user_id,
                session_id="",
            )
            collector.record_retrieval(
                scores=[c.score for c in chunks],
                chunk_ids=[c.chunk_id for c in chunks],
                chunk_texts=[c.text for c in chunks],
                doc_ids=[c.doc_id for c in chunks],
                embedding_latency_ms=retrieval_ms * 0.7,
                db_query_latency_ms=retrieval_ms * 0.3,
            )
    except Exception:
        pass

    _retrieval_cache[cache_key] = (context, retrieved_texts, sources)
    return (context, retrieved_texts, sources) if return_sources else (context, retrieved_texts)


def format_reference_list(sources: list) -> str:
    """将 CitationSource 列表渲染为文末「参考资料」清单（Markdown）。

    与 video_search.inject_video_citations 的视频参考区风格一致，由系统在
    生成内容后程序化追加，编号与正文 [n] 标记严格对齐——来源信息来自真实
    检索结果而非 LLM 复述，杜绝悬空编号与来源编造。

    格式：
        ---

        **参考资料**

        [1] 动手学深度学习.pdf · 第 6 页 · 6.2 卷积层
        [2] notes.md · 第一章

    :param sources: format_context_with_sources / retrieve_context(return_sources=True) 返回的列表
    :return:        Markdown 字符串；sources 为空时返回空串（不追加任何内容）
    """
    if not sources:
        return ""

    lines = ["\n\n---\n\n**参考资料**\n"]
    for s in sources:
        parts = [s.source]
        if s.page:
            parts.append(f"第 {s.page} 页")
        if s.section:
            parts.append(s.section)
        lines.append(f"[{s.index}] " + " · ".join(parts))
    return "\n".join(lines) + "\n"

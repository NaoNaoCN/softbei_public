"""
backend/services/video_search.py
视频搜索服务：Bilibili 为主源，Tavily 为备选。
提供 Perplexity 风格的 [v1][v2] 穿插式引用注入。
"""

from __future__ import annotations

import re
import uuid

import httpx
from loguru import logger
from pydantic import BaseModel

from backend.config import config


class VideoResult(BaseModel):
    title: str
    url: str
    thumbnail: str = ""
    duration: str = ""
    source: str  # "bilibili" | "tavily"
    bvid: str = ""


def extract_search_keywords(text: str) -> str:
    """
    从用户消息中提取搜索关键词。
    使用 jieba TF-IDF 提取核心术语。
    """
    import jieba
    import jieba.analyse

    # 确保自定义词典只加载一次
    if not getattr(extract_search_keywords, "_dict_loaded", False):
        # 添加深度学习/计算机领域常见术语
        domain_terms = [
            "激活函数", "损失函数", "目标函数", "代价函数",
            "快速排序", "归并排序", "冒泡排序", "堆排序", "插入排序",
            "反向传播", "梯度下降", "随机梯度下降", "学习率",
            "卷积神经网络", "循环神经网络", "生成对抗网络", "注意力机制",
            "多层感知机", "深度学习", "机器学习", "强化学习", "迁移学习",
            "过拟合", "欠拟合", "正则化", "批归一化", "dropout",
            "二叉树", "红黑树", "哈希表", "链表", "动态规划",
            "面向对象", "设计模式", "数据结构", "操作系统", "计算机网络",
        ]
        for term in domain_terms:
            jieba.add_word(term)
        extract_search_keywords._dict_loaded = True

    # 不限制词性，纯靠 TF-IDF 权重筛选关键词
    keywords = jieba.analyse.extract_tags(text, topK=4)
    # 过滤单字和常见无意义词
    stopwords = {"什么", "怎么", "样子", "部分", "介绍", "讲解", "一下", "一份",
                 "帮我", "生成", "学习", "文档", "资料", "关于", "区别", "对比"}
    keywords = [w for w in keywords if len(w) >= 2 and w not in stopwords]
    if keywords:
        logger.info("[VideoSearch] 提取关键词: {}", " ".join(keywords[:4]))
    return " ".join(keywords[:4]) if keywords else text[:20]


async def search_bilibili(query: str, limit: int = 3) -> list[VideoResult]:
    """Bilibili 搜索 API（需要 cookie 避免 412）"""
    timeout = httpx.Timeout(config.video_search.bilibili_timeout)
    # Bilibili 要求 buvid3 cookie，否则返回 412
    buvid3 = str(uuid.uuid4()) + "infoc"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://search.bilibili.com",
        "Cookie": f"buvid3={buvid3}",
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.get(
            "https://api.bilibili.com/x/web-interface/search/type",
            params={
                "keyword": query,
                "search_type": "video",
                "order": "totalrank",
                "page": 1,
                "pagesize": limit,
            },
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json().get("data", {}).get("result", [])
        return [
            VideoResult(
                title=re.sub(r"<.*?>", "", item.get("title", "")),
                url=f"https://www.bilibili.com/video/{item['bvid']}",
                thumbnail=f"https:{item.get('pic', '')}",
                duration=item.get("duration", ""),
                source="bilibili",
                bvid=item["bvid"],
            )
            for item in (data or [])[:limit]
        ]


async def search_tavily(query: str, limit: int = 3) -> list[VideoResult]:
    """Tavily 备选搜索（需 TAVILY_API_KEY）"""
    api_key = config.video_search.tavily_api_key
    if not api_key:
        return []
    timeout = httpx.Timeout(config.video_search.tavily_timeout)
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": f"{query} 教学视频 讲解",
                "search_depth": "basic",
                "include_domains": ["bilibili.com", "youtube.com"],
                "max_results": limit,
            },
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        return [
            VideoResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                thumbnail="",
                duration="",
                source="tavily",
            )
            for item in results[:limit]
        ]


def extract_topic_from_history(chat_history: list[dict]) -> str:
    """
    从对话历史中提取主题上下文。
    取最后一条 assistant 消息的 Markdown 标题，或前 200 字的关键词。
    """
    for msg in reversed(chat_history):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            # 优先提取 Markdown 标题作为主题
            headings = re.findall(r"^#{1,3}\s+(.+)$", content, re.MULTILINE)
            if headings:
                # 取最后 3 个标题拼接（通常覆盖用户追问涉及的部分）
                heading_text = " ".join(headings[-3:])
                return extract_search_keywords(heading_text)
            # 无标题时，从前 200 字提取
            return extract_search_keywords(content[:200])
    return ""


async def search_videos(query: str, skip_extraction: bool = False) -> list[VideoResult]:
    """搜索入口：提取关键词，Bilibili 优先，失败时 fallback 到 Tavily"""
    if not config.video_search.enabled:
        return []
    if len(query.strip()) < config.video_search.min_query_length:
        return []

    # 提取搜索关键词（调用方已处理好时可跳过）
    logger.info(f"[VideoSearch] 原始查询: {query[:50]}...")
    keywords = query if skip_extraction else extract_search_keywords(query)
    logger.info(f"[VideoSearch] 原始查询: {query[:50]}... → 关键词: {keywords}")

    # 尝试 Bilibili
    try:
        results = await search_bilibili(keywords, limit=config.video_search.max_results)
        if results:
            return results
    except Exception as e:
        logger.warning(f"[VideoSearch] Bilibili 搜索失败: {e}")

    # Fallback: Tavily
    try:
        return await search_tavily(keywords, limit=config.video_search.max_results)
    except Exception as e:
        logger.warning(f"[VideoSearch] Tavily 搜索失败: {e}")

    return []


def inject_video_citations(content: str, videos: list[VideoResult]) -> str:
    """后处理：在 Markdown 段落中插入 [v1] 标记，末尾追加视频参考区"""
    if not videos:
        return content

    lines = content.split("\n")
    video_assigned = [False] * len(videos)

    # 对每个视频标题提取关键词，在正文中找到最佳匹配段落末尾插入标记
    for i, video in enumerate(videos):
        title_words = set(re.findall(r"[\u4e00-\u9fff]{2,}|[a-zA-Z]{3,}", video.title))
        best_line_idx = -1
        best_score = 0
        for idx, line in enumerate(lines):
            if line.startswith("#") or not line.strip():
                continue
            score = sum(1 for w in title_words if w in line)
            if score > best_score:
                best_score = score
                best_line_idx = idx
        if best_line_idx >= 0 and best_score >= 1:
            lines[best_line_idx] = lines[best_line_idx].rstrip() + f" [v{i+1}]"
            video_assigned[i] = True

    # 未匹配的视频标记追加到最后一个非空段落
    unassigned = [i for i, assigned in enumerate(video_assigned) if not assigned]
    if unassigned:
        for idx in range(len(lines) - 1, -1, -1):
            if lines[idx].strip() and not lines[idx].startswith("#"):
                markers = " ".join(f"[v{i+1}]" for i in unassigned)
                lines[idx] = lines[idx].rstrip() + " " + markers
                break

    content = "\n".join(lines)

    # 追加视频参考区
    refs = "\n\n---\n\n**视频参考**\n\n"
    for i, v in enumerate(videos):
        refs += f"**[v{i+1}]** [{v.title}]({v.url})"
        if v.duration:
            refs += f" · {v.duration}"
        refs += f" · {v.source}\n\n"

    return content + refs

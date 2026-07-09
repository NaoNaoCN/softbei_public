"""
tests/test_video_search.py
视频搜索服务单元测试
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from backend.services.video_search import (
    VideoResult,
    search_bilibili,
    search_tavily,
    search_videos,
    inject_video_citations,
    extract_search_keywords,
)


# ============================================================
# inject_video_citations 测试
# ============================================================


class TestInjectVideoCitations:
    def test_empty_videos_returns_original(self):
        content = "这是一段关于快速排序的内容。"
        assert inject_video_citations(content, []) == content

    def test_single_video_matched(self):
        content = "快速排序是一种高效的排序算法。\n\n它使用分治策略来排序。"
        videos = [
            VideoResult(
                title="快速排序算法详解",
                url="https://www.bilibili.com/video/BV123",
                source="bilibili",
            )
        ]
        result = inject_video_citations(content, videos)
        assert "[v1]" in result
        assert "**视频参考**" in result
        assert "https://www.bilibili.com/video/BV123" in result

    def test_multiple_videos(self):
        content = "快速排序使用分治法。\n\n归并排序也是分治法的应用。"
        videos = [
            VideoResult(title="快速排序讲解", url="https://b.com/1", source="bilibili"),
            VideoResult(title="归并排序教程", url="https://b.com/2", source="bilibili"),
        ]
        result = inject_video_citations(content, videos)
        assert "[v1]" in result
        assert "[v2]" in result

    def test_unmatched_videos_appended_to_last_paragraph(self):
        content = "这是一段普通文本。\n\n这是最后一段。"
        videos = [
            VideoResult(title="XYZ完全无关", url="https://b.com/1", source="bilibili"),
        ]
        result = inject_video_citations(content, videos)
        # 无关视频标记应追加到最后一个非空段落
        assert "[v1]" in result
        assert "**视频参考**" in result

    def test_heading_lines_not_modified(self):
        content = "# 标题\n\n正文内容关于排序。"
        videos = [
            VideoResult(title="排序算法", url="https://b.com/1", source="bilibili"),
        ]
        result = inject_video_citations(content, videos)
        # 标记不应出现在标题行
        lines = result.split("\n")
        for line in lines:
            if line.startswith("#"):
                assert "[v" not in line

    def test_duration_shown_in_refs(self):
        content = "内容关于数据结构。"
        videos = [
            VideoResult(
                title="数据结构入门",
                url="https://b.com/1",
                source="bilibili",
                duration="15:30",
            ),
        ]
        result = inject_video_citations(content, videos)
        assert "15:30" in result


# ============================================================
# extract_search_keywords 测试
# ============================================================


class TestExtractSearchKeywords:
    def test_extracts_technical_term(self):
        text = "你第三部分介绍的几种激活函数都是什么样子的，有什么优劣，分别适合什么应用场景？"
        keywords = extract_search_keywords(text)
        assert "激活函数" in keywords

    def test_short_technical_term_unchanged(self):
        keywords = extract_search_keywords("快速排序")
        assert "快速排序" in keywords

    def test_english_terms_preserved(self):
        keywords = extract_search_keywords("请帮我讲解一下ReLU和Sigmoid的区别")
        # jieba 应能提取英文术语
        lower = keywords.lower()
        assert "relu" in lower or "sigmoid" in lower

    def test_limits_keyword_count(self):
        text = "深度学习中的卷积神经网络和循环神经网络以及注意力机制和Transformer架构的对比"
        keywords = extract_search_keywords(text)
        parts = keywords.split()
        assert len(parts) <= 4


# ============================================================
# search_videos 测试（mock 网络调用）
# ============================================================


class TestSearchVideos:
    @pytest.mark.asyncio
    async def test_disabled_returns_empty(self):
        with patch("backend.services.video_search.config") as mock_config:
            mock_config.video_search.enabled = False
            result = await search_videos("快速排序")
            assert result == []

    @pytest.mark.asyncio
    async def test_short_query_returns_empty(self):
        with patch("backend.services.video_search.config") as mock_config:
            mock_config.video_search.enabled = True
            mock_config.video_search.min_query_length = 2
            result = await search_videos("x")
            assert result == []

    @pytest.mark.asyncio
    async def test_bilibili_success(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": {
                "result": [
                    {
                        "title": "<em>快速排序</em>算法",
                        "bvid": "BV1test123",
                        "pic": "//i0.hdslb.com/test.jpg",
                        "duration": "10:25",
                    }
                ]
            }
        }
        mock_response.raise_for_status = MagicMock()

        with patch("backend.services.video_search.config") as mock_config:
            mock_config.video_search.enabled = True
            mock_config.video_search.min_query_length = 2
            mock_config.video_search.max_results = 3
            mock_config.video_search.bilibili_timeout = 5

            with patch("httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()
                mock_client.get = AsyncMock(return_value=mock_response)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=False)
                mock_client_cls.return_value = mock_client

                results = await search_videos("快速排序")
                assert len(results) == 1
                assert results[0].title == "快速排序算法"
                assert results[0].bvid == "BV1test123"
                assert results[0].source == "bilibili"

    @pytest.mark.asyncio
    async def test_bilibili_failure_falls_back_to_tavily(self):
        with patch("backend.services.video_search.config") as mock_config:
            mock_config.video_search.enabled = True
            mock_config.video_search.min_query_length = 2
            mock_config.video_search.max_results = 3
            mock_config.video_search.bilibili_timeout = 5
            mock_config.video_search.tavily_api_key = "test-key"
            mock_config.video_search.tavily_timeout = 8

            with patch(
                "backend.services.video_search.search_bilibili",
                side_effect=Exception("timeout"),
            ):
                mock_tavily_resp = MagicMock()
                mock_tavily_resp.json.return_value = {
                    "results": [
                        {"title": "Tavily Result", "url": "https://example.com"}
                    ]
                }
                mock_tavily_resp.raise_for_status = MagicMock()

                with patch("httpx.AsyncClient") as mock_client_cls:
                    mock_client = AsyncMock()
                    mock_client.post = AsyncMock(return_value=mock_tavily_resp)
                    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                    mock_client.__aexit__ = AsyncMock(return_value=False)
                    mock_client_cls.return_value = mock_client

                    results = await search_videos("快速排序")
                    assert len(results) == 1
                    assert results[0].source == "tavily"

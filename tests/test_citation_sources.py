"""
tests/test_citation_sources.py
验证 format_context_with_sources 的编号对齐 + 截断同步，
以及 format_reference_list 的渲染。

这些是纯函数测试，无 DB/LLM 依赖，可直接运行。
"""

from __future__ import annotations

import re

from backend.rag.retriever import (
    RetrievedChunk,
    CitationSource,
    format_context,
    format_context_with_sources,
)
from backend.agents.utils import format_reference_list


def _make_chunk(i: int, section: str | None = None, page: int | None = None, text: str = "正文内容") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"c{i}",
        text=f"{text}{i}",
        score=1.0 - i * 0.01,
        doc_id=f"doc{i}",
        source=f"动手学深度学习.pdf",
        page=page,
        section=section,
    )


class TestFormatContextWithSources:
    def test_sources_index_matches_context_markers(self):
        """sources 的 index 必须与 context 字符串里的 [n] 编号一致。"""
        chunks = [_make_chunk(i, section=f"第{i}节", page=i) for i in range(1, 4)]
        context, sources = format_context_with_sources(chunks, max_tokens=100000)

        # context 中出现的编号
        markers = sorted(int(m) for m in re.findall(r"\[(\d+)\]", context))
        indices = sorted(s.index for s in sources)
        assert markers == indices == [1, 2, 3]

    def test_sources_preserve_source_page_section(self):
        chunks = [_make_chunk(1, section="6.2 卷积层", page=6)]
        _, sources = format_context_with_sources(chunks, max_tokens=100000)
        assert len(sources) == 1
        s = sources[0]
        assert s.index == 1
        assert s.source == "动手学深度学习.pdf"
        assert s.page == 6
        assert s.section == "6.2 卷积层"

    def test_empty_chunks_returns_empty_sources(self):
        context, sources = format_context_with_sources([], max_tokens=100000)
        assert context == "（暂无参考资料）"
        assert sources == []

    def test_truncation_keeps_sources_in_sync(self):
        """token 截断时，sources 不应包含被截断（未展示）的 chunk，避免悬空编号。"""
        # 每条 chunk 适中长度，小 token 预算只能容纳前几条（非首条即超）
        chunks = [_make_chunk(i, text="内容" * 60) for i in range(1, 6)]
        context, sources = format_context_with_sources(chunks, max_tokens=300)

        # 实际展示条数应 < 总数（发生了截断）
        assert 0 < len(sources) < 5
        # context 里出现的最大编号不超过 sources 数量
        markers = [int(m) for m in re.findall(r"\[(\d+)\]", context) if m.isdigit()]
        # 过滤掉截断提示里的 "[shown+1]" —— 它是提示不可引用的编号
        valid_markers = [m for m in markers if m <= len(sources)]
        assert max(valid_markers) == len(sources)
        # sources 编号连续从 1 开始
        assert [s.index for s in sources] == list(range(1, len(sources) + 1))

    def test_format_context_backward_compatible(self):
        """format_context 仍只返回字符串，与 format_context_with_sources 的 context 一致。"""
        chunks = [_make_chunk(i, section=f"第{i}节") for i in range(1, 4)]
        ctx_only = format_context(chunks, max_tokens=100000)
        ctx_with, _ = format_context_with_sources(chunks, max_tokens=100000)
        assert ctx_only == ctx_with


class TestFormatReferenceList:
    def test_renders_index_source_page_section(self):
        sources = [
            CitationSource(index=1, source="动手学深度学习.pdf", page=6, section="6.2 卷积层"),
            CitationSource(index=2, source="notes.md", section="第一章"),
            CitationSource(index=3, source="slides.pdf", page=12),
        ]
        out = format_reference_list(sources)
        assert "**参考资料**" in out
        assert "[1] 动手学深度学习.pdf · 第 6 页 · 6.2 卷积层" in out
        assert "[2] notes.md · 第一章" in out
        assert "[3] slides.pdf · 第 12 页" in out

    def test_empty_sources_returns_empty_string(self):
        assert format_reference_list([]) == ""

    def test_source_only_no_separators(self):
        sources = [CitationSource(index=1, source="a.pdf")]
        out = format_reference_list(sources)
        assert "[1] a.pdf" in out
        # 没有页码/章节时不应出现悬空的 · 分隔符
        assert "·" not in out.split("[1] a.pdf")[1].split("\n")[0]

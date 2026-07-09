"""
tests/test_study_plan.py
学习计划表模块单元测试：scheduler（纯函数）、collector、sequencer（mock LLM）、resource_linker。
asyncio_mode = auto，无需 @pytest.mark.asyncio。
"""

from __future__ import annotations

import json
from datetime import date
from unittest.mock import AsyncMock, patch

import pytest

from backend.services.study_plan.collector import CandidateKP
from backend.services.study_plan.scheduler import (
    SequencedKP,
    schedule_items,
)
from backend.services.study_plan import sequencer


# ============================================================
# scheduler —— 纯函数，重点单测对象
# ============================================================

class TestScheduler:
    def _kps(self, durations):
        return [
            SequencedKP(kp_id=f"kp_{i}", kp_name=f"知识点{i}", estimated_minutes=d)
            for i, d in enumerate(durations)
        ]

    def test_empty_input_returns_empty(self):
        result = schedule_items([], daily_time_minutes=60, start_date=date(2026, 6, 5))
        assert result.items == []
        assert result.start_date == date(2026, 6, 5)
        assert result.end_date == date(2026, 6, 5)

    def test_daily_budget_not_exceeded(self):
        # 4 个 40min 任务，日预算 100 → 每天最多 2 个
        kps = self._kps([40, 40, 40, 40])
        result = schedule_items(kps, daily_time_minutes=100, start_date=date(2026, 6, 5))

        by_date: dict = {}
        for it in result.items:
            by_date.setdefault(it.scheduled_date, 0)
            by_date[it.scheduled_date] += it.estimated_minutes
        for total in by_date.values():
            assert total <= 100

    def test_rolls_to_next_day_when_full(self):
        kps = self._kps([40, 40, 40, 40])
        result = schedule_items(kps, daily_time_minutes=100, start_date=date(2026, 6, 5))
        dates = sorted({it.scheduled_date for it in result.items})
        # 2 天装下 4 个任务
        assert len(dates) == 2
        assert dates[0] == date(2026, 6, 5)
        assert dates[1] == date(2026, 6, 6)

    def test_oversized_kp_occupies_its_own_day(self):
        # 单点 200min > 日预算 60 → 独占一天，不与他人共享
        kps = [
            SequencedKP(kp_id="a", kp_name="大", estimated_minutes=200),
            SequencedKP(kp_id="b", kp_name="小", estimated_minutes=30),
        ]
        result = schedule_items(kps, daily_time_minutes=60, start_date=date(2026, 6, 5))
        big = next(i for i in result.items if i.kp_name == "大")
        small = next(i for i in result.items if i.kp_name == "小")
        assert big.scheduled_date != small.scheduled_date

    def test_time_slots_generated_and_non_overlapping(self):
        kps = self._kps([30, 30])  # 同一天 19:00-19:30, 19:30-20:00
        result = schedule_items(
            kps, daily_time_minutes=120, start_date=date(2026, 6, 5),
            default_start_hour="19:00",
        )
        same_day = [i for i in result.items if i.scheduled_date == date(2026, 6, 5)]
        assert same_day[0].start_time == "19:00"
        assert same_day[0].end_time == "19:30"
        assert same_day[1].start_time == "19:30"
        assert same_day[1].end_time == "20:00"
        # 第一个的结束 == 第二个的开始（不重叠）
        assert same_day[0].end_time == same_day[1].start_time

    def test_no_time_slots_when_start_hour_absent(self):
        kps = self._kps([30, 30])
        result = schedule_items(kps, daily_time_minutes=120, start_date=date(2026, 6, 5))
        assert all(i.start_time is None and i.end_time is None for i in result.items)

    def test_days_param_compresses_schedule(self):
        # 6 个 60min 任务 = 360min；days=2 → 日预算抬到 180 → 2 天完成
        kps = self._kps([60] * 6)
        result = schedule_items(
            kps, daily_time_minutes=60, start_date=date(2026, 6, 5), days=2,
        )
        dates = {it.scheduled_date for it in result.items}
        assert len(dates) <= 2

    def test_order_index_resets_per_day(self):
        kps = self._kps([40, 40, 40, 40])
        result = schedule_items(kps, daily_time_minutes=100, start_date=date(2026, 6, 5))
        for d in {it.scheduled_date for it in result.items}:
            day_items = sorted(
                [i for i in result.items if i.scheduled_date == d],
                key=lambda x: x.order_index,
            )
            assert [i.order_index for i in day_items] == list(range(len(day_items)))


# ============================================================
# sequencer._fallback —— 确定性回退（无 LLM）
# ============================================================

class TestSequencerFallback:
    def test_weak_first_mastered_last(self):
        cands = [
            CandidateKP(kp_id="m", kp_name="已掌握点", is_mastered=True),
            CandidateKP(kp_id="n", kp_name="普通点"),
            CandidateKP(kp_id="w", kp_name="薄弱点", is_weak=True),
        ]
        result = sequencer._fallback(cands)
        names = [r.kp_name for r in result]
        assert names.index("薄弱点") < names.index("普通点") < names.index("已掌握点")

    def test_fallback_preserves_all_candidates(self):
        cands = [CandidateKP(kp_id=f"k{i}", kp_name=f"点{i}") for i in range(5)]
        result = sequencer._fallback(cands)
        assert len(result) == 5


# ============================================================
# sequencer.sequence_candidates —— mock LLM
# ============================================================

class TestSequenceCandidates:
    async def test_empty_returns_empty(self):
        result = await sequencer.sequence_candidates([], "画像")
        assert result == []

    async def test_parses_llm_json(self):
        cands = [
            CandidateKP(kp_id="kp_a", kp_name="栈"),
            CandidateKP(kp_id="kp_b", kp_name="队列"),
        ]
        llm_out = json.dumps([
            {"kp_id": "kp_b", "kp_name": "队列", "estimated_minutes": 50, "priority": "high", "tip": "先学队列"},
            {"kp_id": "kp_a", "kp_name": "栈", "estimated_minutes": 40, "priority": "medium", "tip": ""},
        ])
        with patch(
            "backend.services.llm.chat_completion",
            new=AsyncMock(return_value=llm_out),
        ):
            result = await sequencer.sequence_candidates(cands, "画像")
        assert [r.kp_name for r in result] == ["队列", "栈"]
        assert result[0].estimated_minutes == 50

    async def test_filters_fabricated_kp(self):
        # LLM 编造了候选列表外的知识点 → 必须被丢弃
        cands = [CandidateKP(kp_id="kp_a", kp_name="栈")]
        llm_out = json.dumps([
            {"kp_id": "kp_a", "kp_name": "栈", "estimated_minutes": 40},
            {"kp_id": "kp_fake", "kp_name": "不存在的点", "estimated_minutes": 30},
        ])
        with patch(
            "backend.services.llm.chat_completion",
            new=AsyncMock(return_value=llm_out),
        ):
            result = await sequencer.sequence_candidates(cands, "画像")
        names = [r.kp_name for r in result]
        assert "不存在的点" not in names
        assert "栈" in names

    async def test_falls_back_on_invalid_json(self):
        cands = [CandidateKP(kp_id="kp_a", kp_name="栈")]
        with patch(
            "backend.services.llm.chat_completion",
            new=AsyncMock(return_value="这不是JSON"),
        ):
            result = await sequencer.sequence_candidates(cands, "画像")
        # 回退仍产出全部候选
        assert len(result) == 1
        assert result[0].kp_name == "栈"

    async def test_missing_candidates_are_backfilled(self):
        # LLM 只返回部分候选 → 缺失项应被补齐，知识点不丢失
        cands = [
            CandidateKP(kp_id="kp_a", kp_name="栈"),
            CandidateKP(kp_id="kp_b", kp_name="队列"),
        ]
        llm_out = json.dumps([{"kp_id": "kp_a", "kp_name": "栈", "estimated_minutes": 40}])
        with patch(
            "backend.services.llm.chat_completion",
            new=AsyncMock(return_value=llm_out),
        ):
            result = await sequencer.sequence_candidates(cands, "画像")
        names = {r.kp_name for r in result}
        assert names == {"栈", "队列"}

    async def test_clamps_estimated_minutes(self):
        from backend.config import config
        cands = [CandidateKP(kp_id="kp_a", kp_name="栈")]
        # 远超上限的预估值应被钳制
        llm_out = json.dumps([{"kp_id": "kp_a", "kp_name": "栈", "estimated_minutes": 99999}])
        with patch(
            "backend.services.llm.chat_completion",
            new=AsyncMock(return_value=llm_out),
        ):
            result = await sequencer.sequence_candidates(cands, "画像")
        assert result[0].estimated_minutes == config.study_plan.max_kp_minutes

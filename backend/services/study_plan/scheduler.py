"""确定性日历排程：把有序知识点按每日时间预算装箱到连续日期上。

纯函数模块，不依赖数据库 / LLM，便于单元测试。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional


@dataclass
class SequencedKP:
    """排序后的待排程知识点（scheduler 的输入单元）。"""
    kp_id: Optional[str]
    kp_name: str
    estimated_minutes: int
    priority: str = "medium"
    tip: Optional[str] = None


@dataclass
class ScheduledItem:
    """已排程到具体日期（可含时段）的计划项（scheduler 的输出单元）。"""
    kp_id: Optional[str]
    kp_name: str
    estimated_minutes: int
    scheduled_date: date
    order_index: int
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class ScheduleResult:
    items: list[ScheduledItem] = field(default_factory=list)
    start_date: Optional[date] = None
    end_date: Optional[date] = None


def _parse_hour(hhmm: str) -> int:
    """把 'HH:MM' 解析为当天起始分钟偏移；非法输入回退到 19:00。"""
    try:
        h, m = hhmm.split(":")
        return int(h) * 60 + int(m)
    except (ValueError, AttributeError):
        return 19 * 60


def _fmt_time(total_minutes: int) -> str:
    """把当天分钟偏移格式化为 'HH:MM'（对 24h 取模，跨天少见但需防御）。"""
    total_minutes %= 24 * 60
    return f"{total_minutes // 60:02d}:{total_minutes % 60:02d}"


def schedule_items(
    kps: list[SequencedKP],
    daily_time_minutes: int,
    start_date: date,
    default_start_hour: Optional[str] = None,
    days: Optional[int] = None,
) -> ScheduleResult:
    """
    将有序知识点贪心装箱到连续日期。

    规则：
    - 逐天累加 estimated_minutes，超过当日预算则滚到下一天。
    - 单个知识点不跨天切分（保持原子）：若单点时长超过日预算，则该点独占一天。
    - 若指定 days，则按 ceil(总时长 / days) 重算每日预算，使计划压缩到目标天数附近。
    - default_start_hour 给定时按当天累计偏移生成 start_time/end_time；否则时段留空（纯日期清单）。

    Args:
        kps: 已按学习顺序排好的知识点列表
        daily_time_minutes: 每日学习时间预算（分钟）
        start_date: 计划起始日期
        default_start_hour: 'HH:MM'，给定则生成时段
        days: 目标天数，给定则均摊日预算

    Returns:
        ScheduleResult（items 按 (date, order_index) 自然有序）
    """
    if not kps:
        return ScheduleResult(items=[], start_date=start_date, end_date=start_date)

    budget = max(1, int(daily_time_minutes or 0))

    # 指定 days 时按总时长均摊重算每日预算
    if days and days > 0:
        total = sum(max(1, k.estimated_minutes) for k in kps)
        budget = max(budget, -(-total // days))  # ceil division，且不低于原预算

    start_offset = _parse_hour(default_start_hour) if default_start_hour else None

    items: list[ScheduledItem] = []
    current_date = start_date
    day_used = 0
    order_in_day = 0

    for kp in kps:
        dur = max(1, int(kp.estimated_minutes or 1))

        # 当天放不下且当天已有内容 → 滚到下一天（单点超预算时独占当天，不再滚）
        if day_used > 0 and day_used + dur > budget:
            current_date = current_date + timedelta(days=1)
            day_used = 0
            order_in_day = 0

        start_time = end_time = None
        if start_offset is not None:
            start_time = _fmt_time(start_offset + day_used)
            end_time = _fmt_time(start_offset + day_used + dur)

        items.append(
            ScheduledItem(
                kp_id=kp.kp_id,
                kp_name=kp.kp_name,
                estimated_minutes=dur,
                scheduled_date=current_date,
                order_index=order_in_day,
                start_time=start_time,
                end_time=end_time,
                notes=kp.tip,
            )
        )

        day_used += dur
        order_in_day += 1

    return ScheduleResult(
        items=items,
        start_date=start_date,
        end_date=items[-1].scheduled_date,
    )

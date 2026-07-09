"""
backend/services/study_plan/sequencer.py
知识点智能排序：调用 LLM 对候选知识点排序并预估学习时长。

LLM 失败 / JSON 解析失败时回退到候选原始顺序 + 默认时长，
保证不依赖 LLM 也能产出可用计划。
"""

from __future__ import annotations

import json

from loguru import logger

from backend.config import config, prompts as _prompts
from backend.agents.utils import parse_json_llm_response
from backend.services.study_plan.collector import CandidateKP
from backend.services.study_plan.scheduler import SequencedKP


def _clamp_minutes(value, default: int) -> int:
    """把 LLM 预估时长钳制到 [min, max] 区间，非法值回退默认。"""
    sp = config.study_plan
    try:
        v = int(value)
    except (TypeError, ValueError):
        v = default
    return max(sp.min_kp_minutes, min(sp.max_kp_minutes, v))


def _fallback(candidates: list[CandidateKP]) -> list[SequencedKP]:
    """LLM 不可用时的确定性回退：保持候选顺序（薄弱点提前），用默认时长。"""
    default_min = config.study_plan.default_kp_minutes
    # 薄弱点提前、已掌握点靠后，其余保持原序（稳定排序）
    def _rank(c: CandidateKP) -> int:
        if c.is_weak:
            return 0
        if c.is_mastered:
            return 2
        return 1

    ordered = sorted(candidates, key=_rank)
    return [
        SequencedKP(
            kp_id=c.kp_id,
            kp_name=c.kp_name,
            estimated_minutes=config.study_plan.min_kp_minutes if c.is_mastered else default_min,
            priority="high" if c.is_weak else ("low" if c.is_mastered else "medium"),
            tip=None,
        )
        for c in ordered
    ]


def _format_candidates(candidates: list[CandidateKP]) -> str:
    """把候选列表渲染成 prompt 文本：每行 'kp_id | kp_name | 标记'。"""
    lines = []
    for c in candidates:
        tags = []
        if c.is_weak:
            tags.append("薄弱")
        if c.is_mastered:
            tags.append("已掌握")
        if c.from_path:
            tags.append("来自路径")
        tag_str = "、".join(tags) if tags else "新知识点"
        lines.append(f"{c.kp_id or 'null'} | {c.kp_name} | {tag_str}")
    return "\n".join(lines)


async def sequence_candidates(
    candidates: list[CandidateKP],
    profile_text: str,
) -> list[SequencedKP]:
    """
    调用 LLM 对候选知识点排序 + 预估时长。失败时回退到确定性顺序。

    Args:
        candidates: 候选知识点列表
        profile_text: 画像上下文字符串（来自 profile_svc.build_profile_context）

    Returns:
        SequencedKP 列表（已按建议学习顺序排列）
    """
    if not candidates:
        return []

    # 延迟导入，避免循环依赖
    from backend.services.llm import chat_completion

    template = _prompts.get("agents.study_plan.sequence")
    if not template:
        logger.warning("[StudyPlan.sequencer] 未找到 study_plan.sequence 提示词，使用回退顺序")
        return _fallback(candidates)

    prompt = template.format(
        profile=profile_text or "（暂无画像信息）",
        candidates=_format_candidates(candidates),
    )

    # 合法 kp_id 集合（用于过滤 LLM 编造项）；保留候选的 name→CandidateKP 映射
    valid_ids = {c.kp_id for c in candidates if c.kp_id}
    by_name = {c.kp_name: c for c in candidates}

    try:
        raw = await chat_completion(
            [{"role": "user", "content": prompt}],
            temperature=config.study_plan.sequence.temperature,
            max_tokens=config.study_plan.sequence.max_tokens,
        )
        parsed = json.loads(parse_json_llm_response(raw))
        if not isinstance(parsed, list) or not parsed:
            logger.warning("[StudyPlan.sequencer] LLM 返回非预期结构，使用回退顺序")
            return _fallback(candidates)
    except (json.JSONDecodeError, Exception) as e:
        logger.warning("[StudyPlan.sequencer] LLM 排序失败，使用回退顺序: {}", e)
        return _fallback(candidates)

    result: list[SequencedKP] = []
    used_names: set[str] = set()
    default_min = config.study_plan.default_kp_minutes

    for entry in parsed:
        if not isinstance(entry, dict):
            continue
        kp_id = entry.get("kp_id")
        if kp_id in ("null", "", "None"):
            kp_id = None
        kp_name = entry.get("kp_name")

        # 防编造：有 kp_id 的必须在合法集合内；优先按 name 对回候选
        cand = by_name.get(kp_name)
        if cand is None and kp_id and kp_id in valid_ids:
            cand = next((c for c in candidates if c.kp_id == kp_id), None)
        if cand is None:
            # LLM 编造了列表外知识点 → 丢弃
            logger.debug("[StudyPlan.sequencer] 丢弃列表外知识点: {} ({})", kp_name, kp_id)
            continue
        if cand.kp_name in used_names:
            continue
        used_names.add(cand.kp_name)

        result.append(
            SequencedKP(
                kp_id=cand.kp_id,
                kp_name=cand.kp_name,
                estimated_minutes=_clamp_minutes(entry.get("estimated_minutes"), default_min),
                priority=str(entry.get("priority", "medium")),
                tip=entry.get("tip") or None,
            )
        )

    # LLM 可能漏掉部分候选 → 用回退顺序补齐缺失项，避免知识点丢失
    if len(result) < len(candidates):
        for c in candidates:
            if c.kp_name not in used_names:
                result.append(
                    SequencedKP(
                        kp_id=c.kp_id,
                        kp_name=c.kp_name,
                        estimated_minutes=config.study_plan.min_kp_minutes if c.is_mastered else default_min,
                        priority="high" if c.is_weak else "medium",
                        tip=None,
                    )
                )

    return result or _fallback(candidates)

"""候选知识点收集：汇总用户已有学习路径的知识点，并用画像薄弱点补全缺口。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.crud import select, select_one
from backend.db.models import KGNode, LearningPath


@dataclass
class CandidateKP:
    """待排程的候选知识点。"""
    kp_id: Optional[str]      # 对应 KGNode.id；画像薄弱点无匹配节点时为 None
    kp_name: str
    is_mastered: bool = False  # 路径项已完成 → 标记已掌握
    is_weak: bool = False      # 来自画像薄弱点
    from_path: bool = False    # 来自已有 LearningPath


async def collect_candidates(
    user_id: int,
    db: AsyncSession,
    profile,  # StudentProfileOut | None
    source: str = "aggregate",
    path_ids: Optional[list[int]] = None,
) -> list[CandidateKP]:
    """
    收集候选知识点（去重保序）。

    source == "aggregate"：汇总用户全部学习路径 + 画像薄弱点补全。
    source == "path"：仅汇总 path_ids 指定的学习路径（不做画像补全）。

    Returns:
        CandidateKP 列表，已按"路径出现顺序 → 画像薄弱点"去重保序。
    """
    if source == "path" and path_ids:
        paths = await select(
            db, LearningPath,
            where=LearningPath.id.in_(path_ids),
            loadRelations=["items.kp"],
        )
        # 仅保留属于当前用户的路径
        paths = [p for p in paths if p.user_id == user_id]
    else:
        paths = await select(
            db, LearningPath,
            filters={"user_id": user_id},
            loadRelations=["items.kp"],
        )

    candidates: list[CandidateKP] = []
    seen_ids: set[str] = set()
    seen_names: set[str] = set()

    # 汇总路径知识点（按 order_index 排序，去重保序）
    for path in paths:
        items = sorted(path.items, key=lambda x: x.order_index) if path.items else []
        for item in items:
            kp_id = item.kp_id
            kp_name = item.kp.name if item.kp else kp_id
            if not kp_name:
                continue
            if kp_id and kp_id in seen_ids:
                continue
            if kp_name in seen_names:
                continue
            if kp_id:
                seen_ids.add(kp_id)
            seen_names.add(kp_name)
            candidates.append(
                CandidateKP(
                    kp_id=kp_id,
                    kp_name=kp_name,
                    is_mastered=bool(item.is_completed),
                    from_path=True,
                )
            )

    # 画像薄弱点补全（仅 aggregate 模式）
    if source != "path" and profile is not None:
        weak = getattr(profile, "knowledge_weak", None) or []
        for name in weak:
            if not isinstance(name, str) or not name.strip():
                continue
            name = name.strip()
            if name in seen_names:
                # 已在路径中出现 → 仅补打薄弱标记
                for c in candidates:
                    if c.kp_name == name:
                        c.is_weak = True
                continue
            # 尝试按名称解析 KGNode（公共或本人节点）
            kp_id = await _resolve_kp_id_by_name(name, user_id, db)
            if kp_id and kp_id in seen_ids:
                continue
            if kp_id:
                seen_ids.add(kp_id)
            seen_names.add(name)
            candidates.append(
                CandidateKP(kp_id=kp_id, kp_name=name, is_weak=True, from_path=False)
            )

    logger.info(
        "[StudyPlan.collector] user={} source={} 收集候选知识点 {} 个",
        user_id, source, len(candidates),
    )
    return candidates


async def _resolve_kp_id_by_name(name: str, user_id: int, db: AsyncSession) -> Optional[str]:
    """按名称查 KGNode：优先本人节点，其次公共节点。查不到返回 None。"""
    node = await select_one(db, KGNode, filters={"name": name, "user_id": user_id})
    if node:
        return node.id
    node = await select_one(db, KGNode, filters={"name": name, "user_id": None})
    if node:
        return node.id
    return None

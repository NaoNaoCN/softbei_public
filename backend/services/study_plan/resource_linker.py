"""资源关联：为计划项匹配用户已生成的 ResourceMeta，并计算缺失的资源类型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from sqlalchemy import select as sa_select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import config
from backend.db.models import ResourceMeta


@dataclass
class LinkedResources:
    """单个知识点的资源关联结果。"""
    resource_ids: list[int] = field(default_factory=list)
    missing_resource_types: list[str] = field(default_factory=list)


async def link_resources(
    user_id: int,
    kp_ids: list[Optional[str]],
    db: AsyncSession,
    target_types: Optional[list[str]] = None,
) -> dict[str, LinkedResources]:
    """
    为一批知识点批量匹配已有资源，并计算缺失类型。

    单次 IN 查询拉取所有相关 ResourceMeta，避免逐条查询。

    Args:
        user_id: 用户 ID
        kp_ids: 知识点 ID 列表（可含 None，None 项不匹配）
        db: 数据库会话
        target_types: 目标资源类型集，缺省取 config.study_plan.target_resource_types

    Returns:
        {kp_id: LinkedResources}，仅包含非 None 的 kp_id
    """
    targets = target_types or config.study_plan.target_resource_types
    valid_ids = [k for k in kp_ids if k]
    if not valid_ids:
        return {}

    rows = (
        await db.execute(
            sa_select(
                ResourceMeta.id,
                ResourceMeta.kp_id,
                ResourceMeta.resource_type,
            ).where(
                ResourceMeta.user_id == user_id,
                ResourceMeta.kp_id.in_(valid_ids),
            )
        )
    ).all()

    # 聚合：kp_id → {type: [resource_id, ...]}
    by_kp: dict[str, dict[str, list[int]]] = {}
    for rid, kp_id, rtype in rows:
        by_kp.setdefault(kp_id, {}).setdefault(rtype, []).append(rid)

    result: dict[str, LinkedResources] = {}
    for kp_id in valid_ids:
        type_map = by_kp.get(kp_id, {})
        resource_ids = [rid for rids in type_map.values() for rid in rids]
        missing = [t for t in targets if t not in type_map]
        result[kp_id] = LinkedResources(
            resource_ids=resource_ids,
            missing_resource_types=missing,
        )
    return result

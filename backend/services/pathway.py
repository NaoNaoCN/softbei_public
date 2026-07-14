"""学习路径服务：LearningPath 和 LearningPathItem 的 CRUD 操作。"""

from __future__ import annotations

from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.crud import (
    select,
    select_one,
    insert,
    update_by_id,
    delete,
    delete_by_id,
)
from backend.db.models import LearningPath, LearningPathItem, LearningRecord, KGNode
from backend.models.schemas import (
    LearningPathCreate,
    LearningPathUpdate,
    LearningPathItemCreate,
    LearningPathItemUpdate,
    LearningPathOut,
    LearningPathItemOut,
)


async def get_pathway(
    path_id: int,
    db: AsyncSession,
) -> Optional[LearningPathOut]:
    path = await select_one(
        db, LearningPath,
        filters={"id": path_id},
        loadRelations=["items.kp"],
    )
    if not path:
        return None
    return _path_to_out(path)


async def list_pathways(
    user_id: int,
    db: AsyncSession,
) -> list[LearningPathOut]:
    paths = await select(
        db, LearningPath,
        filters={"user_id": user_id},
        loadRelations=["items.kp"],
    )
    return [_path_to_out(p) for p in paths]


async def create_pathway(
    user_id: int,
    data: LearningPathCreate,
    db: AsyncSession,
) -> LearningPathOut:
    path = await insert(
        db, LearningPath,
        data={"user_id": user_id, "title": data.name, "description": data.description},
    )
    return LearningPathOut(
        id=path.id,
        name=path.title or "",
        description=path.description,
        items=[],
        created_at=path.created_at,
    )


async def update_pathway(
    path_id: int,
    user_id: int,
    data: LearningPathUpdate,
    db: AsyncSession,
) -> Optional[LearningPathOut]:
    path = await select_one(db, LearningPath, filters={"id": path_id, "user_id": user_id})
    if not path:
        return None
    update_data = {}
    if data.name is not None:
        update_data["title"] = data.name
    if data.description is not None:
        update_data["description"] = data.description
    if not update_data:
        return await get_pathway(path_id, db)

    updated = await update_by_id(db, LearningPath, path_id, update_data)
    if not updated:
        return None
    return await get_pathway(path_id, db)


async def delete_pathway(
    path_id: int,
    user_id: int,
    db: AsyncSession,
) -> bool:
    """
    删除学习路径（级联删除 items）。
    先删 items 再删 path。
    """
    path = await select_one(db, LearningPath, filters={"id": path_id, "user_id": user_id})
    if not path:
        return False
    await delete(db, LearningPathItem, {"path_id": path_id}, commit=False)
    return await delete_by_id(db, LearningPath, path_id)


async def add_pathway_item(
    path_id: int,
    user_id: int,
    data: LearningPathItemCreate,
    db: AsyncSession,
) -> Optional[LearningPathItemOut]:
    path = await select_one(db, LearningPath, filters={"id": path_id, "user_id": user_id})
    if not path:
        return None

    kp_node = await select_one(db, KGNode, filters={"id": data.kp_id})
    resolved_kp_id = data.kp_id

    if kp_node and kp_node.user_id and kp_node.user_id != user_id:
        # 节点属于其他用户，尝试按名称查找当前用户的同名节点
        own_node = await select_one(db, KGNode, filters={"name": kp_node.name, "user_id": user_id})
        if own_node:
            resolved_kp_id = own_node.id
            kp_node = own_node
        # 如果找不到同名节点，仍使用原 ID（FK 约束允许跨用户引用）

    kp_name = kp_node.name if kp_node else data.kp_id

    item = await insert(
        db, LearningPathItem,
        data={
            "path_id": path_id,
            "kp_id": resolved_kp_id,
            "order_index": data.order_index,
        },
    )
    return LearningPathItemOut(
        id=item.id,
        order_index=item.order_index,
        kp_id=item.kp_id,
        kp_name=kp_name,
        is_completed=item.is_completed,
    )


async def update_pathway_item(
    item_id: int,
    user_id: int,
    data: LearningPathItemUpdate,
    db: AsyncSession,
) -> Optional[LearningPathItemOut]:
    """更新学习路径项（顺序/完成状态）。"""
    # 验证归属：item -> path -> user_id
    item = await select_one(db, LearningPathItem, filters={"id": item_id}, loadRelations=["kp"])
    if not item:
        return None
    path = await select_one(db, LearningPath, filters={"id": item.path_id, "user_id": user_id})
    if not path:
        return None

    update_data = {}
    if data.order_index is not None:
        update_data["order_index"] = data.order_index
    if data.is_completed is not None:
        update_data["is_completed"] = data.is_completed
    if not update_data:
        return _item_to_out(item)

    await update_by_id(db, LearningPathItem, item_id, update_data)

    # 标记完成时，同步写入 LearningRecord 以便"已学习视图"能识别
    if data.is_completed:
        await insert(db, LearningRecord, data={
            "user_id": user_id,
            "kp_id": item.kp_id,
            "action": "complete",
        })

    updated = await select_one(db, LearningPathItem, filters={"id": item_id}, loadRelations=["kp"])
    return _item_to_out(updated)


async def remove_pathway_item(
    item_id: int,
    user_id: int,
    db: AsyncSession,
) -> bool:
    """从学习路径移除一个知识点项。"""
    item = await select_one(db, LearningPathItem, filters={"id": item_id})
    if not item:
        return False
    path = await select_one(db, LearningPath, filters={"id": item.path_id, "user_id": user_id})
    if not path:
        return False
    return await delete_by_id(db, LearningPathItem, item_id)


def _item_to_out(item: LearningPathItem) -> LearningPathItemOut:
    return LearningPathItemOut(
        id=item.id,
        order_index=item.order_index,
        kp_id=item.kp_id,
        kp_name=item.kp.name if item.kp else item.kp_id,
        is_completed=item.is_completed,
    )


def _path_to_out(path: LearningPath) -> LearningPathOut:
    items = sorted(path.items, key=lambda x: x.order_index) if path.items else []
    return LearningPathOut(
        id=path.id,
        name=path.title or "",
        description=path.description,
        items=[_item_to_out(i) for i in items],
        created_at=path.created_at,
    )

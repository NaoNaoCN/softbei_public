"""数据库基础增删改查封装（基于 SQLAlchemy 异步会话）。"""

from __future__ import annotations

from typing import Any, Sequence, TypeVar

from sqlalchemy import select as sa_select, delete as sa_delete, update as sa_update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload


ModelT = TypeVar("ModelT")


async def insert(
    session: AsyncSession,
    model: type[ModelT],
    data: dict[str, Any],
    commit: bool = True,
    refresh: bool = True,
) -> ModelT:
    """
    插入单条记录。

    Args:
        session: 数据库会话
        model: ORM 模型类
        data: 要插入的字段数据字典
        commit: 是否立即提交，默认 True
        refresh: 是否在 commit 后刷新实例，默认 True（需要 DB 默认值时保持 True）

    Returns:
        新创建的模型实例
    """
    instance = model(**data)
    session.add(instance)
    if commit:
        await session.commit()
        if refresh:
            await session.refresh(instance)
    return instance


async def insert_many(
    session: AsyncSession,
    model: type[ModelT],
    data_list: list[dict[str, Any]],
    commit: bool = True,
) -> list[ModelT]:
    """
    批量插入多条记录。

    Args:
        session: 数据库会话
        model: ORM 模型类
        data_list: 要插入的字段数据字典列表
        commit: 是否立即提交，默认 True

    Returns:
        新创建模型实例列表
    """
    instances = [model(**data) for data in data_list]
    session.add_all(instances)
    if commit:
        await session.commit()
    return instances


async def select(
    session: AsyncSession,
    model: type[ModelT],
    filters: dict[str, Any] | None = None,
    where: Any | None = None,
    order_by: Any | None = None,
    limit: int | None = None,
    offset: int | None = None,
    loadRelations: list[str] | None = None,
) -> Sequence[ModelT]:
    """
    查询记录列表。

    Args:
        session: 数据库会话
        model: ORM 模型类
        filters: 过滤条件字典，键为字段名，值为过滤值（仅支持 == 和 is None）
        where: 额外 SQLAlchemy WHERE 表达式，支持 or_/in_/like/between 等复杂条件
        order_by: 排序字段，如 User.username 或 desc(User.id)
        limit: 返回条数限制
        offset: 跳过条数
        loadRelations: 预加载的关系属性名列表

    Returns:
        模型实例序列
    """
    stmt = sa_select(model)

    if loadRelations:
        for rel in loadRelations:
            # 支持嵌套路径如 "items.kp"，也支持普通关系名
            rel_path = rel.split(".")
            if len(rel_path) == 1:
                stmt = stmt.options(selectinload(getattr(model, rel)))
            else:
                # 多层嵌套：items.kp -> selectinload(model.items).selectinload(ChildModel.kp)
                loader = selectinload(getattr(model, rel_path[0]))
                # 获取第一层关系的目标模型
                parent_rel = getattr(model, rel_path[0]).property
                child_model = parent_rel.mapper.class_
                for part in rel_path[1:]:
                    loader = loader.selectinload(getattr(child_model, part))
                    child_rel = getattr(child_model, part).property
                    child_model = child_rel.mapper.class_
                stmt = stmt.options(loader)

    if filters:
        for key, value in filters.items():
            if value is None:
                stmt = stmt.where(getattr(model, key).is_(None))
            else:
                stmt = stmt.where(getattr(model, key) == value)

    if where is not None:
        stmt = stmt.where(where)

    if order_by is not None:
        stmt = stmt.order_by(order_by)

    if limit is not None:
        stmt = stmt.limit(limit)

    if offset is not None:
        stmt = stmt.offset(offset)

    result = await session.execute(stmt)
    return result.scalars().all()


async def select_one(
    session: AsyncSession,
    model: type[ModelT],
    filters: dict[str, Any] | None = None,
    where: Any | None = None,
    loadRelations: list[str] | None = None,
) -> ModelT | None:
    """
    查询单条记录。

    Args:
        session: 数据库会话
        model: ORM 模型类
        filters: 过滤条件字典
        where: 额外 SQLAlchemy WHERE 表达式
        loadRelations: 预加载的关系属性名列表

    Returns:
        模型实例或 None
    """
    results = await select(
        session, model, filters=filters, where=where, limit=1, loadRelations=loadRelations,
    )
    return results[0] if results else None


async def select_by_id(
    session: AsyncSession,
    model: type[ModelT],
    id: Any,
    loadRelations: list[str] | None = None,
) -> ModelT | None:
    """
    根据主键 ID 查询单条记录。

    Args:
        session: 数据库会话
        model: ORM 模型类
        id: 主键值
        loadRelations: 预加载的关系属性名列表

    Returns:
        模型实例或 None
    """
    return await select_one(
        session, model, filters={"id": id}, loadRelations=loadRelations
    )


async def update_(
    session: AsyncSession,
    model: type[ModelT],
    filters: dict[str, Any],
    data: dict[str, Any],
    commit: bool = True,
) -> int:
    """
    更新符合条件的记录。

    Args:
        session: 数据库会话
        model: ORM 模型类
        filters: 过滤条件字典
        data: 要更新的字段数据字典
        commit: 是否立即提交，默认 True

    Returns:
        实际更新的记录数
    """
    stmt = sa_update(model)
    for key, value in filters.items():
        if value is None:
            stmt = stmt.where(getattr(model, key).is_(None))
        else:
            stmt = stmt.where(getattr(model, key) == value)
    stmt = stmt.values(**data)

    result = await session.execute(stmt)
    if commit:
        await session.commit()
    return result.rowcount


async def update_by_id(
    session: AsyncSession,
    model: type[ModelT],
    id: Any,
    data: dict[str, Any],
    commit: bool = True,
) -> bool:
    """
    根据主键 ID 更新记录。

    Args:
        session: 数据库会话
        model: ORM 模型类
        id: 主键值
        data: 要更新的字段数据字典
        commit: 是否立即提交，默认 True

    Returns:
        是否更新了记录
    """
    rows = await update_(session, model, filters={"id": id}, data=data, commit=commit)
    return rows > 0


async def delete(
    session: AsyncSession,
    model: type[ModelT],
    filters: dict[str, Any],
    commit: bool = True,
) -> int:
    """
    删除符合条件的记录。

    Args:
        session: 数据库会话
        model: ORM 模型类
        filters: 过滤条件字典
        commit: 是否立即提交，默认 True

    Returns:
        实际删除的记录数
    """
    stmt = sa_delete(model)
    for key, value in filters.items():
        if value is None:
            stmt = stmt.where(getattr(model, key).is_(None))
        else:
            stmt = stmt.where(getattr(model, key) == value)

    result = await session.execute(stmt)
    if commit:
        await session.commit()
    return result.rowcount


async def delete_by_id(
    session: AsyncSession,
    model: type[ModelT],
    id: Any,
    commit: bool = True,
) -> bool:
    """
    根据主键 ID 删除记录。

    Args:
        session: 数据库会话
        model: ORM 模型类
        id: 主键值
        commit: 是否立即提交，默认 True

    Returns:
        是否删除了记录
    """
    rows = await delete(session, model, filters={"id": id}, commit=commit)
    return rows > 0


async def delete_user_cascade(session: AsyncSession, user_id: int) -> bool:
    """
    硬删除用户账号及其全部关联数据（注销账号）。

    背景：除 chat_message→chat_session 外，挂在 user.id 上的外键均未配置
    ON DELETE CASCADE，故直接删 user 会触发外键约束错误。本函数在单个事务内
    按「先子表后父表」的依赖顺序逐表清理，任一步失败整体回滚，不留半删脏数据。

    依赖顺序（自底向上）：
        profile_history → student_profile
        quiz_attempt / generation_task / learning_record → quiz_item → resource_meta
        generation_batch
        learning_path_item → learning_path
        study_plan_item → study_plan
        chat_session（chat_message 由 CASCADE 自动删）
        kg_edge → kg_node
        kg_build_task / email_verification
        document_chunk（user_id 为 String 类型，无外键）
        user

    Args:
        session: 数据库会话
        user_id: 要注销的用户 ID

    Returns:
        是否删除了 user 记录（False 表示用户不存在）
    """
    from sqlalchemy import text

    # 延迟导入，避免与 models 形成循环依赖
    from backend.db.models import (
        ChatSession,
        EmailVerification,
        GenerationBatch,
        GenerationTask,
        KGBuildTask,
        KGEdge,
        KGNode,
        LearningPath,
        LearningPathItem,
        LearningRecord,
        ProfileHistory,
        QuizAttempt,
        QuizItem,
        ResourceMeta,
        StudentProfile,
        StudyPlan,
        StudyPlanItem,
        User,
    )

    uid_str = str(user_id)

    # ---- student_profile 子树 ----
    profile_ids = (
        await session.execute(
            sa_select(StudentProfile.id).where(StudentProfile.user_id == user_id)
        )
    ).scalars().all()
    if profile_ids:
        await session.execute(
            sa_delete(ProfileHistory).where(ProfileHistory.profile_id.in_(profile_ids))
        )
    await session.execute(sa_delete(StudentProfile).where(StudentProfile.user_id == user_id))

    # ---- resource_meta 子树（quiz_item / generation_task / learning_record 引用其 id）----
    resource_ids = (
        await session.execute(
            sa_select(ResourceMeta.id).where(ResourceMeta.user_id == user_id)
        )
    ).scalars().all()

    # learning_record 既挂 user_id 又挂 resource_id，先按 user_id 全删
    await session.execute(sa_delete(LearningRecord).where(LearningRecord.user_id == user_id))

    if resource_ids:
        quiz_item_ids = (
            await session.execute(
                sa_select(QuizItem.id).where(QuizItem.resource_id.in_(resource_ids))
            )
        ).scalars().all()
        if quiz_item_ids:
            await session.execute(
                sa_delete(QuizAttempt).where(QuizAttempt.quiz_item_id.in_(quiz_item_ids))
            )
        # 兜底：清理任何仍以该用户为 user_id 的答题记录
        await session.execute(sa_delete(QuizAttempt).where(QuizAttempt.user_id == user_id))
        await session.execute(
            sa_delete(GenerationTask).where(GenerationTask.resource_id.in_(resource_ids))
        )
        await session.execute(
            sa_delete(QuizItem).where(QuizItem.resource_id.in_(resource_ids))
        )
    else:
        await session.execute(sa_delete(QuizAttempt).where(QuizAttempt.user_id == user_id))

    # generation_batch（其下 generation_task 已随 resource 删除，再兜底清 batch_id 引用）
    batch_ids = (
        await session.execute(
            sa_select(GenerationBatch.id).where(GenerationBatch.user_id == user_id)
        )
    ).scalars().all()
    if batch_ids:
        await session.execute(
            sa_delete(GenerationTask).where(GenerationTask.batch_id.in_(batch_ids))
        )
    await session.execute(sa_delete(GenerationBatch).where(GenerationBatch.user_id == user_id))

    await session.execute(sa_delete(ResourceMeta).where(ResourceMeta.user_id == user_id))

    # ---- learning_path 子树 ----
    path_ids = (
        await session.execute(
            sa_select(LearningPath.id).where(LearningPath.user_id == user_id)
        )
    ).scalars().all()
    if path_ids:
        await session.execute(
            sa_delete(LearningPathItem).where(LearningPathItem.path_id.in_(path_ids))
        )
    await session.execute(sa_delete(LearningPath).where(LearningPath.user_id == user_id))

    # ---- study_plan 子树 ----
    plan_ids = (
        await session.execute(
            sa_select(StudyPlan.id).where(StudyPlan.user_id == user_id)
        )
    ).scalars().all()
    if plan_ids:
        await session.execute(
            sa_delete(StudyPlanItem).where(StudyPlanItem.plan_id.in_(plan_ids))
        )
    await session.execute(sa_delete(StudyPlan).where(StudyPlan.user_id == user_id))

    # ---- chat_session（chat_message 经 ON DELETE CASCADE 自动删除）----
    await session.execute(sa_delete(ChatSession).where(ChatSession.user_id == user_id))

    # ---- kg_node 子树（kg_edge 的 source_id/target_id 引用 kg_node）----
    kg_node_ids = (
        await session.execute(
            sa_select(KGNode.id).where(KGNode.user_id == user_id)
        )
    ).scalars().all()
    if kg_node_ids:
        await session.execute(
            sa_delete(KGEdge).where(
                KGEdge.source_id.in_(kg_node_ids) | KGEdge.target_id.in_(kg_node_ids)
            )
        )
    await session.execute(sa_delete(KGNode).where(KGNode.user_id == user_id))

    # ---- 其余直接挂 user_id 的表 ----
    await session.execute(sa_delete(KGBuildTask).where(KGBuildTask.user_id == user_id))
    await session.execute(sa_delete(EmailVerification).where(EmailVerification.user_id == user_id))

    # document_chunk.user_id 是 String 类型且无外键，用字符串匹配
    await session.execute(
        text("DELETE FROM document_chunk WHERE user_id = :uid"), {"uid": uid_str}
    )

    # ---- 最后删除 user 本身 ----
    result = await session.execute(sa_delete(User).where(User.id == user_id))

    await session.commit()
    return result.rowcount > 0

"""学习资源服务：元数据管理、生成任务跟踪、学习记录。"""

from __future__ import annotations

from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import config
from backend.db.crud import select_one, select, insert, update_by_id, delete_by_id
from backend.db.models import ResourceMeta, GenerationTask, GenerationBatch, LearningRecord
from backend.utils.snowflake import generate_id
from backend.models.schemas import (
    BatchGenerateOut,
    BatchGenerateRequest,
    BatchTaskItem,
    GenerateRequest,
    GenerateTaskOut,
    LearningRecordCreate,
    LearningRecordOut,
    ResourceListOut,
    ResourceMetaOut,
    ResourceType,
    TaskStatus,
)


async def get_resource(resource_id: int, db: AsyncSession) -> Optional[ResourceMetaOut]:
    """按 ID 查询资源元数据。"""
    resource = await select_one(db, ResourceMeta, filters={"id": resource_id})
    if not resource:
        return None
    return ResourceMetaOut(
        id=resource.id,
        user_id=resource.user_id,
        kp_id=resource.kp_id,
        resource_type=resource.resource_type,
        title=resource.title or "",
        content=resource.content,
        content_json=resource.content_json,
        created_at=resource.created_at,
    )


async def list_resources(
    user_id: int,
    db: AsyncSession,
    resource_type: Optional[str] = None,
    kp_id: Optional[str] = None,
    skip: int = 0,
    limit: int = None,
) -> ResourceListOut:
    """分页列举用户的资源，可按类型或知识点过滤。"""
    if limit is None:
        limit = config.pagination.default_limit
    import sqlalchemy as sa

    filters = {"user_id": user_id}
    if resource_type:
        filters["resource_type"] = resource_type
    if kp_id:
        filters["kp_id"] = kp_id

    count_query = sa.select(sa.func.count()).select_from(ResourceMeta).where(ResourceMeta.user_id == user_id)
    if resource_type:
        count_query = count_query.where(ResourceMeta.resource_type == resource_type)
    if kp_id:
        count_query = count_query.where(ResourceMeta.kp_id == kp_id)
    total = (await db.execute(count_query)).scalar() or 0

    resources = await select(
        db, ResourceMeta,
        filters=filters,
        limit=limit,
        offset=skip,
    )
    items = [
        ResourceMetaOut(
            id=r.id,
            user_id=r.user_id,
            kp_id=r.kp_id,
            resource_type=r.resource_type,
            title=r.title or "",
            content=r.content,
            content_json=r.content_json,
            created_at=r.created_at,
        )
        for r in resources
    ]
    return ResourceListOut(items=items, total=total)


async def delete_resource(resource_id: int, db: AsyncSession) -> bool:
    """物理删除资源元数据（先删/断开所有引用它的子表行，再删 resource_meta）。"""
    from backend.db.models import GenerationTask, QuizItem, QuizAttempt, LearningRecord
    from sqlalchemy import select as sa_select, delete as sa_delete, update as sa_update

    # quiz_attempt → quiz_item → resource_meta（需按依赖顺序删）
    quiz_item_ids_result = await db.execute(
        sa_select(QuizItem.id).where(QuizItem.resource_id == resource_id)
    )
    quiz_item_ids = [row[0] for row in quiz_item_ids_result]
    if quiz_item_ids:
        await db.execute(sa_delete(QuizAttempt).where(QuizAttempt.quiz_item_id.in_(quiz_item_ids)))
    await db.execute(sa_delete(QuizItem).where(QuizItem.resource_id == resource_id))
    await db.execute(sa_delete(GenerationTask).where(GenerationTask.resource_id == resource_id))
    # learning_record.resource_id 可为 NULL：保留学习行为记录，仅断开对该资源的引用
    await db.execute(
        sa_update(LearningRecord)
        .where(LearningRecord.resource_id == resource_id)
        .values(resource_id=None)
    )
    return await delete_by_id(db, ResourceMeta, resource_id)


async def create_generation_task(
    user_id: int,
    request: GenerateRequest,
    db: AsyncSession,
) -> GenerateTaskOut:
    """
    在数据库中创建一条 pending 状态的生成任务记录，返回任务 ID 供前端轮询进度。
    实际异步执行由 BackgroundTasks / Celery 触发。
    """
    # 解析 kp_id → 知识点名称用于标题
    kp_title = request.kp_id
    if request.kp_id.startswith("kp_"):
        from backend.db.models import KGNode
        node = await select_one(db, KGNode, filters={"id": request.kp_id})
        if node:
            kp_title = node.name
    resource = await insert(
        db, ResourceMeta,
        data={
            "user_id": user_id,
            "kp_id": request.kp_id,
            "resource_type": request.resource_type.value,
            "title": f"{kp_title} — {request.resource_type.value}",
        },
        commit=False,
    )
    await db.flush()  # 确保 resource.id 已生成

    task = await insert(
        db, GenerationTask,
        data={
            "resource_id": resource.id,
            "status": TaskStatus.pending.value,
            "progress": 0,
        },
    )
    return GenerateTaskOut(
        task_id=task.id,
        status=TaskStatus.pending,
        progress=0,
    )


async def get_task_status(task_id: int, db: AsyncSession) -> Optional[GenerateTaskOut]:
    """轮询接口：返回任务当前状态与进度。"""
    task = await select_one(db, GenerationTask, filters={"id": task_id})
    if not task:
        return None
    return GenerateTaskOut(
        task_id=task.id,
        status=TaskStatus(task.status),
        progress=task.progress,
        error_message=task.error_message,
        result_id=task.resource_id,
    )


async def update_task_progress(
    task_id: int,
    progress: int,
    status: TaskStatus,
    db: AsyncSession,
    error_message: Optional[str] = None,
    result_id: Optional[int] = None,
) -> None:
    """由 Agent 执行过程中调用，更新进度与状态。"""
    update_data = {"progress": progress, "status": status.value}
    if error_message is not None:
        update_data["error_message"] = error_message
    if result_id is not None:
        update_data["resource_id"] = result_id
    await update_by_id(db, GenerationTask, task_id, update_data)


async def record_learning(
    user_id: int,
    data: LearningRecordCreate,
    db: AsyncSession,
) -> LearningRecordOut:
    """记录用户对某资源的学习行为（时长、评分、反馈）。"""
    record = await insert(
        db, LearningRecord,
        data={
            "user_id": user_id,
            "resource_id": data.resource_id,
            "kp_id": data.kp_id,
            "action": data.action,
            "duration_seconds": data.duration_seconds,
        },
    )
    return LearningRecordOut.model_validate(record)


async def list_learning_records(
    user_id: int,
    db: AsyncSession,
    skip: int = 0,
    limit: int = None,
    kp_id: Optional[str] = None,
) -> list[LearningRecordOut]:
    """列举用户的学习历史，包含知识点名称。"""
    from sqlalchemy import select as sa_select
    from backend.db.models import KGNode

    if limit is None:
        limit = config.pagination.default_limit

    stmt = (
        sa_select(
            LearningRecord,
            KGNode.name.label("kp_name"),
        )
        .outerjoin(KGNode, LearningRecord.kp_id == KGNode.id)
        .where(LearningRecord.user_id == user_id)
        .order_by(LearningRecord.recorded_at.desc())
        .offset(skip)
        .limit(limit)
    )
    if kp_id is not None:
        stmt = stmt.where(LearningRecord.kp_id == kp_id)

    rows = (await db.execute(stmt)).all()
    results = []
    for record, kp_name in rows:
        out = LearningRecordOut.model_validate(record)
        out.kp_name = kp_name
        results.append(out)
    return results


async def create_batch(
    user_id: int,
    request: BatchGenerateRequest,
    db: AsyncSession,
) -> BatchGenerateOut:
    """
    创建批量生成批次：为每个 resource_type 创建 ResourceMeta + GenerationTask，
    并创建 GenerationBatch 关联所有子任务。
    """
    from backend.db.models import KGNode

    kp_title = request.kp_id
    if request.kp_id.startswith("kp_"):
        node = await select_one(db, KGNode, filters={"id": request.kp_id})
        if node:
            kp_title = node.name

    # 预生成所有 ID，避免中间 flush 等待数据库分配
    num_types = len(request.resource_types)
    batch_id = generate_id()
    resource_ids = [generate_id() for _ in range(num_types)]
    task_ids = [generate_id() for _ in range(num_types)]

    batch = GenerationBatch(
        id=batch_id,
        user_id=user_id,
        kp_id=request.kp_id,
        status=TaskStatus.pending.value,
        progress=0,
        resource_types=[rt.value for rt in request.resource_types],
    )
    db.add(batch)

    resource_instances = [
        ResourceMeta(
            id=resource_ids[i],
            user_id=user_id,
            kp_id=request.kp_id,
            resource_type=rt.value,
            title=f"{kp_title} — {rt.value}",
        )
        for i, rt in enumerate(request.resource_types)
    ]
    db.add_all(resource_instances)

    task_instances = [
        GenerationTask(
            id=task_ids[i],
            resource_id=resource_ids[i],
            batch_id=batch_id,
            status=TaskStatus.pending.value,
            progress=0,
        )
        for i in range(num_types)
    ]
    db.add_all(task_instances)

    await db.commit()

    task_items: list[BatchTaskItem] = [
        BatchTaskItem(
            task_id=task_ids[i],
            resource_type=request.resource_types[i],
            status=TaskStatus.pending,
            progress=0,
        )
        for i in range(num_types)
    ]

    return BatchGenerateOut(
        batch_id=batch_id,
        status=TaskStatus.pending,
        progress=0,
        tasks=task_items,
    )


async def get_batch_status(batch_id: int, db: AsyncSession) -> Optional[BatchGenerateOut]:
    """查询批次状态及所有子任务明细。"""
    batch = await select_one(db, GenerationBatch, filters={"id": batch_id})
    if not batch:
        return None

    tasks = await select(
        db, GenerationTask,
        filters={"batch_id": batch_id},
        loadRelations=["resource"],
    )

    task_items = []
    total_progress = 0
    all_done = True
    any_failed = False

    for t in tasks:
        rt = ResourceType(t.resource.resource_type) if t.resource else ResourceType.doc

        task_items.append(BatchTaskItem(
            task_id=t.id,
            resource_type=rt,
            status=TaskStatus(t.status),
            progress=t.progress,
            result_id=t.resource_id if t.status == TaskStatus.done.value else None,
            error_message=t.error_message,
        ))
        total_progress += t.progress
        if t.status != TaskStatus.done.value:
            all_done = False
        if t.status == TaskStatus.failed.value:
            any_failed = True

    num_tasks = len(tasks) or 1
    avg_progress = total_progress // num_tasks

    if all_done:
        batch_status = TaskStatus.done
    elif any_failed and all(t.status in (TaskStatus.done.value, TaskStatus.failed.value) for t in tasks):
        batch_status = TaskStatus.failed
    elif any(t.status == TaskStatus.running.value for t in tasks):
        batch_status = TaskStatus.running
    else:
        batch_status = TaskStatus(batch.status)

    await update_by_id(db, GenerationBatch, batch_id, {
        "status": batch_status.value,
        "progress": avg_progress,
    })

    return BatchGenerateOut(
        batch_id=batch.id,
        status=batch_status,
        progress=avg_progress,
        tasks=task_items,
    )

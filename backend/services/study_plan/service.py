"""学习计划编排与 CRUD：collector → sequencer → scheduler → resource_linker → 持久化。"""

from __future__ import annotations

from datetime import date
from typing import Optional

from loguru import logger
from sqlalchemy import select as sa_select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import config
from backend.db.crud import (
    insert,
    insert_many,
    select,
    select_one,
    update_by_id,
    delete,
    delete_by_id,
)
from backend.db.models import (
    LearningRecord,
    ResourceMeta,
    StudyPlan,
    StudyPlanItem,
)
from backend.models.schemas import (
    ResourceType,
    StudyPlanGenerateRequest,
    StudyPlanItemOut,
    StudyPlanItemUpdate,
    StudyPlanOut,
    StudyPlanResourceRef,
    StudyPlanUpdate,
)
from backend.services import profile as profile_svc
from backend.services.study_plan.collector import collect_candidates
from backend.services.study_plan.resource_linker import link_resources
from backend.services.study_plan.scheduler import schedule_items
from backend.services.study_plan.sequencer import sequence_candidates


async def generate_study_plan(
    user_id: int,
    req: StudyPlanGenerateRequest,
    db: AsyncSession,
) -> Optional[StudyPlanOut]:
    """
    生成一份学习计划：collector 收集候选 → sequencer LLM 排序 + 时长预估 →
    scheduler 确定性装箱 → linker 匹配已有资源 → 持久化 StudyPlan + StudyPlanItem。

    候选为空时返回 None（调用方据此提示用户先建立学习路径/画像）。
    """
    sp_cfg = config.study_plan
    profile = await profile_svc.get_profile(user_id, db)

    candidates = await collect_candidates(
        user_id, db, profile, source=req.source, path_ids=req.path_ids,
    )
    if not candidates:
        logger.info("[StudyPlan] user={} 无候选知识点，跳过生成", user_id)
        return None

    profile_text = ""
    if profile is not None:
        try:
            profile_text = await profile_svc.build_profile_context(profile)
        except Exception:
            profile_text = ""
    sequenced = await sequence_candidates(candidates, profile_text)
    if not sequenced:
        return None

    daily = (
        req.daily_time_minutes
        or (profile.daily_time_minutes if profile else None)
        or sp_cfg.default_daily_minutes
    )
    start = req.start_date or date.today()
    schedule = schedule_items(
        sequenced,
        daily_time_minutes=daily,
        start_date=start,
        default_start_hour=req.default_start_hour,
        days=req.days,
    )

    kp_ids = [it.kp_id for it in schedule.items]
    links = await link_resources(user_id, kp_ids, db, target_types=sp_cfg.target_resource_types)

    source_path_ids = req.path_ids if req.source == "path" else await _user_path_ids(user_id, db)
    goal = profile.learning_goal if profile else None
    title = req.title or _default_title(goal, candidates)

    plan = await insert(
        db, StudyPlan,
        data={
            "user_id": user_id,
            "title": title,
            "goal": goal,
            "start_date": schedule.start_date,
            "end_date": schedule.end_date,
            "daily_time_minutes": daily,
            "status": "active",
            "source_path_ids": source_path_ids,
        },
        commit=False,
    )
    await db.flush()  # 取得 plan.id

    items_data = []
    for it in schedule.items:
        link = links.get(it.kp_id) if it.kp_id else None
        items_data.append({
            "plan_id": plan.id,
            "kp_id": it.kp_id,
            "kp_name": it.kp_name,
            "scheduled_date": it.scheduled_date,
            "start_time": it.start_time,
            "end_time": it.end_time,
            "estimated_minutes": it.estimated_minutes,
            "order_index": it.order_index,
            "resource_ids": link.resource_ids if link else [],
            "missing_resource_types": link.missing_resource_types if link else list(sp_cfg.target_resource_types),
            "notes": it.notes,
        })
    await insert_many(db, StudyPlanItem, data_list=items_data, commit=False)
    await db.commit()

    logger.success(
        "[StudyPlan] user={} 计划已生成 plan={} items={} 跨度 {}~{}",
        user_id, plan.id, len(items_data), schedule.start_date, schedule.end_date,
    )
    return await get_study_plan(plan.id, user_id, db)


async def _user_path_ids(user_id: int, db: AsyncSession) -> list[int]:
    from backend.db.models import LearningPath
    rows = (await db.execute(
        sa_select(LearningPath.id).where(LearningPath.user_id == user_id)
    )).all()
    return [r[0] for r in rows]


def _default_title(goal: Optional[str], candidates) -> str:
    if goal:
        return f"个性化学习计划 · {goal[:20]}"
    if candidates:
        return f"个性化学习计划 · {candidates[0].kp_name}"
    return "个性化学习计划"


async def get_study_plan(
    plan_id: int,
    user_id: int,
    db: AsyncSession,
) -> Optional[StudyPlanOut]:
    """按 ID 获取单份计划（校验归属），含资源引用解析。"""
    plan = await select_one(
        db, StudyPlan,
        filters={"id": plan_id, "user_id": user_id},
        loadRelations=["items"],
    )
    if not plan:
        return None
    return await _plan_to_out(plan, user_id, db)


async def list_study_plans(
    user_id: int,
    db: AsyncSession,
) -> list[StudyPlanOut]:
    """列举用户的所有计划（按创建时间倒序），含资源引用解析。"""
    plans = await select(
        db, StudyPlan,
        filters={"user_id": user_id},
        order_by=StudyPlan.created_at.desc(),
        loadRelations=["items"],
    )
    return [await _plan_to_out(p, user_id, db) for p in plans]


async def update_study_plan(
    plan_id: int,
    user_id: int,
    data: StudyPlanUpdate,
    db: AsyncSession,
) -> Optional[StudyPlanOut]:
    """更新计划元数据（标题/描述/状态）。"""
    plan = await select_one(db, StudyPlan, filters={"id": plan_id, "user_id": user_id})
    if not plan:
        return None
    update_data = {}
    if data.title is not None:
        update_data["title"] = data.title
    if data.description is not None:
        update_data["description"] = data.description
    if data.status is not None:
        update_data["status"] = data.status
    if update_data:
        await update_by_id(db, StudyPlan, plan_id, update_data)
    return await get_study_plan(plan_id, user_id, db)


async def delete_study_plan(
    plan_id: int,
    user_id: int,
    db: AsyncSession,
) -> bool:
    """删除计划（先删 items 再删 plan）。"""
    plan = await select_one(db, StudyPlan, filters={"id": plan_id, "user_id": user_id})
    if not plan:
        return False
    await delete(db, StudyPlanItem, {"plan_id": plan_id}, commit=False)
    return await delete_by_id(db, StudyPlan, plan_id)


async def update_study_plan_item(
    plan_id: int,
    item_id: int,
    user_id: int,
    data: StudyPlanItemUpdate,
    db: AsyncSession,
) -> Optional[StudyPlanItemOut]:
    """
    更新单个计划项（改期/时段/完成状态/备注）。
    标记完成且 kp_id 非空时，同步写入 LearningRecord 以接入"已学习视图"。
    """
    # 校验归属：item → plan → user
    item = await select_one(db, StudyPlanItem, filters={"id": item_id, "plan_id": plan_id})
    if not item:
        return None
    plan = await select_one(db, StudyPlan, filters={"id": plan_id, "user_id": user_id})
    if not plan:
        return None

    update_data = {}
    if data.scheduled_date is not None:
        update_data["scheduled_date"] = data.scheduled_date
    if data.start_time is not None:
        update_data["start_time"] = data.start_time
    if data.end_time is not None:
        update_data["end_time"] = data.end_time
    if data.order_index is not None:
        update_data["order_index"] = data.order_index
    if data.is_completed is not None:
        update_data["is_completed"] = data.is_completed
    if data.notes is not None:
        update_data["notes"] = data.notes

    if update_data:
        await update_by_id(db, StudyPlanItem, item_id, update_data)

    # 标记完成 → 写学习记录（对齐 pathway.update_pathway_item）
    if data.is_completed and item.kp_id:
        await insert(db, LearningRecord, data={
            "user_id": user_id,
            "kp_id": item.kp_id,
            "action": "complete",
        })

    updated = await select_one(db, StudyPlanItem, filters={"id": item_id})
    return await _item_to_out(updated, user_id, db)


async def get_study_plan_item(
    plan_id: int,
    item_id: int,
    user_id: int,
    db: AsyncSession,
) -> Optional[StudyPlanItemOut]:
    """获取单个计划项（校验归属）。"""
    item = await select_one(db, StudyPlanItem, filters={"id": item_id, "plan_id": plan_id})
    if not item:
        return None
    plan = await select_one(db, StudyPlan, filters={"id": plan_id, "user_id": user_id})
    if not plan:
        return None
    return await _item_to_out(item, user_id, db)


async def generate_resources_for_item(
    plan_id: int,
    item_id: int,
    user_id: int,
    resource_types: list[str],
):
    """后台任务：为计划项生成缺失资源。"""
    from backend.db.database import _session_factory
    from backend.services.generation import run_batch_generation
    from backend.services import resource as resource_svc
    from backend.models.schemas import BatchGenerateRequest, ResourceType

    kp_id = None
    async with _session_factory() as db:
        item = await select_one(db, StudyPlanItem, filters={"id": item_id})
        if not item:
            logger.warning("[StudyPlan] generate_resources: item={} 不存在", item_id)
            return
        kp_id = item.kp_id

    if not kp_id:
        logger.warning("[StudyPlan] generate_resources: item={} 无 kp_id", item_id)
        return

    try:
        types_enum = [ResourceType(t) for t in resource_types]
        async with _session_factory() as db:
            batch_out = await resource_svc.create_batch(user_id, BatchGenerateRequest(
                kp_id=kp_id,
                resource_types=types_enum,
                num_questions=5,
            ), db)

            task_configs = []
            for task_item in batch_out.tasks:
                task_configs.append({
                    "task_id": task_item.task_id,
                    "request": {
                        "kp_id": kp_id,
                        "resource_type": task_item.resource_type.value,
                        "num_questions": 5,
                    },
                })

            await db.commit()

        await run_batch_generation(batch_out.batch_id, user_id, 0, task_configs)
        logger.success("[StudyPlan] item={} types={} 资源生成完成", item_id, resource_types)
    except Exception as e:
        logger.exception("[StudyPlan] item={} 资源生成失败: {}", item_id, e)


async def _resolve_resource_refs(
    resource_ids: list[int],
    db: AsyncSession,
) -> list[StudyPlanResourceRef]:
    """把 resource_id 列表解析为轻量资源引用（不含正文）。"""
    if not resource_ids:
        return []
    rows = (await db.execute(
        sa_select(
            ResourceMeta.id, ResourceMeta.resource_type, ResourceMeta.title,
        ).where(ResourceMeta.id.in_(resource_ids))
    )).all()
    refs = []
    for rid, rtype, title in rows:
        try:
            rt = ResourceType(rtype)
        except ValueError:
            continue
        refs.append(StudyPlanResourceRef(resource_id=rid, resource_type=rt, title=title or ""))
    return refs


async def _item_to_out(
    item: StudyPlanItem,
    user_id: int,
    db: AsyncSession,
) -> StudyPlanItemOut:
    refs = await _resolve_resource_refs(list(item.resource_ids or []), db)
    return StudyPlanItemOut(
        id=item.id,
        kp_id=item.kp_id,
        kp_name=item.kp_name,
        scheduled_date=item.scheduled_date,
        start_time=item.start_time,
        end_time=item.end_time,
        estimated_minutes=item.estimated_minutes,
        order_index=item.order_index,
        is_completed=item.is_completed,
        resources=refs,
        missing_resource_types=list(item.missing_resource_types or []),
        notes=item.notes,
    )


async def _plan_to_out(
    plan: StudyPlan,
    user_id: int,
    db: AsyncSession,
) -> StudyPlanOut:
    items = sorted(
        plan.items or [],
        key=lambda x: (x.scheduled_date, x.order_index),
    )
    item_outs = [await _item_to_out(it, user_id, db) for it in items]
    return StudyPlanOut(
        id=plan.id,
        user_id=plan.user_id,
        title=plan.title,
        description=plan.description,
        goal=plan.goal,
        start_date=plan.start_date,
        end_date=plan.end_date,
        daily_time_minutes=plan.daily_time_minutes,
        status=plan.status,
        source_path_ids=list(plan.source_path_ids or []),
        items=item_outs,
        created_at=plan.created_at,
        updated_at=plan.updated_at,
    )

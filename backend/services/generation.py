"""
backend/services/generation.py
资源生成服务：封装 LangGraph Agent 调用与结果持久化。
"""

from __future__ import annotations

import json
import re
import traceback
from typing import Any

import asyncio
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from backend.agents.graph import get_graph
from sqlalchemy import select as sa_select

from backend.db.crud import insert_many, select_one, update_by_id
from backend.db.models import GenerationTask, KGNode, LearningPathItem, QuizItem, ResourceMeta
from backend.models.schemas import (
    AgentState,
    GenerateRequest,
    LearningPathCreate,
    LearningPathItemCreate,
    ResourceType,
    TaskStatus,
)


async def run_generation(
    task_id: int,
    user_id: int,
    session_id: int,
    request: dict,
) -> None:
    """
    后台资源生成任务：
    1. 调用 LangGraph Agent Pipeline 生成内容
    2. 将内容持久化到 ResourceMeta
    3. quiz 类型需额外批量写入 quiz_item 表
    4. 更新 GenerationTask 状态
    """
    from backend.db.database import _session_factory
    from backend.services import pathway as pathway_svc

    req = GenerateRequest(**request)
    logger.info("[run_generation] started task_id={} kp_id={} type={}", task_id, req.kp_id, req.resource_type)

    # 为后台任务绑定 trace_id（非 HTTP 上下文，需手动设置）
    from backend.middleware.logging_middleware import generate_trace_id, trace_id_var
    trace_id_var.set(generate_trace_id())

    try:
        async with _session_factory() as db:
            # -- 阶段 1：初始化 AgentState，执行 Agent Pipeline --
            await update_by_id(
                db, GenerationTask, task_id,
                {"status": TaskStatus.running.value, "progress": 10},
            )

            # 解析 kp_id → 知识点名称
            kp_name = req.kp_id
            if req.kp_id.startswith("kp_"):
                node = await select_one(db, KGNode, filters={"id": req.kp_id})
                if node:
                    kp_name = node.name

            initial_state = AgentState(
                user_id=user_id,
                session_id=session_id,
                user_message=f"请生成一份关于 {kp_name} 的 {req.resource_type.value} 学习资源",
                kp_id=req.kp_id,
                resource_type=req.resource_type,
                num_questions=req.num_questions,
                question_type_counts=req.question_type_counts,
            )

            try:
                result = await get_graph().ainvoke(
                    initial_state,
                    config={"configurable": {"db": db}},
                )
                state = AgentState(**result)
            except Exception as e:
                await update_by_id(
                    db, GenerationTask, task_id,
                    {"status": TaskStatus.failed.value, "progress": 0, "error_message": str(e)},
                )
                return

            # -- RAG 评估采集：与 graph.invoke() 中的钩子保持一致 --
            try:
                from backend.agents.graph import (
                    _collect_generation_eval,
                    _maybe_trigger_async_judge,
                )
                _collect_generation_eval(state, session_id)
                _maybe_trigger_async_judge(state, session_id)
            except Exception:
                pass  # 评估采集失败不应影响生成主流程

            # -- 阶段 2：内容持久化 --
            await update_by_id(db, GenerationTask, task_id, {"progress": 80})

            draft = state.draft_content or ""
            resource_type = req.resource_type

            # 检测是否是错误信息（各 agent 失败时写入）
            is_error = (
                draft.startswith("文档生成失败")
                or draft.startswith("思维导图生成失败")
                or draft.startswith("题目生成失败")
                or draft.startswith("代码生成失败")
                or draft.startswith("动画生成失败")
                or draft.startswith("总结生成失败")
                or not draft
            )

            if is_error and not req.resource_type == ResourceType.quiz:
                await update_by_id(
                    db, GenerationTask, task_id,
                    {"status": TaskStatus.failed.value, "progress": 0, "error_message": draft},
                )
                return

            try:
                if resource_type == ResourceType.quiz:
                    await _persist_quiz(task_id, req.kp_id, draft, db)
                else:
                    await _persist_content(task_id, resource_type, draft, db)
            except Exception as e:
                await update_by_id(
                    db, GenerationTask, task_id,
                    {"status": TaskStatus.failed.value, "error_message": str(e)},
                )
                return

            # -- 阶段 3：完成 --
            await update_by_id(
                db, GenerationTask, task_id,
                {"status": TaskStatus.done.value, "progress": 100},
            )

            # -- 兜底：若用户尚无学习路径，自动从推荐创建一条 --
            try:
                recommendations = (state.metadata or {}).get("recommendations", [])
                if recommendations:
                    # 批量验证 KGNode 存在性（单次 IN 查询替代逐条 select_one）
                    kp_ids = [rec.get("kp_id") for rec in recommendations if rec.get("kp_id")]
                    existing_kp_ids: set[str] = set()
                    if kp_ids:
                        result = await db.execute(
                            sa_select(KGNode.id).where(KGNode.id.in_(kp_ids))
                        )
                        existing_kp_ids = {row[0] for row in result.fetchall()}

                    valid_recs = [rec for rec in recommendations if rec.get("kp_id") in existing_kp_ids]

                    if valid_recs:
                        existing = await pathway_svc.list_pathways(int(user_id), db)
                        if not existing:
                            new_path = await pathway_svc.create_pathway(
                                int(user_id),
                                LearningPathCreate(name=f"{kp_name} 学习路径"),
                                db,
                            )
                            if new_path:
                                # 批量插入 LearningPathItem（单次 insert_many 替代逐条 add_pathway_item）
                                items_data = [
                                    {
                                        "path_id": new_path.id,
                                        "kp_id": rec["kp_id"],
                                        "order_index": i,
                                    }
                                    for i, rec in enumerate(valid_recs)
                                ]
                                await insert_many(db, LearningPathItem, data_list=items_data)
            except Exception as e:
                logger.warning("[auto_pathway] failed to auto-create pathway: %s", e)

    except Exception as exc:
        logger.exception("[run_generation] unexpected error: {}", exc)


def _parse_code_block(draft: str) -> tuple[str, str]:
    """从 LLM 返回的 Markdown 中提取代码块和语言标识。"""
    answer_section = draft
    sep_idx = draft.find("参考答案")
    if sep_idx != -1:
        answer_section = draft[sep_idx:]

    blocks = re.findall(r"```(\w*)\s*\n([\s\S]*?)```", answer_section)
    if blocks:
        lang, code = blocks[-1]
        lang = lang.strip() or "python"
        code = code.strip()
        return code, lang

    if sep_idx != -1:
        blocks = re.findall(r"```(\w*)\s*\n([\s\S]*?)```", draft)
        if blocks:
            lang, code = blocks[-1]
            lang = lang.strip() or "python"
            return code.strip(), lang

    return draft.strip(), "python"


def _parse_anim_block(draft: str) -> str:
    """
    从 LLM 返回的内容中提取 p5.js 动画 sketch 源码。
    优先取 ```js/javascript 代码块；找不到 fence 则退回裸文本。
    要求最终至少包含 defineAnimation 调用。
    """
    # 优先匹配显式标注 js/javascript 的代码块
    blocks = re.findall(r"```(?:javascript|js)?\s*\n([\s\S]*?)```", draft, re.IGNORECASE)
    for code in blocks:
        if "defineAnimation" in code:
            return code.strip()
    if blocks:
        return blocks[-1].strip()
    # 无 fence：退回裸文本（可能 LLM 直接输出了 defineAnimation(...)）
    return draft.strip()


async def _persist_content(
    task_id: int,
    resource_type: ResourceType,
    draft: str,
    db: AsyncSession,
) -> None:
    """将非 quiz 类型的生成内容写入 ResourceMeta。"""
    task = await select_one(db, GenerationTask, filters={"id": task_id})
    if not task:
        return
    resource_id = task.resource_id

    if resource_type == ResourceType.mindmap:
        try:
            content_json = json.loads(draft)
        except json.JSONDecodeError:
            match = re.search(r"\{[\s\S]*\}", draft)
            content_json = json.loads(match.group(0)) if match else {"tree": {}}
        await update_by_id(db, ResourceMeta, resource_id, {"content_json": content_json})

    elif resource_type == ResourceType.code:
        code_text, language = _parse_code_block(draft)
        content_json = {"code": code_text, "language": language}
        await update_by_id(
            db, ResourceMeta, resource_id,
            {"content": draft, "content_json": content_json},
        )

    elif resource_type == ResourceType.animation:
        sketch = _parse_anim_block(draft)
        content_json = {"library": "p5", "code": sketch}
        await update_by_id(
            db, ResourceMeta, resource_id,
            {"content": draft, "content_json": content_json},
        )

    else:
        await update_by_id(db, ResourceMeta, resource_id, {"content": draft})


async def _persist_quiz(
    task_id: int,
    kp_id: str,
    draft: str,
    db: AsyncSession,
) -> None:
    """解析 quiz JSON，批量写入 quiz_item 表，并存入 ResourceMeta.content_json。"""
    task = await select_one(db, GenerationTask, filters={"id": task_id})
    if not task:
        return
    resource_id = task.resource_id

    try:
        questions = json.loads(draft)
    except json.JSONDecodeError:
        questions = []

    if questions:
        items_data = [
            {
                "resource_id": resource_id,
                "kp_id": kp_id,
                "question_type": q.get("question_type", "single"),
                "stem": q.get("stem", ""),
                "options": q.get("options"),
                "answer": str(q.get("answer", "")),
                "explanation": q.get("explanation"),
                "order_index": i,
            }
            for i, q in enumerate(questions)
        ]
        instances = await insert_many(db, QuizItem, data_list=items_data)
        # 将数据库生成的雪花 ID 写回 questions，供 content_json 存储
        for q, inst in zip(questions, instances):
            q["id"] = inst.id

    await update_by_id(
        db, ResourceMeta, resource_id,
        {"content_json": {"items": questions}},
    )


async def run_batch_generation(
    batch_id: int,
    user_id: int,
    session_id: int,
    task_configs: list[dict[str, Any]],
) -> None:
    """
    并行执行多个资源生成任务。
    task_configs: [{"task_id": int, "request": dict}, ...]
    """
    from backend.db.database import _session_factory
    from backend.db.models import GenerationBatch

    async def _run_single(cfg: dict) -> None:
        """包装单个 run_generation 调用，捕获异常避免影响其他任务。"""
        try:
            await run_generation(
                task_id=cfg["task_id"],
                user_id=user_id,
                session_id=session_id,
                request=cfg["request"],
            )
        except Exception as e:
            logger.error("[batch] task %s failed: %s", cfg["task_id"], e)

    # 更新 batch 状态为 running
    try:
        async with _session_factory() as db:
            await update_by_id(db, GenerationBatch, batch_id, {
                "status": TaskStatus.running.value,
                "progress": 5,
            })
    except Exception:
        pass

    # 并行执行所有子任务
    await asyncio.gather(*[_run_single(cfg) for cfg in task_configs], return_exceptions=True)

    # 聚合结果，更新 batch 最终状态
    try:
        async with _session_factory() as db:
            from backend.db.crud import select
            tasks = await select(db, GenerationTask, filters={"batch_id": batch_id})
            all_done = all(t.status == TaskStatus.done.value for t in tasks)
            any_failed = any(t.status == TaskStatus.failed.value for t in tasks)

            if all_done:
                final_status = TaskStatus.done.value
            elif any_failed:
                final_status = TaskStatus.failed.value
            else:
                final_status = TaskStatus.done.value

            await update_by_id(db, GenerationBatch, batch_id, {
                "status": final_status,
                "progress": 100 if all_done else 80,
            })
    except Exception as e:
        logger.error("[batch] failed to update batch status: %s", e)

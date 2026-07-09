"""
backend/services/profile.py
学生画像服务：读取、更新、历史版本管理。
"""

from __future__ import annotations

import asyncio
from typing import Optional

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.crud import select_one, select, insert, update_by_id
from backend.db.models import StudentProfile, ProfileHistory
from backend.models.schemas import CognitiveStyle, StudentProfileIn, StudentProfileOut
from backend.config import config

# 保持后台任务引用，避免被 GC。任务完成后自动移除。
_BG_TASKS: set[asyncio.Task] = set()

# 学习目标概括 prompt：对历史提问列表进行总体准确的概括
_GOAL_SUMMARY_PROMPT = """你是一个学习画像分析助手。以下是用户近期在学习对话中的全部提问（按时间顺序）：
{questions}

请你基于这些提问，总结出用户的总体学习目标。要求：
1. 一句话概括，不超过 50 个字；
2. 优先提炼用户的**核心学习领域和阶段目标**，而不是罗列知识点；
3. 概括要体现用户在学什么方向、要达到什么效果，比如“夯实计算机专业基础”“备考xx考试”；
4. 不要逐字拼接提问里的关键词，而是提炼共性、合并同类知识；
5. 避免使用“辅以”这类生硬的书面词，用自然的表述体现主次关系；
6. 概括要体现用户的学习方向和要达成的效果，直接陈述目标，不要用“旨在、辅以”这类词；
7. 在知识点较多时自动归类合并同类知识点，不零散罗列细碎内容，提炼核心学习板块；
8. 优先提炼用户的核心学习方向与阶段目标，聚焦主干能力而非零散知识点；
9. 直接输出目标文本，不要加前缀（如“学习目标：”），不要加引号。
仅输出概括结果。"""



async def _summarize_learning_goal(questions: list[str]) -> Optional[str]:
    """调用 LLM 对历史提问列表进行总体概括，得到精准的学习目标。失败时返回 None。"""
    if not questions:
        return None
    # 延迟导入，避免循环依赖
    from backend.services.llm import chat_completion

    numbered = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(questions))
    prompt = _GOAL_SUMMARY_PROMPT.format(questions=numbered)
    try:
        raw = await chat_completion(
            [{"role": "user", "content": prompt}], temperature=config.agents.profile.goal_summary_temperature
        )
        goal = (raw or "").strip().strip("「」\"'")
        return goal or None
    except Exception as e:
        logger.warning(f"学习目标概括失败: {e}")
        return None


# ----------------------------------------------------------
# 公开接口
# ----------------------------------------------------------

async def get_profile(user_id: int, db: AsyncSession) -> Optional[StudentProfileOut]:
    """
    查询指定用户的当前画像。
    若用户尚未建立画像，返回 None。
    """
    profile = await select_one(
        db, StudentProfile, filters={"user_id": user_id}
    )
    if not profile:
        return None
    return StudentProfileOut.model_validate(profile)


async def create_or_update_profile(
    user_id: int,
    data: StudentProfileIn,
    db: AsyncSession,
) -> StudentProfileOut:
    """
    创建或更新用户画像。
    同时向 profile_history 插入历史快照（版本号自增）。
    """
    existing = await select_one(db, StudentProfile, filters={"user_id": user_id})

    # 序列化当前数据为快照
    if existing:
        snapshot = StudentProfileOut.model_validate(existing).model_dump(mode="json")
        await insert(db, ProfileHistory, {"profile_id": existing.id, "snapshot": snapshot}, commit=False)

        await update_by_id(
            db, StudentProfile, existing.id,
            data={
                "major": data.major,
                "learning_goal": data.learning_goal,
                "cognitive_style": data.cognitive_style.value if data.cognitive_style else None,
                "daily_time_minutes": data.daily_time_minutes,
                "knowledge_mastered": data.knowledge_mastered,
                "knowledge_weak": data.knowledge_weak,
                "error_prone": data.error_prone,
                "current_progress": data.current_progress,
                "goal_questions": data.goal_questions or [],
            }
        )
        await db.refresh(existing)
        return StudentProfileOut.model_validate(existing)
    else:
        # 新建画像
        new_profile = await insert(
            db, StudentProfile,
            data={
                "user_id": user_id,
                "major": data.major,
                "learning_goal": data.learning_goal,
                "cognitive_style": data.cognitive_style.value if data.cognitive_style else None,
                "daily_time_minutes": data.daily_time_minutes,
                "knowledge_mastered": data.knowledge_mastered,
                "knowledge_weak": data.knowledge_weak,
                "error_prone": data.error_prone,
                "current_progress": data.current_progress,
                "goal_questions": data.goal_questions or [],
            }
        )
        # 初始化历史
        snapshot = StudentProfileOut.model_validate(new_profile).model_dump(mode="json")
        await insert(db, ProfileHistory, {"profile_id": new_profile.id, "snapshot": snapshot})
        return StudentProfileOut.model_validate(new_profile)


async def get_profile_history(
    user_id: int,
    db: AsyncSession,
    limit: int = config.agents.profile.history_max_versions,
) -> list[StudentProfileOut]:
    """返回用户的画像历史版本列表（倒序）。"""
    profile = await select_one(db, StudentProfile, filters={"user_id": user_id})
    if not profile:
        return []

    history_records = await select(
        db, ProfileHistory,
        filters={"profile_id": profile.id},
        order_by=ProfileHistory.created_at.desc(),
        limit=limit,
    )
    return [
        StudentProfileOut(**record.snapshot)
        for record in history_records
    ]


async def merge_chat_updates(
    user_id: int,
    updates: dict,
    db: AsyncSession,
    user_message: Optional[str] = None,
) -> StudentProfileOut:
    """
    将 ProfileAgent 从对话中提取的画像字段增量合并进当前画像。
    只更新 updates 中非 None 的字段。

    学习目标（learning_goal）采用增量更新策略：
    在本函数里仅将本轮 user_message 追加到 goal_questions 历史列表（纯DB写，不阻塞）；
    真正的 LLM 概括由调用方在回复返回后，通过 BackgroundTasks 异步调用 refresh_learning_goal 完成，
    以避免对话接口因多次 LLM 调用叠加而超时。
    """
    # 防御：LLM 或上游可能传入 None，统一归一为空 dict
    if not isinstance(updates, dict):
        updates = {}
    existing = await select_one(db, StudentProfile, filters={"user_id": user_id})

    # 准备：追加本轮提问到历史列表（仅记录，不在此处调 LLM）
    existing_questions: list[str] = []
    if existing is not None:
        existing_questions = list(getattr(existing, "goal_questions", None) or [])

    new_questions = list(existing_questions)
    if user_message and user_message.strip():
        new_questions.append(user_message.strip())
        # 只保留最近 N 条，避免无限增长
        if len(new_questions) > config.agents.profile.max_goal_questions:
            new_questions = new_questions[-config.agents.profile.max_goal_questions:]

    if not existing:
        # 不存在则创建新画像；学习目标暂用 LLM 单轮提取结果占位，后台任务会重新概括写回
        # cognitive_style: LLM 可能返回 enum name，需转为 CognitiveStyle 枚举
        raw_style = updates.get("cognitive_style")
        cs_enum = None
        if raw_style:
            try:
                cs_enum = CognitiveStyle(raw_style)
            except ValueError:
                try:
                    cs_enum = CognitiveStyle[raw_style]
                except KeyError:
                    cs_enum = None
        created = await create_or_update_profile(
            user_id,
            StudentProfileIn(
                major=updates.get("major"),
                learning_goal=updates.get("learning_goal"),
                cognitive_style=cs_enum,
                daily_time_minutes=updates.get("daily_time_minutes"),
                knowledge_mastered=updates.get("knowledge_mastered") or [],
                knowledge_weak=updates.get("knowledge_weak") or [],
                error_prone=updates.get("error_prone") or [],
                current_progress=updates.get("current_progress"),
                goal_questions=new_questions,
            ),
            db,
        )
        # fire-and-forget：新建画像有提问时也异步触发 LLM 概括
        if new_questions:
            _schedule_refresh_learning_goal(user_id)
        return created

    # 快照当前状态
    snapshot = StudentProfileOut.model_validate(existing).model_dump(mode="json")
    await insert(db, ProfileHistory, {"profile_id": existing.id, "snapshot": snapshot}, commit=False)

    # 只更新非 None 的字段（learning_goal 由后台任务异步覆写）
    update_data = {}
    for key in ["major", "cognitive_style", "daily_time_minutes",
                "current_progress"]:
        if key in updates and updates[key] is not None:
            val = updates[key]
            # cognitive_style: LLM 返回 enum name (visual/text/practice)，需转为中文 value
            if key == "cognitive_style" and isinstance(val, str):
                try:
                    val = CognitiveStyle(val).value
                except ValueError:
                    try:
                        val = CognitiveStyle[val].value
                    except KeyError:
                        val = None
            if val is not None:
                update_data[key] = val

    # 已掌握知识点 / 薄弱知识点 / 易错点：改为增量叠加，去重保序，不做 LLM 概括
    for list_key in ("knowledge_mastered", "knowledge_weak", "error_prone"):
        incoming = updates.get(list_key)
        if not incoming:
            continue
        # 兼容单字符串的极端情况
        if isinstance(incoming, str):
            incoming = [incoming]
        existing_list = list(getattr(existing, list_key, None) or [])
        seen = {item.strip() for item in existing_list if isinstance(item, str) and item.strip()}
        merged = list(existing_list)
        for item in incoming:
            if not isinstance(item, str):
                continue
            key = item.strip()
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(key)
        if merged != existing_list:
            update_data[list_key] = merged

    # 增量记录提问
    if new_questions != existing_questions:
        update_data["goal_questions"] = new_questions

    if update_data:
        await update_by_id(db, StudentProfile, existing.id, update_data)
        await db.refresh(existing)

    # fire-and-forget：如果 goal_questions 有新增，异步触发 LLM 重新概括 learning_goal。
    # 不使用 FastAPI BackgroundTasks，避免 /chat 接口因资源生成耗时超时时后台任务不执行。
    if new_questions != existing_questions:
        _schedule_refresh_learning_goal(user_id, new_questions)

    return StudentProfileOut.model_validate(existing)


def _schedule_refresh_learning_goal(user_id: int, questions: Optional[list[str]] = None) -> None:
    """在当前 event loop 中投递一个后台协程，用独立 session 刷新 learning_goal。失败静默忽略。

    若传入 questions，则直接用它做 LLM 概括，避免因调用方 session 尚未 commit
    而导致后台任务读到旧快照的事务可见性问题。"""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        logger.warning("_schedule_refresh_learning_goal: 当前无 running loop，跳过")
        return
    task = loop.create_task(refresh_learning_goal(user_id, questions))
    _BG_TASKS.add(task)
    task.add_done_callback(_BG_TASKS.discard)


async def refresh_learning_goal(
    user_id: int,
    questions: Optional[list[str]] = None,
) -> None:
    """
    后台任务：对历史提问做总体概括，将结果写回 learning_goal。

    :param questions: 可选。若由调用方传入则直接使用（绕过事务可见性问题）；否则从 DB 重读。
    自行创建独立的 DB session，不与请求生命周期耦合。失败时仅记录 warning。
    """
    try:
        from backend.db import database as db_mod
    except Exception as e:
        logger.warning(f"refresh_learning_goal 导入数据库模块失败: {e}")
        return

    factory = getattr(db_mod, "_session_factory", None)
    if factory is None:
        logger.warning("refresh_learning_goal: session factory 未初始化，跳过")
        return

    try:
        async with factory() as db:
            try:
                # 在获取 profile 之前先延迟一下，给调用方 session 留出提交窗口，
                # 保证我们能读到最新的 goal_questions（在没传 questions 时生效）。
                if questions is None:
                    await asyncio.sleep(0.2)
                existing = await select_one(db, StudentProfile, filters={"user_id": user_id})
                if not existing:
                    return
                final_questions = questions if questions is not None else list(
                    getattr(existing, "goal_questions", None) or []
                )
                if not final_questions:
                    return
                summarized = await _summarize_learning_goal(final_questions)
                if summarized:
                    await update_by_id(db, StudentProfile, existing.id, {"learning_goal": summarized})
                    await db.commit()
                    logger.info(f"[refresh_learning_goal] 学习目标已刷新: {summarized[:60]}")
            except Exception as e:
                logger.warning(f"后台学习目标刷新失败: {e}")
                await db.rollback()
    except Exception as e:
        logger.warning(f"后台学习目标刷新 session 创建失败: {e}")


async def build_profile_context(profile: StudentProfileOut) -> str:
    """
    将画像对象序列化为 prompt 上下文字符串，注入到 Agent System Prompt 中。
    例如：'学生专业：计算机，目标：掌握深度学习基础，薄弱点：反向传播...'
    """
    parts = []

    if profile.major:
        parts.append(f"学生专业：{profile.major}")
    if profile.learning_goal:
        parts.append(f"学习目标：{profile.learning_goal}")
    if profile.cognitive_style:
        parts.append(f"认知风格：{profile.cognitive_style.value}")
    if profile.daily_time_minutes:
        parts.append(f"每日学习时间：{profile.daily_time_minutes}分钟")
    if profile.knowledge_mastered:
        parts.append(f"已掌握的知识点：{', '.join(profile.knowledge_mastered)}")
    if profile.knowledge_weak:
        parts.append(f"薄弱知识点：{', '.join(profile.knowledge_weak)}")
    if profile.error_prone:
        parts.append(f"容易出错的知识点：{', '.join(profile.error_prone)}")
    if profile.current_progress:
        parts.append(f"当前进度：{profile.current_progress}")

    if not parts:
        return "暂无学生画像信息"
    return "，".join(parts)


# ----------------------------------------------------------
# 测验驱动画像更新
# ----------------------------------------------------------

# 掌握度阈值
_MASTERY_THRESHOLD = 0.8   # 正确率 >= 80% 视为已掌握
_WEAK_THRESHOLD = 0.6      # 正确率 < 60% 视为薄弱
_MIN_ATTEMPTS = 2          # 至少做过 2 题才纳入统计


async def update_profile_from_quiz(
    user_id: int,
    kp_id: str,
    db: AsyncSession,
) -> None:
    """
    测验提交后根据该知识点的历史正确率自动更新学生画像。

    规则：
    - 正确率 >= 80% 且做题数 >= 2：加入 knowledge_mastered，从 knowledge_weak 移除
    - 正确率 < 60%：加入 knowledge_weak，从 knowledge_mastered 移除
    - 答错时：加入 error_prone（去重）
    """
    import sqlalchemy as sa
    from backend.db.models import QuizAttempt, KGNode

    # 1. 查询该用户在该知识点的所有答题记录
    result = await db.execute(
        sa.select(
            sa.func.count().label("total"),
            sa.func.sum(sa.case((QuizAttempt.is_correct == True, 1), else_=0)).label("correct"),
        ).where(
            QuizAttempt.user_id == user_id,
            QuizAttempt.kp_id == kp_id,
        )
    )
    row = result.one_or_none()
    if not row or not row.total or row.total < _MIN_ATTEMPTS:
        return

    total = row.total
    correct = row.correct or 0
    accuracy = correct / total

    # 2. 解析知识点名称
    kp_node = await select_one(db, KGNode, filters={"id": kp_id})
    kp_name = kp_node.name if kp_node else kp_id

    # 3. 获取当前画像
    profile = await select_one(db, StudentProfile, filters={"user_id": user_id})
    if not profile:
        return

    mastered = list(profile.knowledge_mastered or [])
    weak = list(profile.knowledge_weak or [])
    error_prone = list(profile.error_prone or [])
    changed = False

    # 4. 根据正确率更新画像
    if accuracy >= _MASTERY_THRESHOLD:
        # 加入已掌握
        if kp_name not in mastered:
            mastered.append(kp_name)
            changed = True
        # 从薄弱移除
        if kp_name in weak:
            weak.remove(kp_name)
            changed = True
    elif accuracy < _WEAK_THRESHOLD:
        # 加入薄弱
        if kp_name not in weak:
            weak.append(kp_name)
            changed = True
        # 从已掌握移除
        if kp_name in mastered:
            mastered.remove(kp_name)
            changed = True

    # 5. 最近一次答错 → 加入易错点
    last_attempt = await db.execute(
        sa.select(QuizAttempt).where(
            QuizAttempt.user_id == user_id,
            QuizAttempt.kp_id == kp_id,
        ).order_by(QuizAttempt.created_at.desc()).limit(1)
    )
    last = last_attempt.scalar_one_or_none()
    if last and not last.is_correct:
        if kp_name not in error_prone:
            error_prone.append(kp_name)
            changed = True

    # 6. 持久化
    if changed:
        update_data = {
            "knowledge_mastered": mastered,
            "knowledge_weak": weak,
            "error_prone": error_prone,
        }
        await update_by_id(db, StudentProfile, profile.id, update_data)
        logger.info(
            f"[ProfileQuiz] user={user_id} kp={kp_name} "
            f"accuracy={accuracy:.0%}({correct}/{total}) "
            f"mastered={len(mastered)} weak={len(weak)} error={len(error_prone)}"
        )

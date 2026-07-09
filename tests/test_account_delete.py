"""
tests/test_account_delete.py
账号注销（硬删除）级联清理测试。

需要可连接的 PostgreSQL（本地 Docker pgvector）。若数据库不可用则整体 skip，
不影响纯单元测试套件。asyncio_mode = auto，无需 @pytest.mark.asyncio。
"""

from __future__ import annotations

from datetime import date

import pytest

from backend.auth.hash_utils import hash_password
from backend.db.crud import insert, select, select_one, delete_user_cascade
from backend.db.database import init_db, get_session_ctx, health_check
from backend.db.models import (
    ChatSession,
    LearningPath,
    LearningRecord,
    QuizAttempt,
    QuizItem,
    ResourceMeta,
    StudentProfile,
    StudyPlan,
    User,
)

_TEST_USERNAME = "__pytest_del_user__"


async def _db_available() -> bool:
    try:
        await init_db()
        return await health_check()
    except Exception:
        return False


async def _count(db, model, uid) -> int:
    return len(await select(db, model, filters={"user_id": uid}))


async def _cleanup_leftover():
    async with get_session_ctx() as db:
        u = await select_one(db, User, filters={"username": _TEST_USERNAME})
        if u:
            await delete_user_cascade(db, u.id)


@pytest.fixture
async def seeded_user():
    """造一个带各类关联数据的临时用户，返回 user_id；测试后清理。"""
    if not await _db_available():
        pytest.skip("数据库不可用，跳过账号注销集成测试")

    await _cleanup_leftover()

    async with get_session_ctx() as db:
        user = await insert(db, User, {
            "username": _TEST_USERNAME,
            "hashed_password": hash_password("pw123456"),
            "email": "__pytest_del_user__@example.com",
        })
        uid = user.id
        await insert(db, StudentProfile, {"user_id": uid, "major": "CS"})
        await insert(db, ChatSession, {"user_id": uid, "title": "t"})
        res = await insert(db, ResourceMeta, {
            "user_id": uid, "resource_type": "doc", "kp_id": "kp_x",
            "title": "r", "content": "c",
        })
        qi = await insert(db, QuizItem, {
            "resource_id": res.id, "stem": "q?", "question_type": "single",
            "options": ["a", "b"], "answer": "a",
        })
        await insert(db, QuizAttempt, {
            "quiz_item_id": qi.id, "user_id": uid, "user_answer": "a", "is_correct": True,
        })
        await insert(db, LearningRecord, {"user_id": uid, "action": "view"})
        await insert(db, LearningPath, {"user_id": uid, "title": "p"})
        await insert(db, StudyPlan, {
            "user_id": uid, "title": "sp", "status": "active",
            "start_date": date(2026, 6, 15), "end_date": date(2026, 6, 20),
        })

    yield uid

    await _cleanup_leftover()


class TestDeleteUserCascade:
    async def test_cascade_removes_user_and_all_associations(self, seeded_user):
        uid = seeded_user

        async with get_session_ctx() as db:
            ok = await delete_user_cascade(db, uid)
        assert ok is True

        async with get_session_ctx() as db:
            assert await select_one(db, User, filters={"id": uid}) is None
            for model in (
                StudentProfile, ChatSession, ResourceMeta,
                QuizAttempt, LearningRecord, LearningPath, StudyPlan,
            ):
                assert await _count(db, model, uid) == 0, f"{model.__name__} 未清空"

    async def test_delete_nonexistent_user_returns_false(self):
        if not await _db_available():
            pytest.skip("数据库不可用，跳过账号注销集成测试")
        async with get_session_ctx() as db:
            ok = await delete_user_cascade(db, 999_999_999_999_999)
        assert ok is False

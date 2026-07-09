"""
backend/main.py
FastAPI 应用入口：路由注册、生命周期管理、中间件配置。
"""

from __future__ import annotations

import asyncio
import json
from backend.utils.snowflake import generate_id, string_to_id
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Optional, Annotated

import jwt
from fastapi import BackgroundTasks, Body, Depends, FastAPI, HTTPException, status, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

# -------------------------------------------------------
# 自定义 JSON 编码器：将超过 JS 安全整数范围的 int 序列化为字符串
# JavaScript Number 只能精确表示 ±2^53-1，Snowflake ID（64-bit）会丢失精度
# -------------------------------------------------------
_JS_MAX_SAFE_INTEGER = 2**53 - 1  # 9007199254740991


class _SafeIntEncoder(json.JSONEncoder):
    """JSON 编码器：大于 2^53-1 的整数自动转为字符串。"""

    def encode(self, o):
        return super().encode(self._convert(o))

    def _convert(self, obj):
        if isinstance(obj, dict):
            return {k: self._convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._convert(item) for item in obj]
        if isinstance(obj, int) and abs(obj) > _JS_MAX_SAFE_INTEGER:
            return str(obj)
        return obj


class BigIntJSONResponse(JSONResponse):
    """使用 SafeIntEncoder 的自定义 JSON 响应。"""

    def render(self, content) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            indent=None,
            separators=(",", ":"),
            cls=_SafeIntEncoder,
        ).encode("utf-8")

from backend.auth.hash_utils import hash_password, verify_password
from backend.auth.deps import get_current_user_id
from backend.agents.graph import get_graph, invoke, stream_invoke
from backend.middleware.logging_middleware import LoggingMiddleware
from backend.db.database import close_db, get_session, health_check as db_health, init_db
import backend.db.database as _db_module
from backend.db.vector import health_check as vec_health_check, init_vector_db
from backend.models.schemas import (
    BatchGenerateOut,
    BatchGenerateRequest,
    ChatMessageIn,
    ChatMessageOut,
    ChatSessionOut,
    EmailVerificationOut,
    ForgotPasswordIn,
    GenerateRequest,
    GenerateTaskOut,
    KGGraphOut,
    KGEdgeOut,
    KGNodeOut,
    LearningPathCreate,
    LearningPathItemCreate,
    LearningPathItemOut,
    LearningPathItemUpdate,
    LearningPathOut,
    LearningPathUpdate,
    LearningRecordCreate,
    LearningRecordOut,
    QuizAttemptOut,
    QuizItemOut,
    QuizSubmitIn,
    QuestionType,
    ResetPasswordIn,
    ResourceListOut,
    ResourceMetaOut,
    ResourceType,
    StudentProfileIn,
    StudentProfileOut,
    TokenOut,
    UserUpdate,
    AccountDeleteIn,
    UserCreate,
    UserOut,
    KGNodeType,
    KGRelation,
    StudyPlanGenerateRequest,
    StudyPlanItemOut,
    StudyPlanItemUpdate,
    StudyPlanOut,
    StudyPlanUpdate,
)
from backend.db.models import User, ChatSession, ChatMessage, KGNode, KGEdge, QuizItem, QuizAttempt, LearningPath, LearningPathItem, ResourceMeta, LearningRecord, EmailVerification
from backend.services import profile as profile_svc
from backend.services import resource as resource_svc
from backend.services import document as document_svc
from backend.email.sender import email_sender
from backend.email.utils import generate_token, hash_token, expires_at
from backend.db.models import User, ChatSession, ChatMessage, KGNode, KGEdge, QuizItem, QuizAttempt, LearningPath, LearningPathItem, ResourceMeta, LearningRecord

# 内存任务字典：{task_id: {status, progress, stage, doc_id, error, result}}
_doc_import_tasks: dict[str, dict] = {}

# ===========================================================
# JWT 配置（从 configs/config.yaml 读取）
# ===========================================================

from backend.config import config as app_config
from backend.logging_config import logger  # noqa: F401

JWT_SECRET = app_config.jwt.secret
JWT_ALGORITHM = app_config.jwt.algorithm
JWT_EXPIRE_HOURS = app_config.jwt.expire_hours

# ===========================================================
# Lifespan（应用启动 / 关闭）
# ===========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """初始化数据库连接池与向量库，启动后台清理任务，关闭时释放资源。"""
    await init_db()
    init_vector_db()
    get_graph()  # 预热 LangGraph

    # 启动聊天会话过期清理后台任务
    from backend.services.chat_cleanup import start_cleanup_task as start_chat_cleanup_task
    cleanup_task = asyncio.create_task(start_chat_cleanup_task())

    # 启动文档文件清理后台任务
    from backend.services.cleanup import start_cleanup_task as start_doc_cleanup_task
    doc_cleanup_task = asyncio.create_task(start_doc_cleanup_task())

    # 知识库为空时，自动从配置的知识库目录索引
    from backend.db.vector import get_collection
    from backend.rag.indexer import index_directory
    import os
    KB_DIR = app_config.storage.knowledge_base_dir
    col = get_collection()
    doc_count = await col.count()
    if doc_count == 0 and os.path.isdir(KB_DIR):
        logger.info("[Lifespan] 知识库为空，开始自动索引...")
        indexed = await index_directory(KB_DIR)
        logger.info(f"[Lifespan] 知识库索引完成，共写入 {indexed} 个文本块。")

    try:
        yield
    finally:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except (asyncio.CancelledError, Exception):
            pass
        doc_cleanup_task.cancel()
        try:
            await doc_cleanup_task
        except (asyncio.CancelledError, Exception):
            pass
        await close_db()


# ===========================================================
# 应用实例
# ===========================================================

app = FastAPI(
    title="A3 个性化学习多智能体系统",
    version=app_config.server.version,
    lifespan=lifespan,
    default_response_class=BigIntJSONResponse,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=app_config.server.cors_origins,   # 生产环境应限制为 Streamlit 域名
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(LoggingMiddleware)

# 对 HTML 和 JS 文件禁用浏览器缓存，确保代码更新后立即生效
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest

class NoCacheHTMLMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: StarletteRequest, call_next):
        response = await call_next(request)
        if request.url.path.startswith('/app/'):
            if request.url.path.endswith(('.html', '.js')) or '/app/' == request.url.path[-1:]:
                response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
                response.headers['Pragma'] = 'no-cache'
                response.headers['Expires'] = '0'
        return response

app.add_middleware(NoCacheHTMLMiddleware)

# 静态文件服务：挂载 frontend 目录到 /app 路径
from pathlib import Path
html_dir = Path(__file__).parent.parent / "frontend"
if html_dir.exists():
    from fastapi.staticfiles import StaticFiles
    app.mount("/app", StaticFiles(directory=str(html_dir), html=True), name="html")


# ===========================================================
# 健康检查
# ===========================================================

@app.get("/health", tags=["system"])
async def health():
    """系统健康检查接口，返回各组件状态。"""
    return {
        "status": "ok",
        "db": await db_health(),
        "vector_db": await vec_health_check(),
    }


# ===========================================================
# 用户认证
# ===========================================================

# 内存限流：{user_id: [timestamp, ...]}
_rate_limit_store: dict[int, list[datetime]] = {}


def _check_rate_limit(user_id: int) -> bool:
    """检查用户是否超出每小时发送限制。返回 True 表示允许发送。"""
    now = datetime.utcnow()
    limit = app_config.email.rate_limit_send_per_hour
    if limit <= 0:
        return True
    timestamps = _rate_limit_store.get(user_id, [])
    timestamps = [t for t in timestamps if (now - t).total_seconds() < 3600]
    _rate_limit_store[user_id] = timestamps
    return len(timestamps) < limit


def _record_rate_limit(user_id: int):
    timestamps = _rate_limit_store.get(user_id, [])
    timestamps.append(datetime.utcnow())
    _rate_limit_store[user_id] = timestamps


async def _send_verification_email(user_id: int, email: str):
    """后台任务：发送邮箱验证邮件。使用独立 DB 会话。"""
    try:
        from backend.db.database import _session_factory
        from backend.db.crud import insert as bg_insert, select_one as bg_select_one
        if _session_factory is None:
            logger.error("[Email] _session_factory is None, cannot send verification email")
            return
        token = generate_token()
        token_hash_val = hash_token(token)
        async with _session_factory() as bg_db:
            await bg_insert(bg_db, EmailVerification, data={
                "user_id": user_id,
                "token_hash": token_hash_val,
                "purpose": "email_verify",
                "expires_at": expires_at("email_verify"),
            })
            await bg_db.commit()
        user_for_email = await bg_select_one(bg_db, User, filters={"id": user_id})
        username_for_email = user_for_email.username if user_for_email else email
        await email_sender.send_verification(email, username_for_email, token)
        logger.info("[Email] 验证邮件已发送至 {}", email)
    except Exception as e:
        logger.exception("[Email] 发送验证邮件失败: {}", e)


@app.post("/auth/register", response_model=UserOut, tags=["auth"])
async def register(body: UserCreate, db: AsyncSession = Depends(get_session)):
    """注册新用户。"""
    from backend.db.crud import select_one, insert

    existing = await select_one(db, User, filters={"username": body.username})
    if existing:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username already exists")

    if body.email:
        existing_email = await select_one(db, User, filters={"email": body.email})
        if existing_email:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")

    user_data = {"username": body.username, "hashed_password": hash_password(body.password)}
    if body.email:
        user_data["email"] = body.email

    user = await insert(db, User, data=user_data)

    if body.email:
        import asyncio as _asyncio
        _asyncio.create_task(_send_verification_email(user.id, body.email))

    return UserOut.model_validate(user)


@app.post("/auth/login", response_model=TokenOut, tags=["auth"])
async def login(body: UserCreate, db: AsyncSession = Depends(get_session)):
    """用户名密码登录，返回 JWT Token。"""
    from backend.db.crud import select_one

    user = await select_one(db, User, filters={"username": body.username})
    if not user or not verify_password(body.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid username or password")

    expire = datetime.utcnow() + timedelta(hours=JWT_EXPIRE_HOURS)
    payload = {"sub": str(user.id), "exp": expire}
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return TokenOut(user_id=user.id, access_token=token, token_type="bearer")


@app.post("/auth/send-verification", response_model=EmailVerificationOut, tags=["auth"])
async def send_verification(
    body: dict = Body(...),
    db: AsyncSession = Depends(get_session),
):
    """重新发送邮箱验证邮件。"""
    from backend.db.crud import select_one

    user_id = body.get("user_id")
    email = body.get("email")
    if not user_id or not email:
        raise HTTPException(status_code=400, detail="user_id and email are required")

    user = await select_one(db, User, filters={"id": int(user_id)})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if user.email_verified:
        return EmailVerificationOut(message="邮箱已验证，无需重复验证")

    if not _check_rate_limit(user.id):
        raise HTTPException(status_code=429, detail="发送过于频繁，请稍后再试")

    _record_rate_limit(user.id)

    import asyncio as _asyncio
    _asyncio.create_task(_send_verification_email(user.id, email))

    return EmailVerificationOut(message="验证邮件已发送，请查收邮箱")


@app.get("/auth/verify-email", tags=["auth"])
async def verify_email(
    token: str,
    db: AsyncSession = Depends(get_session),
):
    """验证邮箱 Token，重定向到前端验证结果页面。"""
    from backend.db.crud import select_one, update_by_id
    from fastapi.responses import RedirectResponse

    token_hash_val = hash_token(token)
    record = await select_one(db, EmailVerification, filters={
        "token_hash": token_hash_val,
        "purpose": "email_verify",
    })

    base_url = "/app/verify-email.html"

    if not record:
        return RedirectResponse(f"{base_url}?status=error&reason=invalid_token")
    if record.used:
        return RedirectResponse(f"{base_url}?status=already_verified")
    if record.expires_at < datetime.utcnow():
        return RedirectResponse(f"{base_url}?status=error&reason=expired")

    await update_by_id(db, EmailVerification, record.id, data={"used": True})
    await update_by_id(db, User, record.user_id, data={
        "email_verified": True,
        "email_verified_at": datetime.utcnow(),
    })

    return RedirectResponse(f"{base_url}?status=success", status_code=302)


@app.post("/auth/forgot-password", response_model=EmailVerificationOut, tags=["auth"])
async def forgot_password(
    body: ForgotPasswordIn,
    db: AsyncSession = Depends(get_session),
):
    """忘记密码：发送密码重置邮件。反枚举保护 — 无论邮箱是否存在都返回相同信息。"""
    from backend.db.crud import select_one, insert as crud_insert

    user = await select_one(db, User, filters={"email": body.email})
    if user:
        if _check_rate_limit(user.id):
            _record_rate_limit(user.id)
            token = generate_token()
            token_hash_val = hash_token(token)
            await crud_insert(db, EmailVerification, data={
                "user_id": user.id,
                "token_hash": token_hash_val,
                "purpose": "password_reset",
                "expires_at": expires_at("password_reset"),
            })
            import asyncio as _asyncio
            _asyncio.create_task(email_sender.send_password_reset(body.email, user.username, token))

    return EmailVerificationOut(message="如果该邮箱已注册，重置邮件已发送，请查收")


@app.post("/auth/reset-password", response_model=EmailVerificationOut, tags=["auth"])
async def reset_password(
    body: ResetPasswordIn,
    db: AsyncSession = Depends(get_session),
):
    """使用 Token 重置密码。"""
    from backend.db.crud import select_one, update_by_id

    token_hash_val = hash_token(body.token)
    record = await select_one(db, EmailVerification, filters={
        "token_hash": token_hash_val,
        "purpose": "password_reset",
    })

    if not record:
        raise HTTPException(status_code=400, detail="无效的重置链接")
    if record.used:
        raise HTTPException(status_code=400, detail="该链接已被使用")
    if record.expires_at < datetime.utcnow():
        raise HTTPException(status_code=400, detail="该链接已过期，请重新申请")

    await update_by_id(db, EmailVerification, record.id, data={"used": True})
    await update_by_id(db, User, record.user_id, data={
        "hashed_password": hash_password(body.new_password),
    })

    return EmailVerificationOut(message="密码重置成功，请使用新密码登录")


@app.post("/email/learning-report", tags=["email"])
async def send_learning_report(
    user_id: int,
    db: AsyncSession = Depends(get_session),
):
    """发送学习报告邮件。"""
    import sqlalchemy as sa

    user = await db.execute(sa.select(User).where(User.id == user_id))
    user = user.scalar_one_or_none()
    if not user or not user.email:
        raise HTTPException(status_code=400, detail="未设置邮箱")

    # 统计数据
    resource_count = (await db.execute(
        sa.select(sa.func.count(ResourceMeta.id)).where(ResourceMeta.user_id == user_id)
    )).scalar() or 0

    quiz_attempts = await db.execute(
        sa.select(
            sa.func.count(QuizAttempt.id),
            sa.func.sum(sa.case((QuizAttempt.is_correct == True, 1), else_=0)),
        ).where(QuizAttempt.user_id == user_id)
    )
    quiz_total, quiz_correct = quiz_attempts.one()
    mastery_pct = round(quiz_correct / quiz_total * 100) if quiz_total else 0

    pathway_count = (await db.execute(
        sa.select(sa.func.count(LearningPath.id)).where(LearningPath.user_id == user_id)
    )).scalar() or 0

    stats = {
        "resource_count": resource_count,
        "quiz_total": quiz_total or 0,
        "quiz_correct": quiz_correct or 0,
        "mastery_pct": mastery_pct,
        "pathway_count": pathway_count,
    }

    import asyncio as _asyncio
    _asyncio.create_task(email_sender.send_learning_report(user.email, user.username, stats))
    return {"message": "学习报告邮件已发送"}


@app.post("/study-plan/email", tags=["study-plan"])
async def send_study_plan_email(
    plan_id: int,
    user_id: int,
    db: AsyncSession = Depends(get_session),
):
    """将学习计划表发送到用户邮箱。"""
    import asyncio as _asyncio
    import sqlalchemy as sa
    import backend.services.study_plan as sp

    # 获取用户信息
    user = (await db.execute(sa.select(User).where(User.id == user_id))).scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
    if not user.email:
        raise HTTPException(status_code=400, detail="未设置邮箱地址，请先在个人中心绑定邮箱")

    # 获取学习计划
    plan = await sp.get_study_plan(plan_id, user_id, db)
    if not plan:
        raise HTTPException(status_code=404, detail="学习计划不存在")

    # 构建邮件数据：按日期分组 items
    from collections import OrderedDict
    items_by_date = OrderedDict()
    for item in plan.items:
        date_str = item.scheduled_date.isoformat() if item.scheduled_date else "未排期"
        # 格式化日期为中文
        d = item.scheduled_date
        weekdays = ["一", "二", "三", "四", "五", "六", "日"]
        label = f"{d.month}月{d.day}日 周{weekdays[d.weekday()]}"
        if label not in items_by_date:
            items_by_date[label] = []
        resources_info = [
            {"id": r.resource_id, "type": r.resource_type.value if hasattr(r.resource_type, 'value') else str(r.resource_type), "title": r.title}
            for r in (item.resources or [])
        ]
        items_by_date[label].append({
            "kp_name": item.kp_name,
            "start_time": item.start_time,
            "end_time": item.end_time,
            "estimated_minutes": item.estimated_minutes or "-",
            "order_index": item.order_index,
            "is_completed": item.is_completed,
            "notes": item.notes or "",
            "resources": resources_info,
            "missing_resource_types": [str(t) if not isinstance(t, str) else t for t in (item.missing_resource_types or [])],
        })

    plan_data = {
        "title": plan.title,
        "start_date": plan.start_date.isoformat() if plan.start_date else "",
        "end_date": plan.end_date.isoformat() if plan.end_date else "",
        "daily_time_minutes": plan.daily_time_minutes or "-",
        "goal": plan.goal or "",
        "items_by_date": items_by_date,
    }

    _asyncio.create_task(email_sender.send_study_plan(user.email, user.username, plan_data))
    return {
        "message": "学习计划表邮件已发送",
        "email": user.email,
        "plan_title": plan.title,
    }


@app.get("/study-plans", response_model=list[StudyPlanOut], tags=["study-plan"])
async def list_study_plans(
    user_id: int,
    db: AsyncSession = Depends(get_session),
):
    """列出用户的所有学习计划（按创建时间倒序）。"""
    import backend.services.study_plan as sp
    return await sp.list_study_plans(user_id, db)


@app.get("/study-plans/{plan_id}", response_model=StudyPlanOut, tags=["study-plan"])
async def get_study_plan(
    plan_id: int,
    user_id: int,
    db: AsyncSession = Depends(get_session),
):
    """获取单个学习计划详情。"""
    import backend.services.study_plan as sp
    plan = await sp.get_study_plan(plan_id, user_id, db)
    if not plan:
        raise HTTPException(status_code=404, detail="学习计划不存在")
    return plan


@app.post("/study-plans/generate", response_model=StudyPlanOut, tags=["study-plan"])
async def generate_study_plan(
    user_id: int,
    body: StudyPlanGenerateRequest,
    db: AsyncSession = Depends(get_session),
):
    """生成一份新的学习计划。"""
    import backend.services.study_plan as sp
    plan = await sp.generate_study_plan(user_id, body, db)
    if not plan:
        raise HTTPException(status_code=400, detail="没有可用的候选知识点，请先建立学习路径或完善学生画像")
    return plan


@app.put("/study-plans/{plan_id}", response_model=StudyPlanOut, tags=["study-plan"])
async def update_study_plan(
    plan_id: int,
    user_id: int,
    body: StudyPlanUpdate,
    db: AsyncSession = Depends(get_session),
):
    """更新学习计划（标题/描述/状态）。"""
    import backend.services.study_plan as sp
    plan = await sp.update_study_plan(plan_id, user_id, body, db)
    if not plan:
        raise HTTPException(status_code=404, detail="学习计划不存在")
    return plan


@app.delete("/study-plans/{plan_id}", tags=["study-plan"])
async def delete_study_plan(
    plan_id: int,
    user_id: int,
    db: AsyncSession = Depends(get_session),
):
    """删除学习计划（级联删除所有计划项）。"""
    import backend.services.study_plan as sp
    ok = await sp.delete_study_plan(plan_id, user_id, db)
    if not ok:
        raise HTTPException(status_code=404, detail="学习计划不存在")
    return {"ok": True}


@app.put("/study-plans/{plan_id}/items/{item_id}", response_model=StudyPlanItemOut, tags=["study-plan"])
async def update_study_plan_item(
    plan_id: int,
    item_id: int,
    user_id: int,
    body: StudyPlanItemUpdate,
    db: AsyncSession = Depends(get_session),
):
    """更新单个计划项（日期/时段/完成状态/备注）。"""
    import backend.services.study_plan as sp
    item = await sp.update_study_plan_item(plan_id, item_id, user_id, body, db)
    if not item:
        raise HTTPException(status_code=404, detail="计划项不存在")
    return item


@app.post("/study-plans/{plan_id}/items/{item_id}/generate-resource", tags=["study-plan"])
async def generate_study_plan_item_resources(
    plan_id: int,
    item_id: int,
    user_id: int,
    body: dict,
    db: AsyncSession = Depends(get_session),
):
    """为计划项生成缺失资源（后台异步）。"""
    import asyncio as _asyncio
    import backend.services.study_plan as sp

    item = await sp.get_study_plan_item(plan_id, item_id, user_id, db)
    if not item:
        raise HTTPException(status_code=404, detail="计划项不存在")

    resource_types = body.get("resource_types") if body else None
    types = resource_types or item.missing_resource_types
    if not types:
        raise HTTPException(status_code=400, detail="没有需要生成的资源类型")

    _asyncio.create_task(sp.generate_resources_for_item(plan_id, item_id, user_id, types))
    return {"message": f"已触发 {len(types)} 种资源生成", "resource_types": types}


# ===========================================================
# 学生画像
# ===========================================================

@app.get("/profile", response_model=Optional[StudentProfileOut], tags=["profile"])
async def get_profile(
    user_id: int,
    db: AsyncSession = Depends(get_session),
):
    """获取当前用户画像，若尚未建立则返回 null。"""
    return await profile_svc.get_profile(user_id, db)


@app.put("/profile", response_model=StudentProfileOut, tags=["profile"])
async def update_profile(
    user_id: int,
    body: StudentProfileIn,
    db: AsyncSession = Depends(get_session),
):
    """手动更新用户画像。"""
    return await profile_svc.create_or_update_profile(user_id, body, db)


@app.get("/profile/history", response_model=list[StudentProfileOut], tags=["profile"])
async def get_profile_history(
    user_id: int,
    db: AsyncSession = Depends(get_session),
):
    """获取画像历史版本。"""
    return await profile_svc.get_profile_history(user_id, db)


@app.put("/user/account", response_model=UserOut, tags=["user"])
async def update_account(
    user_id: int,
    body: UserUpdate,
    db: AsyncSession = Depends(get_session),
):
    """更新用户名和邮箱。"""
    from backend.db.crud import select_one, update_by_id

    updates = {}
    if body.username is not None:
        existing = await select_one(db, User, filters={"username": body.username})
        if existing and existing.id != user_id:
            raise HTTPException(status_code=409, detail="用户名已被占用")
        updates["username"] = body.username
    if body.email is not None:
        if body.email != "":
            existing = await select_one(db, User, filters={"email": body.email})
            if existing and existing.id != user_id:
                raise HTTPException(status_code=409, detail="邮箱已被占用")
        updates["email"] = body.email if body.email != "" else None
        updates["email_verified"] = False

    if not updates:
        raise HTTPException(status_code=400, detail="没有需要更新的字段")

    ok = await update_by_id(db, User, user_id, data=updates)
    if not ok:
        raise HTTPException(status_code=404, detail="用户不存在")
    # 重新查询获取更新后的完整 User 对象以匹配 UserOut schema
    updated_user = await select_one(db, User, filters={"id": user_id})
    return updated_user


@app.delete("/user/account", tags=["user"])
async def delete_account(
    user_id: int,
    body: AccountDeleteIn,
    db: AsyncSession = Depends(get_session),
):
    """
    注销（硬删除）当前账号及其全部关联数据。

    安全校验：要求请求体携带用户名 + 密码进行双重确认，且用户名必须与
    user_id 对应账号一致，密码校验通过后方可执行。该操作不可逆。
    """
    from backend.db.crud import select_one, delete_user_cascade

    user = await select_one(db, User, filters={"id": user_id})
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")

    # 双重确认：用户名必须匹配，密码必须正确
    if body.username != user.username:
        raise HTTPException(status_code=400, detail="用户名不匹配，无法注销")
    if not verify_password(body.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="密码错误，无法注销")

    try:
        ok = await delete_user_cascade(db, user_id)
    except Exception as e:
        await db.rollback()
        logger.exception("[Account] 注销账号失败 user_id={}: {}", user_id, e)
        raise HTTPException(status_code=500, detail="注销失败，请稍后重试")

    if not ok:
        raise HTTPException(status_code=404, detail="用户不存在")

    logger.success("[Account] 账号已注销 user_id={}, username={}", user_id, user.username)
    return {"success": True, "message": "账号已注销"}


@app.get("/user", response_model=UserOut, tags=["user"])
async def get_user(
    user_id: int,
    db: AsyncSession = Depends(get_session),
):
    """获取用户基本信息（用户名、邮箱等）。"""
    from backend.db.crud import select_one

    user = await select_one(db, User, filters={"id": user_id})
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
    return user


# ===========================================================
# 对话（Agent 入口）
# ===========================================================

@app.get("/chat/sessions", response_model=list[ChatSessionOut], tags=["chat"])
async def list_sessions(user_id: int, db: AsyncSession = Depends(get_session)):
    """列举用户的所有对话会话。"""
    from backend.db.crud import select as db_select
    sessions = await db_select(db, ChatSession, filters={"user_id": user_id})
    return [ChatSessionOut.model_validate(s) for s in sessions]


@app.post("/chat/sessions", response_model=ChatSessionOut, tags=["chat"])
async def create_chat_session(user_id: int, db: AsyncSession = Depends(get_session)):
    """创建新的对话会话。"""
    from backend.db.crud import insert

    session = await insert(db, ChatSession, data={"user_id": user_id})
    return ChatSessionOut.model_validate(session)


async def _auto_title_session(session_id: int, user_msg: str, ai_msg: str | None):
    """后台任务：用 LLM 为新会话生成简短标题。"""
    try:
        from backend.services.llm import chat_completion
        from backend.db.database import _session_factory
        from backend.db.crud import update_by_id

        prompt = (
            f"根据以下对话的第一轮内容，生成一个简短的会话标题（不超过{app_config.chat.auto_title_max_chars}个字，不要引号和标点）。\n"
            f"用户：{user_msg[:app_config.chat.auto_title_message_truncate]}\n"
            f"助手：{(ai_msg or '')[:app_config.chat.auto_title_message_truncate]}\n"
            "标题："
        )
        title = await chat_completion([{"role": "user", "content": prompt}], max_tokens=app_config.chat.auto_title_max_tokens)
        title = title.strip().strip('"\'""「」').strip()[:app_config.chat.auto_title_final_length]
        if title and _session_factory:
            async with _session_factory() as db:
                await update_by_id(db, ChatSession, session_id, data={"title": title})
                await db.commit()
    except Exception as e:
        logger.warning(f"[auto_title] failed: {e}")


@app.post("/chat/{session_id}", tags=["chat"])
async def chat(
    session_id: int,
    user_id: int,
    body: ChatMessageIn,
    stream: bool = False,
    db: AsyncSession = Depends(get_session),
):
    """
    向 Agent 系统发送消息。
    stream=true 时返回 SSE 流式响应。
    """
    if stream:
        async def event_generator():
            async for event in stream_invoke(user_id, session_id, body.content, db):
                yield f"data: {event}\n\n"
        return StreamingResponse(event_generator(), media_type="text/event-stream")
    result = await invoke(user_id, session_id, body.content, db)

    # 刷新 last_used_at，并持久化本轮对话消息
    try:
        from backend.db.crud import select_by_id, update_by_id, insert as crud_insert
        chat_sess = await select_by_id(db, ChatSession, session_id)
        if chat_sess:
            await update_by_id(
                db, ChatSession, session_id,
                data={"last_used_at": datetime.utcnow()},
            )
            await crud_insert(db, ChatMessage, data={
                "session_id": session_id,
                "role": "user",
                "content": body.content,
            })
            if result.final_content:
                video_refs = result.metadata.get("video_refs") or []
                await crud_insert(db, ChatMessage, data={
                    "session_id": session_id,
                    "role": "assistant",
                    "content": result.final_content,
                    "resource_type": result.resource_type.value if result.resource_type else None,
                    "extra": {"video_refs": video_refs} if video_refs else None,
                })
            # 自动命名：会话尚无标题时，用 LLM 生成简短标题
            if not chat_sess.title:
                import asyncio
                asyncio.create_task(_auto_title_session(session_id, body.content, result.final_content))
    except Exception as e:
        logger.warning(f"聊天消息持久化失败: {e}")

    # 如果生成了资源，持久化到 resource_meta 表
    if result.resource_type and result.draft_content:
        try:
            from backend.db.crud import insert, select_one
            # 解析 kp_id → 知识点名称
            kp_id = result.kp_id or "unknown"
            kp_title = kp_id
            if kp_id.startswith("kp_"):
                node = await select_one(db, KGNode, filters={"id": kp_id})
                if node:
                    kp_title = node.name
            await insert(db, ResourceMeta, data={
                "user_id": user_id,
                "kp_id": kp_id,
                "resource_type": result.resource_type.value,
                "title": f"{kp_title} - {result.resource_type.value}",
                "content": result.draft_content,
            })
        except Exception as e:
            logger.warning(f"资源保存失败: {e}")

    # 检测多资源意图，触发批量生成
    batch_id = None
    extra_types = result.metadata.get("extra_resource_types", [])
    if extra_types and result.kp_id:
        try:
            from backend.models.schemas import BatchGenerateRequest, ResourceType as RT
            from backend.services import resource as resource_svc
            from backend.services.generation import run_batch_generation

            batch_req = BatchGenerateRequest(
                kp_id=result.kp_id,
                resource_types=[RT(t) for t in extra_types],
            )
            batch_out = await resource_svc.create_batch(user_id, batch_req, db)
            batch_id = str(batch_out.batch_id)

            # 构建子任务配置
            task_configs = []
            for task_item in batch_out.tasks:
                task_configs.append({
                    "task_id": task_item.task_id,
                    "request": {
                        "kp_id": result.kp_id,
                        "resource_type": task_item.resource_type.value,
                    },
                })
            import asyncio
            asyncio.create_task(run_batch_generation(
                batch_id=batch_out.batch_id,
                user_id=user_id,
                session_id=session_id,
                task_configs=task_configs,
            ))
            logger.info(f"[chat] 触发批量生成 batch_id={batch_id}, extra_types={extra_types}")
        except Exception as e:
            logger.warning(f"[chat] 批量生成触发失败: {e}")

    logger.info(
        "[chat] returning metadata keys=%s recommendations_count=%d" % (
        list(result.metadata.keys()),
        len(result.metadata.get("recommendations", []))
        )
    )

    # 学习目标异步刷新已在 merge_chat_updates 内部通过 asyncio.create_task 触发，
    # 与资源生成链路并行，且不依赖本响应是否成功返回，此处无需重复注册。

    # 将 batch_id 注入 metadata 供前端使用
    response_metadata = dict(result.metadata)
    if batch_id:
        response_metadata["batch_id"] = batch_id

    return {
        "content": result.final_content,
        "metadata": response_metadata,
        "profile_complete": result.profile_complete,
        "resource_type": result.resource_type.value if result.resource_type else None,
    }


from pydantic import BaseModel


class TitleIn(BaseModel):
    title: str


@app.get("/chat/{session_id}/messages", tags=["chat"])
async def get_session_messages(
    session_id: int,
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_session),
):
    """读取指定会话的历史消息列表。"""
    from backend.db.crud import select_by_id, select as db_select

    chat_sess = await select_by_id(db, ChatSession, session_id)
    if not chat_sess:
        raise HTTPException(status_code=404, detail="Session not found")
    if chat_sess.user_id != user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    messages = await db_select(
        db, ChatMessage,
        filters={"session_id": session_id},
        order_by=ChatMessage.created_at.asc(),
    )
    return [ChatMessageOut.model_validate(m) for m in messages]

@app.get("/documents/file/{filename}", tags=["documents"])
async def get_document_file(filename: str):
    """返回 uploaded_docs 目录下的 PDF 文件供前端 iframe 预览。

    只接受纯文件名，拒绝路径穿越。
    """
    from pathlib import Path
    # 防止路径穿越：文件名不允许含分隔符或上级引用
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="非法文件名")
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="仅支持 PDF 预览")
    upload_dir = Path(__file__).parent.parent / "uploaded_docs"
    file_path = upload_dir / filename
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(
        path=str(file_path),
        media_type="application/pdf",
        filename=filename,
        content_disposition_type="inline",
    )


@app.delete("/chat/sessions/{session_id}", tags=["chat"])
async def delete_chat_session(
    session_id: int,
    user_id: int,
    db: AsyncSession = Depends(get_session),
):
    """删除会话及其关联消息（ChatMessage 通过外键 CASCADE 自动删除）。"""
    from backend.db.crud import select_by_id, delete_by_id

    chat_sess = await select_by_id(db, ChatSession, session_id)
    if not chat_sess or chat_sess.user_id != user_id:
        raise HTTPException(status_code=404, detail="Session not found")
    await delete_by_id(db, ChatSession, session_id)
    return {"deleted": True}


@app.patch("/chat/sessions/{session_id}/title", tags=["chat"])
async def update_session_title(
    session_id: int,
    body: TitleIn,
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_session),
):
    """更新会话标题。"""
    from backend.db.crud import select_by_id, update_by_id

    chat_sess = await select_by_id(db, ChatSession, session_id)
    if not chat_sess or chat_sess.user_id != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    await update_by_id(db, ChatSession, session_id, data={"title": body.title})
    return {"ok": True}


# ===========================================================
# 知识图谱
# ===========================================================

@app.get("/kg/graph", response_model=KGGraphOut, tags=["knowledge-graph"])
async def get_kg_graph(
    root_id: Optional[str] = None,
    doc_id: Optional[str] = None,
    user_id: int = Depends(get_current_user_id),
    depth: int = 3,
    db: AsyncSession = Depends(get_session),
):
    """获取知识图谱子图，供前端 ECharts 渲染。支持按 doc_id 过滤 + depth 控制展开层数。

    优化：depth 过滤下推到 DB 层，避免加载全量数据。
    """
    from sqlalchemy import select as sa_select, and_
    from backend.db.models import KGNode, KGEdge

    type_levels = ["Course", "Chapter", "KnowledgePoint", "SubPoint", "Concept"]
    allowed_types = type_levels[:depth]

    # 基础过滤条件
    base_conditions = []
    if doc_id:
        base_conditions.append(KGNode.course_id == doc_id)
    # 始终按用户隔离：用户自己的节点 + 公共节点（user_id 为 NULL）
    from sqlalchemy import or_
    base_conditions.append(or_(KGNode.user_id == user_id, KGNode.user_id == None))

    if root_id:
        # 有 root_id：加载所有节点建立邻接表，再 BFS 扩展
        all_nodes = await db.execute(
            sa_select(KGNode).where(and_(*base_conditions)) if base_conditions
            else sa_select(KGNode)
        )
        all_nodes = all_nodes.scalars().all()
        node_map = {n.id: n for n in all_nodes}

        # 建立邻接表（只看 IS_PART_OF / CONTAINS 关系）
        children_adj: dict[str, set[str]] = {n.id: set() for n in all_nodes}
        parent_adj: dict[str, set[str]] = {n.id: set() for n in all_nodes}
        for e in await db.execute(sa_select(KGEdge)):
            e = e[0]
            if e.relation in ("IS_PART_OF", "CONTAINS"):
                children_adj.setdefault(e.source_id, set()).add(e.target_id)
                parent_adj.setdefault(e.target_id, set()).add(e.source_id)

        # BFS 找 root 的所有后代（不限层级，先找完全部子树）
        descendants: set[str] = {root_id}
        frontier = [root_id]
        while frontier:
            next_frontier = []
            for nid in frontier:
                for child in children_adj.get(nid, []):
                    if child not in descendants and child in node_map:
                        descendants.add(child)
                        next_frontier.append(child)
            frontier = next_frontier

        # 再按 depth 类型过滤（只保留 allowed_types 的节点）
        reachable = {nid for nid in descendants if node_map[nid].node_type in allowed_types}

        # 如果 root 自身类型不在 allowed_types，也加入（作为入口）
        if root_id in descendants and root_id in node_map:
            reachable.add(root_id)
    else:
        # 无 root_id：直接按 node_type 过滤，大幅减少加载量
        conditions = base_conditions + [KGNode.node_type.in_(allowed_types)]
        all_nodes = await db.execute(
            sa_select(KGNode).where(and_(*conditions)) if conditions
            else sa_select(KGNode).where(KGNode.node_type.in_(allowed_types))
        )
        all_nodes = all_nodes.scalars().all()
        node_map = {n.id: n for n in all_nodes}
        reachable = set(node_map.keys())

    # 加载关联边（只看两端节点都在 reachable 里的）
    if reachable:
        edges_result = await db.execute(
            sa_select(KGEdge).where(
                and_(KGEdge.source_id.in_(reachable), KGEdge.target_id.in_(reachable))
            )
        )
        filtered_edges = edges_result.scalars().all()
    else:
        filtered_edges = []

    kg_nodes = [
        KGNodeOut(
            id=n.id,
            type=KGNodeType(n.node_type),
            name=n.name,
            difficulty=None,
            is_core=False,
            extra={"description": n.description or ""},
        )
        for n in all_nodes if n.id in reachable
    ]
    kg_edges = [
        KGEdgeOut(
            source_id=e.source_id,
            target_id=e.target_id,
            relation=KGRelation(e.relation),
        )
        for e in filtered_edges
    ]
    return KGGraphOut(nodes=kg_nodes, edges=kg_edges)


@app.post("/kg/build", tags=["knowledge-graph"])
async def build_kg_endpoint(
    background_tasks: BackgroundTasks,
    body: dict = Body(...),
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_session),
):
    """异步构建知识图谱，立即返回任务 ID 供轮询。"""
    doc_id: str = body.get("doc_id")
    from backend.db.crud import insert
    from backend.db.models import KGBuildTask
    from backend.models.schemas import KGBuildTaskOut

    logger.info(f"[POST /kg/build] 创建异步构建任务，doc_id={doc_id}")
    task = await insert(db, KGBuildTask, data={
        "doc_id": doc_id,
        "user_id": user_id,
        "status": "pending",
        "progress": 0,
        "stage": "排队中",
    })

    from backend.services.kg_builder import run_kg_build
    background_tasks.add_task(run_kg_build, task.id, doc_id, db, user_id)

    return KGBuildTaskOut(
        task_id=task.id,
        doc_id=doc_id,
        status="pending",
        progress=0,
        stage="排队中",
    )


@app.get("/kg/build/{task_id}/status", tags=["knowledge-graph"])
async def get_kg_build_status(
    task_id: int,
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_session),
):
    """轮询知识图谱构建任务状态。"""
    from backend.db.crud import select_one
    from backend.db.models import KGBuildTask
    from backend.models.schemas import KGBuildTaskOut

    task = await select_one(db, KGBuildTask, filters={"id": task_id})
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    if task.user_id and task.user_id != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    return KGBuildTaskOut(
        task_id=task.id,
        doc_id=task.doc_id,
        status=task.status,
        progress=task.progress,
        stage=task.stage,
        nodes_count=task.nodes_count,
        edges_count=task.edges_count,
        error_message=task.error_message,
    )


@app.get("/kg/build/by-doc/{doc_id}/status", tags=["knowledge-graph"])
async def get_kg_build_status_by_doc(
    doc_id: str,
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_session),
):
    """按 doc_id 查询最新的构建任务状态（刷新浏览器后恢复跟踪）。"""
    from backend.db.crud import select as db_select, select_one
    from backend.db.models import KGBuildTask
    from backend.models.schemas import KGBuildTaskOut

    # 校验文档归属
    doc_resource = await select_one(db, ResourceMeta, filters={"kp_id": doc_id, "user_id": user_id})
    if not doc_resource:
        raise HTTPException(status_code=403, detail="Access denied")

    tasks = await db_select(
        db, KGBuildTask, filters={"doc_id": doc_id},
        order_by=KGBuildTask.created_at.desc(), limit=1,
    )
    if not tasks:
        return {"status": "none"}
    task = tasks[0]
    return KGBuildTaskOut(
        task_id=task.id,
        doc_id=task.doc_id,
        status=task.status,
        progress=task.progress,
        stage=task.stage,
        nodes_count=task.nodes_count,
        edges_count=task.edges_count,
        error_message=task.error_message,
    )


# ===========================================================
# 资源生成
# ===========================================================

@app.post("/generate", response_model=GenerateTaskOut, tags=["generate"])
async def start_generation(
    user_id: int,
    body: GenerateRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_session),
):
    """
    触发异步资源生成任务。
    返回 task_id 供前端轮询 /generate/{task_id}/status。
    """
    import asyncio
    from backend.services.generation import run_generation
    task = await resource_svc.create_generation_task(user_id, body, db)

    # 获取或创建会话 ID
    session_id = str(generate_id())

    # 将 body 转为可序列化的 dict，避免 Pydantic 模型在 background task 中反序列化失败
    body_dict = body.model_dump()

    # 使用 asyncio.create_task 在事件循环中调度后台任务
    asyncio.create_task(
        run_generation(
            task.task_id,
            user_id,
            session_id,
            body_dict,
        )
    )
    return task


@app.get("/generate/{task_id}/status", response_model=GenerateTaskOut, tags=["generate"])
async def get_generation_status(
    task_id: int,
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_session),
):
    """轮询生成任务状态与进度。"""
    task = await resource_svc.get_task_status(task_id, db)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@app.post("/generate/batch", response_model=BatchGenerateOut, tags=["generate"])
async def start_batch_generation(
    user_id: int,
    body: BatchGenerateRequest,
    db: AsyncSession = Depends(get_session),
):
    """
    批量资源生成：一次性生成多种资源类型。
    返回 batch_id 供前端轮询 /generate/batch/{batch_id}/status。
    """
    from backend.services.generation import run_batch_generation

    batch_out = await resource_svc.create_batch(user_id, body, db)

    # 构造每个子任务的配置
    task_configs = []
    for task_item in batch_out.tasks:
        req_dict = {
            "kp_id": body.kp_id,
            "resource_type": task_item.resource_type.value,
            "num_questions": body.num_questions,
            "question_type_counts": body.question_type_counts,
            "extra_params": {},
        }
        task_configs.append({"task_id": task_item.task_id, "request": req_dict})

    session_id = generate_id()
    asyncio.create_task(
        run_batch_generation(batch_out.batch_id, user_id, session_id, task_configs)
    )

    return batch_out


@app.get("/generate/batch/{batch_id}/status", response_model=BatchGenerateOut, tags=["generate"])
async def get_batch_generation_status(
    batch_id: int,
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_session),
):
    """轮询批量生成任务状态与进度。"""
    result = await resource_svc.get_batch_status(batch_id, db)
    if not result:
        raise HTTPException(status_code=404, detail="Batch not found")
    return result


@app.post("/generate/smart", tags=["generate"])
async def smart_plan_resources(
    user_id: int,
    kp_id: str = Body(..., embed=True),
    db: AsyncSession = Depends(get_session),
):
    """
    智能推荐资源类型组合：调用 planner LLM 根据用户画像和知识点推荐。
    返回推荐的资源类型列表。
    """
    from backend.agents.planner_agent import plan_resource_types

    types = await plan_resource_types(user_id, kp_id, db)
    return {"resource_types": [rt.value for rt in types]}


# ===========================================================
# 资源库
# ===========================================================

@app.get("/resources", response_model=ResourceListOut, tags=["resources"])
async def list_resources(
    user_id: int,
    resource_type: Optional[str] = None,
    kp_id: Optional[str] = None,
    skip: int = 0,
    limit: int = app_config.pagination.default_limit,
    db: AsyncSession = Depends(get_session),
):
    """分页列举用户的学习资源。"""
    return await resource_svc.list_resources(user_id, db, resource_type, kp_id, skip, limit)


@app.get("/resources/stats", tags=["resources"])
async def get_resource_stats(user_id: int, db: AsyncSession = Depends(get_session)):
    """返回用户的资源统计：按类型计数的字典。一次 GROUP BY 替代 5 次 COUNT 查询。"""
    from sqlalchemy import func, select

    rows = await db.execute(
        select(ResourceMeta.resource_type, func.count(ResourceMeta.id))
        .where(ResourceMeta.user_id == user_id)
        .group_by(ResourceMeta.resource_type)
    )
    stats = {rt: 0 for rt in ["doc", "mindmap", "quiz", "code", "summary", "animation"]}
    for rt, cnt in rows.all():
        if rt in stats:
            stats[rt] = cnt
    return stats


@app.get("/resources/{resource_id}", response_model=ResourceMetaOut, tags=["resources"])
async def get_resource(
    resource_id: int,
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_session),
):
    """获取单个资源详情。"""
    res = await resource_svc.get_resource(resource_id, db)
    if not res:
        raise HTTPException(status_code=404)
    if res.user_id != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    return res


@app.delete("/resources/{resource_id}", tags=["resources"])
async def delete_resource(
    resource_id: int,
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_session),
):
    """删除资源。"""
    res = await resource_svc.get_resource(resource_id, db)
    if not res:
        raise HTTPException(status_code=404)
    if res.user_id != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    await resource_svc.delete_resource(resource_id, db)
    return {"deleted": True}


# ===========================================================
# 测验
# ===========================================================

@app.get("/resources/{resource_id}/quiz", response_model=list[QuizItemOut], tags=["quiz"])
async def get_quiz_items(
    resource_id: int,
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_session),
):
    """获取某资源下的所有题目。"""
    from backend.db.crud import select as db_select

    res = await resource_svc.get_resource(resource_id, db)
    if not res:
        raise HTTPException(status_code=404, detail="Resource not found")
    if res.user_id != user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    # 优先从 content_json.items 读取（AI 生成的题目存储在此）
    if res.content_json and res.content_json.get("items"):
        items = res.content_json["items"]
        return [
            QuizItemOut(
                id=item.get("id") or string_to_id(f"{resource_id}-{idx}"),
                kp_id=res.kp_id,
                question_type=QuestionType(item["question_type"]) if item.get("question_type") else QuestionType.single,
                difficulty=item.get("difficulty"),
                stem=item["stem"],
                options=item.get("options"),
                answer=item["answer"],
                explanation=item.get("explanation"),
            )
            for idx, item in enumerate(items)
        ]

    # 回退：从 QuizItem 表读取
    quiz_items = await db_select(db, QuizItem, filters={"resource_id": resource_id})
    return [
        QuizItemOut(
            id=item.id,
            kp_id=item.kp_id,
            question_type=item.question_type,
            difficulty=None,
            stem=item.stem,
            options=item.options,
            answer=item.answer,
            explanation=item.explanation,
        )
        for item in quiz_items
    ]


@app.post("/quiz/submit", response_model=QuizAttemptOut, tags=["quiz"])
async def submit_quiz(
    user_id: int,
    body: QuizSubmitIn,
    db: AsyncSession = Depends(get_session),
):
    """提交答题记录，返回批改结果。"""
    from backend.db.crud import select_one, insert
    import re
    import ast

    # 先尝试从 QuizItem 表查找
    quiz_item = await select_one(db, QuizItem, filters={"id": body.quiz_item_id})

    # 如果找不到，尝试从用户的 quiz 资源的 content_json 中查找
    found_answer = None
    found_qtype = None
    found_kp_id = None
    if not quiz_item:
        from backend.db.crud import select
        resources = await select(db, ResourceMeta, filters={"user_id": user_id, "resource_type": "quiz"})
        for res in resources:
            if res.content_json and res.content_json.get("items"):
                for idx, item in enumerate(res.content_json["items"]):
                    expected_id = str(string_to_id(f"{res.id}-{idx}"))
                    if expected_id == str(body.quiz_item_id):
                        found_answer = item.get("answer")
                        found_qtype = item.get("question_type")
                        found_kp_id = res.kp_id
                        break
                if found_answer is not None:
                    break

    if not quiz_item and found_answer is None:
        raise HTTPException(status_code=404, detail="Quiz item not found")

    # 题目来自 content_json 但不在 quiz_item 表中，先持久化以满足外键约束
    if not quiz_item and found_answer is not None:
        quiz_item = await insert(
            db, QuizItem,
            data={
                "id": body.quiz_item_id,
                "resource_id": res.id,
                "kp_id": found_kp_id,
                "question_type": found_qtype or "single",
                "stem": item.get("stem", ""),
                "options": item.get("options"),
                "answer": str(found_answer),
                "explanation": item.get("explanation"),
                "order_index": idx,
            },
        )

    def extract_letters(text: str) -> set[str]:
        return set(re.findall(r'\b([A-Z])\b', text))

    question_type = quiz_item.question_type if quiz_item else (QuestionType(found_qtype) if found_qtype else QuestionType.single)
    answer = quiz_item.answer if quiz_item else found_answer

    if question_type == QuestionType.single:
        user_letters = extract_letters(str(body.user_answer))
        correct_letters = extract_letters(str(answer))
        is_correct = user_letters == correct_letters
    elif question_type == QuestionType.multi:
        user_letters = extract_letters(str(body.user_answer))
        try:
            correct_list = ast.literal_eval(str(answer))
            correct_letters = set(correct_list)
        except Exception:
            correct_letters = extract_letters(str(answer))
        is_correct = user_letters == correct_letters
    else:
        is_correct = str(body.user_answer).strip().lower() == str(answer).strip().lower()

    logger.info(f"[QuizSubmit] quiz_item_id={body.quiz_item_id}, question_type={question_type}, user_answer={body.user_answer!r}, correct_answer={answer!r}, is_correct={is_correct}")
    score = 1.0 if is_correct else 0.0

    attempt = await insert(
        db, QuizAttempt,
        data={
            "quiz_item_id": body.quiz_item_id,
            "user_id": user_id,
            "user_answer": str(body.user_answer),
            "is_correct": is_correct,
            "score": score,
            "kp_id": quiz_item.kp_id if quiz_item else found_kp_id,
        },
    )

    # 答对时写入学习记录，记录学习行为和时长
    kp = quiz_item.kp_id if quiz_item else found_kp_id
    if kp:
        # 时长：优先用前端上报的，否则默认估算 30 秒/题
        quiz_duration = body.duration_seconds if body.duration_seconds else 30
        await insert(db, LearningRecord, data={
            "user_id": user_id,
            "kp_id": kp,
            "action": "quiz",
            "duration_seconds": quiz_duration,
        })

    # 测验提交后自动更新学生画像（基于正确率统计）
    if kp:
        from backend.services.profile import update_profile_from_quiz
        try:
            await update_profile_from_quiz(user_id, kp, db)
        except Exception as e:
            logger.warning(f"[QuizSubmit] 自动更新画像失败: {e}")

    return QuizAttemptOut(
        id=attempt.id,
        quiz_item_id=attempt.quiz_item_id,
        user_answer=attempt.user_answer,
        is_correct=attempt.is_correct,
        score=attempt.score,
        kp_id=attempt.kp_id,
        created_at=attempt.created_at,
    )


@app.get("/quiz/attempts", response_model=list[QuizAttemptOut], tags=["quiz"])
async def get_quiz_attempts(
    user_id: int,
    skip: int = 0,
    limit: int = app_config.pagination.quiz_attempts_limit,
    db: AsyncSession = Depends(get_session),
):
    """获取用户的答题历史。"""
    import sqlalchemy as sa

    query = (
        sa.select(QuizAttempt, KGNode.name, QuizItem)
        .select_from(QuizAttempt)
        .outerjoin(KGNode, QuizAttempt.kp_id == KGNode.id)
        .outerjoin(QuizItem, QuizAttempt.quiz_item_id == QuizItem.id)
        .where(QuizAttempt.user_id == user_id)
        .order_by(QuizAttempt.created_at.desc())
        .limit(limit)
        .offset(skip)
    )
    rows = (await db.execute(query)).all()

    # 收集 content_json 中缺失的题目：尝试从 ResourceMeta.content_json 中补充
    resource_cache = {}  # kp_key -> flat list of all items for that kp
                          # "_res_{res_id}" -> list of items for specific resource (for UUID matching)

    def find_item_content(quiz_item_id: int) -> dict | None:
        """从 resource_cache 中查找题目内容（quiz_item_id 为虚拟 UUID，格式为 {resource_id}-{idx}）"""
        quiz_id_str = str(quiz_item_id)
        for res_id, items in resource_cache.items():
            if res_id.startswith("_res_"):
                for idx, item in enumerate(items):
                    expected = str(string_to_id(f"{res_id[5:]}-{idx}"))
                    if expected == quiz_id_str:
                        return item
        return None

    # 构建结果，处理 qi 为 None 的情况
    result = []
    for a, kp_name, qi in rows:
        if qi is not None:
            stem = qi.stem
            options = qi.options
            answer = qi.answer
            explanation = qi.explanation
            question_type = qi.question_type if qi.question_type else None
            difficulty = None
        else:
            # 尝试通过 kp_id 查找同知识点的 quiz 资源，从其 content_json 中补充题目内容
            kp_key = a.kp_id or ""
            if not kp_key:
                stem = None
                options = None
                answer = None
                explanation = None
                question_type = None
                difficulty = None
            elif kp_key not in resource_cache:
                res_list = await db.execute(
                    sa.select(ResourceMeta).where(
                        ResourceMeta.kp_id == kp_key,
                        ResourceMeta.resource_type == "quiz",
                    )
                )
                all_items = []
                for res in res_list.scalars().all():
                    if res.content_json and res.content_json.get("items"):
                        items = res.content_json["items"]
                        all_items.extend(items)
                        resource_cache[f"_res_{res.id}"] = items
                resource_cache[kp_key] = all_items

            item = find_item_content(a.quiz_item_id)
            if item:
                stem = item.get("stem")
                options = item.get("options")
                answer = item.get("answer")
                explanation = item.get("explanation")
                question_type = item.get("question_type")
                difficulty = item.get("difficulty")
            else:
                stem = None
                options = None
                answer = None
                explanation = None
                question_type = None
                difficulty = None

        result.append(QuizAttemptOut(
            id=a.id,
            quiz_item_id=a.quiz_item_id,
            user_answer=a.user_answer,
            is_correct=a.is_correct,
            score=a.score,
            kp_id=a.kp_id,
            kp_name=kp_name or (a.kp_id if a.kp_id else None),
            created_at=a.created_at,
            stem=stem,
            options=options,
            answer=answer,
            explanation=explanation,
            question_type=question_type,
            difficulty=difficulty,
        ))
    return result


# ===========================================================
# 学习路径
# ===========================================================

from backend.services import pathway as pathway_svc


@app.get("/pathways", response_model=list[LearningPathOut], tags=["pathway"])
async def list_pathways(user_id: int, db: AsyncSession = Depends(get_session)):
    """列举用户的学习路径。"""
    return await pathway_svc.list_pathways(user_id, db)


@app.post("/pathways", response_model=LearningPathOut, tags=["pathway"])
async def create_pathway(
    user_id: int,
    body: LearningPathCreate,
    db: AsyncSession = Depends(get_session),
):
    """创建新学习路径。"""
    return await pathway_svc.create_pathway(user_id, body, db)


@app.get("/pathways/{path_id}", response_model=LearningPathOut, tags=["pathway"])
async def get_pathway(
    path_id: int,
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_session),
):
    """获取单条学习路径详情。"""
    from backend.db.crud import select_one as db_select_one

    path_row = await db_select_one(db, LearningPath, filters={"id": path_id})
    if not path_row:
        raise HTTPException(status_code=404, detail="Pathway not found")
    if path_row.user_id != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    result = await pathway_svc.get_pathway(path_id, db)
    if not result:
        raise HTTPException(status_code=404, detail="Pathway not found")
    return result


@app.put("/pathways/{path_id}", response_model=LearningPathOut, tags=["pathway"])
async def update_pathway(
    path_id: int,
    user_id: int,
    body: LearningPathUpdate,
    db: AsyncSession = Depends(get_session),
):
    """更新学习路径标题/描述。"""
    result = await pathway_svc.update_pathway(path_id, user_id, body, db)
    if not result:
        raise HTTPException(status_code=404, detail="Pathway not found")
    return result


@app.delete("/pathways/{path_id}", tags=["pathway"])
async def delete_pathway(
    path_id: int,
    user_id: int,
    db: AsyncSession = Depends(get_session),
):
    """删除学习路径（级联删除路径项）。"""
    deleted = await pathway_svc.delete_pathway(path_id, user_id, db)
    if not deleted:
        raise HTTPException(status_code=404, detail="Pathway not found")
    return {"deleted": True}


@app.post("/pathways/{path_id}/items", response_model=LearningPathItemOut, tags=["pathway"])
async def add_pathway_item(
    path_id: int,
    user_id: int,
    body: LearningPathItemCreate,
    db: AsyncSession = Depends(get_session),
):
    """向学习路径添加知识点项。"""
    result = await pathway_svc.add_pathway_item(path_id, user_id, body, db)
    if not result:
        raise HTTPException(status_code=404, detail="Pathway not found or unauthorized")
    return result


@app.put("/pathways/{path_id}/items/{item_id}", response_model=LearningPathItemOut, tags=["pathway"])
async def update_pathway_item(
    path_id: int,
    item_id: int,
    user_id: int,
    body: LearningPathItemUpdate,
    db: AsyncSession = Depends(get_session),
):
    """更新学习路径项（顺序/完成状态）。"""
    result = await pathway_svc.update_pathway_item(item_id, user_id, body, db)
    if not result:
        raise HTTPException(status_code=404, detail="Item not found or unauthorized")
    return result


@app.delete("/pathways/{path_id}/items/{item_id}", tags=["pathway"])
async def remove_pathway_item(
    path_id: int,
    item_id: int,
    user_id: int,
    db: AsyncSession = Depends(get_session),
):
    """从学习路径移除知识点项。"""
    deleted = await pathway_svc.remove_pathway_item(item_id, user_id, db)
    if not deleted:
        raise HTTPException(status_code=404, detail="Item not found or unauthorized")
    return {"deleted": True}


# ===============================================================
# 文档导入
# ===============================================================

@app.post("/documents/import", tags=["documents"])
async def import_document(
    user_id: int,
    file: UploadFile = File(...),
    title: Optional[str] = None,
    db: AsyncSession = Depends(get_session),
):
    """
    上传并导入文档（支持 PDF / DOCX / DOC / Markdown / TXT）。

    - 保存文件到 uploaded_docs 目录
    - 转换为 Markdown 并切分为文本块
    - 索引到向量库（供 RAG 检索使用）
    - 创建资源记录到数据库
    """
    file_name = file.filename or "unknown"
    suffix = file_name.lower().rsplit(".", 1)[-1] if "." in file_name else ""
    if f".{suffix}" not in document_svc.SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"不支持的文件格式：.{suffix}，支持：{', '.join(sorted(document_svc.SUPPORTED_SUFFIXES))}",
        )

    try:
        content = await file.read()
        saved_path = document_svc.save_uploaded_file(content, file_name)
        logger.info(f"[import_document] 文件 {file_name} 已保存到 {saved_path}，开始处理...")
        result = await document_svc.import_document(
            file_path=saved_path,
            user_id=user_id,
            title=title,
            db=db,
        )
        return {
            "success": True,
            "doc_id": result["doc_id"],
            "title": result["title"],
            "file_name": result["file_name"],
            "chunks": result["chunks"],
            "indexed": result["indexed"],
            "resource_id": result["resource_id"],
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"导入失败：{str(e)}",
        )


@app.post("/documents/import/async", tags=["documents"])
async def import_document_async(
    user_id: int,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: Annotated[Optional[str], Form()] = None,
):
    """
    异步导入文档（支持 PDF / DOCX / DOC / Markdown / TXT）。
    立即返回 task_id，前端轮询 /documents/import/{task_id}/status。
    """
    logger.info(f"[import_document_async] received title={title!r}, file.filename={file.filename!r}")
    file_name = file.filename or "unknown"
    suffix = file_name.lower().rsplit(".", 1)[-1] if "." in file_name else ""
    if f".{suffix}" not in document_svc.SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"不支持的文件格式：.{suffix}，支持：{', '.join(sorted(document_svc.SUPPORTED_SUFFIXES))}",
        )

    content = await file.read()
    try:
        saved_path = document_svc.save_uploaded_file(content, file_name)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    task_id = str(generate_id())
    _doc_import_tasks[task_id] = {
        "status": "running",
        "progress": 5,
        "stage": "saving",
        "doc_id": None,
        "error_message": None,
        "result": None,
    }

    async def _run_import():
        def _cb(stage: str, pct: int):
            _doc_import_tasks[task_id]["stage"] = stage
            _doc_import_tasks[task_id]["progress"] = pct

        logger.info(f"[_run_import] task_id={task_id}, title={title!r}")

        try:
            sf = _db_module._session_factory
            if sf is None:
                raise RuntimeError("Database not initialized.")
            async with sf() as bg_db:
                try:
                    result = await document_svc.import_document_with_progress(
                        file_path=saved_path,
                        user_id=user_id,
                        title=title,
                        db=bg_db,
                        progress_callback=_cb,
                    )
                    await bg_db.commit()
                except Exception:
                    await bg_db.rollback()
                    raise
            _doc_import_tasks[task_id]["status"] = "done"
            _doc_import_tasks[task_id]["progress"] = 100
            _doc_import_tasks[task_id]["stage"] = "done"
            _doc_import_tasks[task_id]["doc_id"] = result.get("doc_id")
            _doc_import_tasks[task_id]["result"] = result
        except Exception as e:
            logger.exception(f"[import_document_async] 后台任务失败: {e}")
            _doc_import_tasks[task_id]["status"] = "failed"
            _doc_import_tasks[task_id]["error_message"] = str(e)

    background_tasks.add_task(_run_import)
    return {
        "task_id": task_id,
        "status": "running",
        "progress": 5,
        "stage": "saving",
    }


@app.get("/documents/import/{task_id}/status", tags=["documents"])
async def get_import_task_status(task_id: str):
    """轮询异步 PDF 导入任务状态。"""
    task = _doc_import_tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在或已过期")
    return {"task_id": task_id, **task}


@app.post("/documents/cleanup", tags=["documents"])
async def manual_cleanup(
    dry_run: bool = True,
    retention_days: int = 30,
    orphan_retention_days: int = 7,
    db: AsyncSession = Depends(get_session),
):
    """
    手动触发文档文件清理。

    - **dry_run**: true 时仅预览不实际删除（默认 true）
    - **retention_days**: 已索引文件的保留天数
    - **orphan_retention_days**: 孤儿文件的保留天数
    """
    from backend.services.cleanup import cleanup_uploaded_docs
    result = await cleanup_uploaded_docs(
        retention_days=retention_days,
        orphan_retention_days=orphan_retention_days,
        dry_run=dry_run,
    )
    return {"success": True, "dry_run": dry_run, **result}


@app.get("/documents", tags=["documents"])
async def list_documents(
    user_id: int,
    skip: int = 0,
    limit: int = app_config.pagination.default_limit,
    db: AsyncSession = Depends(get_session),
):
    """列举用户导入的文档列表（排除系统生成的资源文档）。"""
    from sqlalchemy import select as sa_select, and_

    stmt = (
        sa_select(ResourceMeta)
        .where(and_(
            ResourceMeta.user_id == user_id,
            ResourceMeta.resource_type == "doc",
            ResourceMeta.kp_id.like("doc_%"),
        ))
        .order_by(ResourceMeta.created_at.desc())
        .offset(skip)
        .limit(limit)
    )
    result = await db.execute(stmt)
    resources = result.scalars().all()
    return [
        {
            "id": str(r.id),
            "title": r.title or "无标题",
            "kp_id": r.kp_id,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in resources
    ]


@app.delete("/documents/{doc_id}", tags=["documents"])
async def delete_document(
    doc_id: str,
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_session),
):
    """删除文档（同时从向量库移除）。"""
    from backend.db.vector import delete_by_doc_id

    # 校验文档归属
    from backend.db.crud import select_one
    resource = await select_one(db, ResourceMeta, filters={"kp_id": doc_id, "user_id": user_id})
    if not resource:
        raise HTTPException(status_code=403, detail="Access denied")

    try:
        await delete_by_doc_id(doc_id)
    except Exception:
        pass

    # 复用 delete_resource：统一清理 quiz/task/learning_record 等子表，避免外键约束错误
    await resource_svc.delete_resource(resource.id, db)
    return {"deleted": True}


# ===========================================================
# 学习记录
# ===========================================================

@app.post("/records", response_model=LearningRecordOut, tags=["records"])
async def add_record(
    user_id: int,
    body: LearningRecordCreate,
    db: AsyncSession = Depends(get_session),
):
    """记录学习行为。"""
    return await resource_svc.record_learning(user_id, body, db)


@app.get("/records", response_model=list[LearningRecordOut], tags=["records"])
async def list_records(
    user_id: int,
    kp_id: Optional[str] = None,
    skip: int = 0,
    limit: int = app_config.pagination.default_limit,
    db: AsyncSession = Depends(get_session),
):
    """获取用户的学习记录列表，可按 kp_id 过滤。"""
    return await resource_svc.list_learning_records(user_id, db, skip, limit, kp_id)


# ===========================================================
# RAG 评估端点
# ===========================================================

@app.post("/eval/rag/query", tags=["evaluation"])
async def evaluate_rag_query(
    kp_name: str = Body(..., embed=True),
    query: str = Body(default="", embed=True),
    generated_content: str = Body(default="", embed=True),
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_session),
):
    """
    对单个知识点执行完整的 RAG 四维度 LLM-as-Judge 评估。

    流程：
    1. 用当前 RAG 管线检索 kp_name 相关文档
    2. 若未提供 generated_content，用 doc_agent 实时生成
    3. 执行 Judge 1-4 评估并返回完整结果
    """
    from backend.evaluation.judge import RAGJudge
    from backend.rag.retriever import retrieve_by_kp

    # 1. 检索
    chunks = await retrieve_by_kp(
        kp_name,
        n_results=app_config.rag.n_results,
        user_id=str(user_id),
    )

    if not chunks:
        return {
            "error": "未检索到相关文档，请先导入知识库",
            "kp_name": kp_name,
            "chunks": [],
        }

    # 2. 若无提供 content，用简单 prompt 生成
    content = generated_content
    if not content:
        from backend.services.llm import chat_completion
        retrieved_text = "\n\n".join(c.text[:800] for c in chunks[:5])
        prompt = f"""请根据以下参考资料，为知识点"{kp_name}"生成一份学习文档。
要求使用 Markdown 格式，在引用处标注 [n]。

参考资料：
{retrieved_text}

知识点：{kp_name}"""
        try:
            content = await chat_completion(
                [{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2000,
            )
        except Exception as e:
            return {"error": f"生成内容失败: {e}", "kp_name": kp_name}

    # 3. LLM-as-Judge 评估
    judge = RAGJudge()
    _query = query or f"知识点：{kp_name}"
    result = await judge.evaluate_full(
        query=_query,
        kp_name=kp_name,
        retrieved_chunks=chunks,
        generated_content=content,
    )

    # 附加 chunk 信息供前端展示
    result["chunks"] = [
        {
            "chunk_id": c.chunk_id,
            "score": c.score,
            "doc_id": c.doc_id,
            "source": c.source,
            "text_preview": c.text[:200],
        }
        for c in chunks
    ]
    result["generated_content"] = content

    return result


@app.get("/eval/rag/report", tags=["evaluation"])
async def get_rag_eval_report(
    period: str = "daily",
):
    """
    获取 RAG 评估报告。

    :param period: "daily"（日报）或 "weekly"（周报）
    """
    from backend.evaluation.collector import collector
    from backend.evaluation.reporter import RAGReporter

    reporter = RAGReporter()
    records = collector.get_recent_records(n=500)

    if period == "weekly":
        report = reporter.generate_weekly_report(records)
    else:
        report = reporter.generate_daily_report(records)

    return {
        "markdown": reporter.to_markdown(report),
        "summary": reporter.to_summary(report),
        "report": report.model_dump(),
    }


@app.get("/eval/rag/records", tags=["evaluation"])
async def list_eval_records(
    n: int = 20,
):
    """获取最近 N 条 RAG 评估记录。"""
    from backend.evaluation.collector import collector

    records = collector.get_recent_records(n)
    return [
        {
            "agent_type": r.agent_type,
            "kp_name": r.kp_name,
            "timestamp": r.timestamp.isoformat() if r.timestamp else None,
            "n_retrieved": r.n_retrieved,
            "draft_length": r.draft_length,
            "safety_passed": r.safety_passed,
            "safety_issues": r.safety_issues,
            "faithfulness": r.faithfulness_score,
            "hallucination_rate": r.hallucination_rate_val,
            "completeness": r.completeness_score,
            "scores": r.retrieval_record.scores if r.retrieval_record else [],
        }
        for r in records
    ]


# ===========================================================
# 学习效果评估 — 综合分析仪表盘
# ===========================================================

@app.get("/analytics/dashboard", tags=["analytics"])
async def get_learning_analytics(
    user_id: int,
    db: AsyncSession = Depends(get_session),
):
    """
    综合学习分析仪表盘，返回：
    1. quiz_mastery: 各知识点掌握度（正确率+做题数）
    2. learning_behavior: 学习行为统计（总时长、日活跃度曲线）
    3. forgetting_curve: 遗忘曲线提醒（需要复习的知识点）
    4. radar_data: 能力雷达图数据
    """
    import sqlalchemy as sa
    from datetime import datetime, timedelta
    from collections import defaultdict

    now = datetime.utcnow()

    # ── 1. 知识掌握度量化（正确率 70% + 答题效率 30%） ──
    quiz_rows = await db.execute(
        sa.select(
            QuizAttempt.kp_id,
            sa.func.count().label("total"),
            sa.func.sum(sa.case((QuizAttempt.is_correct == True, 1), else_=0)).label("correct"),
        ).where(
            QuizAttempt.user_id == user_id,
            QuizAttempt.kp_id.isnot(None),
        ).group_by(QuizAttempt.kp_id)
    )
    quiz_mastery = []
    radar_indicators = []
    for row in quiz_rows.all():
        kp_id, total, correct = row.kp_id, row.total, row.correct or 0
        accuracy = round(correct / total * 100) if total > 0 else 0

        # 查询该知识点的平均答题时长
        dur_row = await db.execute(
            sa.select(
                sa.func.avg(LearningRecord.duration_seconds).label("avg_sec")
            ).where(
                LearningRecord.user_id == user_id,
                LearningRecord.kp_id == kp_id,
                LearningRecord.action == "quiz",
                LearningRecord.duration_seconds.isnot(None),
                LearningRecord.duration_seconds > 0,
            )
        )
        avg_sec = float(dur_row.scalar_one_or_none() or 0)

        # 答题效率分：平均用时越短分越高（满分 100）
        # <=15s: 100, 30s: 80, 45s: 60, 60s: 40, >=90s: 0
        if avg_sec <= 0:
            time_score = 50  # 无时长数据时给中等分
        else:
            time_score = max(0, min(100, round(100 - (avg_sec - 15) * (100 / 75))))

        # 综合掌握度 = 正确率 * 0.7 + 答题效率 * 0.3
        mastery_score = round(accuracy * 0.7 + time_score * 0.3)

        # 查询知识点名称
        kp_node = await db.execute(
            sa.select(KGNode.name).where(KGNode.id == kp_id)
        )
        kp_name = kp_node.scalar_one_or_none() or kp_id
        quiz_mastery.append({
            "kp_id": kp_id,
            "kp_name": kp_name,
            "total": total,
            "correct": correct,
            "accuracy": accuracy,
            "avg_seconds": round(avg_sec, 1),
            "time_score": time_score,
            "mastery_score": mastery_score,
        })
        radar_indicators.append({
            "name": kp_name if len(str(kp_name)) <= 8 else str(kp_name)[:8] + "…",
            "value": mastery_score,
            "full_name": kp_name,
        })

    # ── 2. 学习行为分析 ──
    # 近 30 天日学习时长
    thirty_days_ago = now - timedelta(days=30)
    lr_rows = await db.execute(
        sa.select(
            sa.func.date(LearningRecord.recorded_at).label("day"),
            sa.func.sum(LearningRecord.duration_seconds).label("seconds"),
            sa.func.count().label("actions"),
        ).where(
            LearningRecord.user_id == user_id,
            LearningRecord.recorded_at >= thirty_days_ago,
        ).group_by(sa.func.date(LearningRecord.recorded_at))
        .order_by(sa.func.date(LearningRecord.recorded_at))
    )

    # 单独查询每日“学习次数”：quiz 按分钟级去重（一次答题练习算一次），view/complete 正常计数
    # quiz 次数：同一分钟内的多道题只算一次答题练习
    quiz_daily = await db.execute(
        sa.select(
            sa.func.date(LearningRecord.recorded_at).label("day"),
            sa.func.count(sa.distinct(sa.func.date_trunc('minute', LearningRecord.recorded_at))).label("cnt"),
        ).where(
            LearningRecord.user_id == user_id,
            LearningRecord.recorded_at >= thirty_days_ago,
            LearningRecord.action == 'quiz',
        ).group_by(sa.func.date(LearningRecord.recorded_at))
    )
    quiz_count_by_day = {str(r.day): r.cnt for r in quiz_daily.all()}

    # view 次数：每次预览算一次
    view_daily = await db.execute(
        sa.select(
            sa.func.date(LearningRecord.recorded_at).label("day"),
            sa.func.count().label("cnt"),
        ).where(
            LearningRecord.user_id == user_id,
            LearningRecord.recorded_at >= thirty_days_ago,
            LearningRecord.action == 'view',
            LearningRecord.duration_seconds.isnot(None),
            LearningRecord.duration_seconds > 0,
        ).group_by(sa.func.date(LearningRecord.recorded_at))
    )
    view_count_by_day = {str(r.day): r.cnt for r in view_daily.all()}

    # complete 次数
    complete_daily = await db.execute(
        sa.select(
            sa.func.date(LearningRecord.recorded_at).label("day"),
            sa.func.count().label("cnt"),
        ).where(
            LearningRecord.user_id == user_id,
            LearningRecord.recorded_at >= thirty_days_ago,
            LearningRecord.action == 'complete',
        ).group_by(sa.func.date(LearningRecord.recorded_at))
    )
    complete_count_by_day = {str(r.day): r.cnt for r in complete_daily.all()}

    daily_data = []
    total_seconds = 0
    total_actions = 0
    active_days = 0
    for row in lr_rows.all():
        day_str = str(row.day)
        seconds = row.seconds or 0
        # 学习次数 = quiz次数 + view次数 + complete次数
        actions = (quiz_count_by_day.get(day_str, 0) +
                   view_count_by_day.get(day_str, 0) +
                   complete_count_by_day.get(day_str, 0))
        daily_data.append({
            "date": day_str,
            "minutes": round(seconds / 60, 1),
            "actions": actions,
        })
        total_seconds += seconds
        total_actions += actions
        active_days += 1

    # 连续学习天数（streak）
    streak = 0
    check_date = now.date()
    day_set = {item["date"] for item in daily_data}
    while str(check_date) in day_set:
        streak += 1
        check_date -= timedelta(days=1)

    learning_behavior = {
        "total_minutes": round(total_seconds / 60, 1),
        "total_actions": total_actions,
        "active_days": active_days,
        "streak_days": streak,
        "daily_trend": daily_data,
    }

    # ── 3. 遗忘曲线提醒 ──
    # 对每个知识点，取最后学习时间，计算距今天数
    # 基于艾宾浩斯遗忘曲线: 1天、2天、4天、7天、15天、30天
    REVIEW_INTERVALS = [1, 2, 4, 7, 15, 30]
    last_study_rows = await db.execute(
        sa.select(
            LearningRecord.kp_id,
            sa.func.max(LearningRecord.recorded_at).label("last_at"),
            sa.func.count().label("study_count"),
        ).where(
            LearningRecord.user_id == user_id,
            LearningRecord.kp_id.isnot(None),
        ).group_by(LearningRecord.kp_id)
    )
    forgetting_items = []
    for row in last_study_rows.all():
        kp_id = row.kp_id
        last_at = row.last_at
        study_count = row.study_count or 0
        if not last_at:
            continue
        days_since = (now - last_at).days
        # 判断是否需要复习
        needs_review = False
        next_review_day = None
        for interval in REVIEW_INTERVALS:
            if days_since >= interval:
                needs_review = True
                next_review_day = interval
        # 查 kp name
        kp_node = await db.execute(
            sa.select(KGNode.name).where(KGNode.id == kp_id)
        )
        kp_name = kp_node.scalar_one_or_none() or kp_id
        urgency = "high" if days_since >= 7 else ("medium" if days_since >= 3 else "low")
        forgetting_items.append({
            "kp_id": kp_id,
            "kp_name": kp_name,
            "days_since_last": days_since,
            "study_count": study_count,
            "needs_review": needs_review,
            "urgency": urgency,
        })
    # 按紧迫度排序
    urgency_order = {"high": 0, "medium": 1, "low": 2}
    forgetting_items.sort(key=lambda x: (urgency_order.get(x["urgency"], 3), -x["days_since_last"]))

    # ── 4. 能力雷达图数据 ──
    # 取掌握度最高的 8 个知识点作为雷达维度
    radar_sorted = sorted(radar_indicators, key=lambda x: x["value"], reverse=True)[:8]

    # ── 5. 学习行为分类统计 ──
    # 排除 stay（页面停留仅用于每日学习时长，不纳入行为统计）
    # view 只统计有 resource_id 且 duration > 0 的有效预览
    # quiz 次数按分钟级去重（一次答题练习算一次）
    action_rows = await db.execute(
        sa.select(
            LearningRecord.action,
            sa.func.count().label("count"),
            sa.func.coalesce(sa.func.sum(LearningRecord.duration_seconds), 0).label("total_sec"),
        ).where(
            LearningRecord.user_id == user_id,
            LearningRecord.action != 'stay',
            sa.or_(
                LearningRecord.action != 'view',
                sa.and_(
                    LearningRecord.resource_id.isnot(None),
                    LearningRecord.duration_seconds.isnot(None),
                    LearningRecord.duration_seconds > 0,
                )
            )
        ).group_by(LearningRecord.action)
    )

    # quiz 单独查询“练习次数”：同一分钟内的多道题只算一次
    quiz_session_count = await db.execute(
        sa.select(
            sa.func.count(sa.distinct(sa.func.date_trunc('minute', LearningRecord.recorded_at))).label("cnt"),
        ).where(
            LearningRecord.user_id == user_id,
            LearningRecord.action == 'quiz',
        )
    )
    quiz_sessions = quiz_session_count.scalar_one_or_none() or 0

    behavior_breakdown = []
    for row in action_rows.all():
        action_label = {"view": "浏览资源", "quiz": "答题练习", "complete": "完成学习", "stay": "页面停留"}.get(row.action, row.action)
        # quiz 用分钟级去重的次数
        count = quiz_sessions if row.action == 'quiz' else row.count
        behavior_breakdown.append({
            "action": row.action,
            "label": action_label,
            "count": count,
            "total_minutes": round(float(row.total_sec or 0) / 60, 1),
        })

    # ── 6. 资源使用情况（按实际使用次数统计，而非生成数量） ──
    res_rows = await db.execute(
        sa.select(
            ResourceMeta.resource_type,
            sa.func.count(LearningRecord.id).label("use_count"),
            sa.func.count(sa.distinct(LearningRecord.resource_id)).label("res_count"),
            sa.func.coalesce(sa.func.sum(LearningRecord.duration_seconds), 0).label("total_sec"),
        ).join(
            LearningRecord, LearningRecord.resource_id == ResourceMeta.id
        ).where(
            LearningRecord.user_id == user_id,
        ).group_by(ResourceMeta.resource_type)
    )
    resource_usage = []
    type_labels = {"doc": "文档讲义", "mindmap": "思维导图", "quiz": "测验题", "code": "代码示例", "summary": "知识总结", "animation": "动画演示"}
    for row in res_rows.all():
        rt = row.resource_type.value if hasattr(row.resource_type, 'value') else row.resource_type
        resource_usage.append({
            "type": rt,
            "label": type_labels.get(rt, rt),
            "count": row.use_count,
            "res_count": row.res_count,
            "total_minutes": round(float(row.total_sec or 0) / 60, 1),
        })

    # ── 7. 最近学习记录（最新 10 条，含详情） ──
    # 排除 stay（仅用于每日时长）
    # view 必须有 resource_id 且有 duration_seconds（过滤旧的不完整记录）
    recent_rows = await db.execute(
        sa.select(
            LearningRecord,
            ResourceMeta.title.label("res_title"),
            ResourceMeta.resource_type.label("res_type"),
            KGNode.name.label("kp_name"),
        )
        .outerjoin(ResourceMeta, LearningRecord.resource_id == ResourceMeta.id)
        .outerjoin(KGNode, LearningRecord.kp_id == KGNode.id)
        .where(
            LearningRecord.user_id == user_id,
            LearningRecord.action != 'stay',
            sa.or_(
                LearningRecord.action != 'view',
                sa.and_(
                    LearningRecord.resource_id.isnot(None),
                    LearningRecord.duration_seconds.isnot(None),
                    LearningRecord.duration_seconds > 0,
                )
            )
        )
        .order_by(LearningRecord.recorded_at.desc())
        .limit(10)
    )
    recent_activities = []
    for row in recent_rows.all():
        r = row[0]  # LearningRecord object
        res_title = row.res_title
        res_type = row.res_type
        kp_name = row.kp_name
        action_label = {"view": "浏览资源", "quiz": "答题练习", "complete": "完成学习", "stay": "页面停留"}.get(r.action, r.action)
        rt_val = res_type.value if hasattr(res_type, 'value') else res_type if res_type else None
        recent_activities.append({
            "action": r.action,
            "label": action_label,
            "kp_id": r.kp_id,
            "kp_name": kp_name,
            "resource_id": r.resource_id,
            "resource_title": res_title,
            "resource_type": rt_val,
            "duration_seconds": r.duration_seconds,
            "recorded_at": r.recorded_at.isoformat() if r.recorded_at else None,
        })

    return {
        "quiz_mastery": quiz_mastery,
        "learning_behavior": learning_behavior,
        "forgetting_curve": forgetting_items,
        "radar_data": {
            "indicators": [{"name": r["name"], "max": 100} for r in radar_sorted],
            "values": [r["value"] for r in radar_sorted],
            "full_names": [r["full_name"] for r in radar_sorted],
        },
        "behavior_breakdown": behavior_breakdown,
        "resource_usage": resource_usage,
        "recent_activities": recent_activities,
    }

"""
backend/middleware/logging_middleware.py
请求日志中间件：为每个请求注入 trace_id，记录请求入口、耗时、响应状态。
trace_id 通过 ContextVar 传递，并由 logging_config.py 的 patcher 自动注入日志 extra。
"""

from __future__ import annotations

import time
import uuid
from contextvars import ContextVar
from typing import Callable

from backend.config import config
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# 每个请求的 trace_id 通过 contextvar 传递，方便在任意位置获取
trace_id_var: ContextVar[str] = ContextVar("trace_id", default="-")

# 健康检查路径（不记录日志，避免刷屏）
_HEALTH_CHECK_PATHS: set[str] = {"/health", "/healthz", "/ready", "/livez"}


def generate_trace_id() -> str:
    """生成短 trace_id，供后台任务等非 HTTP 上下文使用。"""
    return str(uuid.uuid4())[:config.logging.trace_id_length]


def get_trace_id() -> str:
    """获取当前请求的 trace_id（未在请求上下文中时返回 '-'）。"""
    return trace_id_var.get()


class LoggingMiddleware(BaseHTTPMiddleware):
    """FastAPI 中间件：记录请求入口、trace_id、HTTP 方法、路径、耗时、状态码。"""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # 健康检查端点跳过日志
        if request.url.path in _HEALTH_CHECK_PATHS:
            return await call_next(request)

        trace_id = request.headers.get("X-Trace-ID") or generate_trace_id()
        trace_id_var.set(trace_id)
        request.state.trace_id = trace_id

        from backend.logging_config import logger

        logger.info(
            "--> {} {} | query: {}",
            request.method, request.url.path, request.query_params,
        )

        start_time = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception as e:
            elapsed = (time.perf_counter() - start_time) * 1000
            logger.error(
                "<-- {} {} | EXCEPTION: {}: {} | {:.1f}ms",
                request.method, request.url.path, type(e).__name__, e, elapsed,
            )
            raise

        elapsed = (time.perf_counter() - start_time) * 1000
        logger.info(
            "<-- {} {} | status: {} | {:.1f}ms",
            request.method, request.url.path, response.status_code, elapsed,
        )
        return response

"""
backend/logging_config.py
统一日志配置：Loguru + 控制台输出 + 文件轮转 + 错误日志分离 + JSON 结构
化输出。
在应用启动前 import 此模块即可生效。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from loguru import logger

from backend.config import config

# 日志目录
LOG_DIR = Path(__file__).parent.parent / config.logging.dir
LOG_DIR.mkdir(exist_ok=True)

# 移除默认 handler
logger.remove()

# ----------------------------------------------------------
# trace_id 注入 patcher — 从 ContextVar 自动读取并注入 extra
# ----------------------------------------------------------

def _inject_trace_id(record: dict) -> None:
    """将 ContextVar 中的 trace_id 注入每条日志的 extra 字段。"""
    try:
        from backend.middleware.logging_middleware import trace_id_var
        record["extra"]["trace_id"] = trace_id_var.get()
    except Exception:
        record["extra"]["trace_id"] = "-"

logger.configure(patcher=_inject_trace_id)

# ----------------------------------------------------------
# 格式定义
# ----------------------------------------------------------

_CONSOLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<yellow>{extra[trace_id]}</yellow> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)

_FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {extra[trace_id]} | "
    "{name}:{function}:{line} - {message}"
)


def _json_formatter(record: dict) -> str:
    """将 loguru record 序列化为 JSON 行（.jsonl 格式）。"""
    log_entry = {
        "timestamp": record["time"].isoformat(),
        "level": record["level"].name,
        "trace_id": record["extra"].get("trace_id", "-"),
        "logger": record["name"],
        "function": record["function"],
        "line": record["line"],
        "message": record["message"],
    }
    if record["exception"]:
        log_entry["exception"] = {
            "type": record["exception"].type.__name__ if record["exception"].type else None,
            "value": str(record["exception"].value),
            "traceback": record["exception"].traceback,
        }
    return json.dumps(log_entry, ensure_ascii=False, default=str) + "\n"


# ----------------------------------------------------------
# Sink 注册
# ----------------------------------------------------------

# 控制台输出（带颜色）
logger.add(
    sys.stdout,
    format=_CONSOLE_FORMAT,
    level=config.logging.console_level.upper(),
    colorize=True,
)

# 普通日志文件（每天轮转）
logger.add(
    LOG_DIR / "app_{time:YYYY-MM-DD}.log",
    rotation="00:00",
    retention=f"{config.logging.retention_days} days",
    compression="zip",
    format=_FILE_FORMAT,
    level=config.logging.level.upper(),
    enqueue=True,
)

# 错误日志单独记录
logger.add(
    LOG_DIR / "error_{time:YYYY-MM-DD}.log",
    rotation="00:00",
    retention=f"{config.logging.error_retention_days} days",
    compression="zip",
    format=_FILE_FORMAT,
    level="ERROR",
    enqueue=True,
)

# JSON 结构化日志（仅在 json_format=true 时启用）
if config.logging.json_format:
    logger.add(
        LOG_DIR / "app_{time:YYYY-MM-DD}.jsonl",
        format=_json_formatter,
        level=config.logging.level.upper(),
        rotation="00:00",
        retention=f"{config.logging.retention_days} days",
        compression="zip",
        enqueue=True,
    )

"""Token 生成/验证、过期时间计算等辅助函数。"""

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta

from backend.config import config


def generate_token() -> str:
    """生成 URL 安全的随机 token（48 字节 = 384 位熵值）。"""
    return secrets.token_urlsafe(48)


def hash_token(token: str) -> str:
    """SHA-256 哈希 token，数据库存储哈希值防泄露。"""
    return hashlib.sha256(token.encode()).hexdigest()


def expires_at(purpose: str) -> datetime:
    """根据 purpose 返回过期时间。"""
    if purpose == "email_verify":
        minutes = config.email.verification_expire_minutes
    elif purpose == "password_reset":
        minutes = config.email.password_reset_expire_minutes
    else:
        minutes = 30
    return datetime.utcnow() + timedelta(minutes=minutes)

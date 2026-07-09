"""
backend/utils/snowflake.py
Snowflake ID 生成器，用于生成全局唯一的 BIGINT 主键。
"""
from __future__ import annotations

import hashlib
import os
import random
import threading
import random
import time

# 自定义起始时间戳 (2024-01-01 00:00:00 UTC in milliseconds)
_EPOCH = 1704067200000

# 机器 ID（0-31），从环境变量或随机生成
_worker_id = int(os.environ.get("SNOWFLAKE_WORKER_ID", str(random.randint(0, 31))))
# 数据中心 ID（0-31），从环境变量或随机生成
_datacenter_id = int(os.environ.get("SNOWFLAKE_DATACENTER_ID", str(random.randint(0, 31))))

_sequence = 0
_last_timestamp = 0
_lock = threading.Lock()


def generate_id() -> int:
    """
    生成一个 Snowflake ID（BIGINT）。

    结构 (64 bits):
    - 41 bits: 毫秒时间戳 (相对 _EPOCH)
    -  5 bits: 数据中心 ID
    -  5 bits: 机器 ID
    - 12 bits: 序列号（每毫秒从 0 递增）
    """
    global _sequence, _last_timestamp

    with _lock:
        timestamp = int(time.time() * 1000)
        if timestamp == _last_timestamp:
            _sequence = (_sequence + 1) & 0xFFF
            if _sequence == 0:
                # 序列号用完，等待到下一毫秒
                while timestamp <= _last_timestamp:
                    timestamp = int(time.time() * 1000)
        else:
            _sequence = 0
        _last_timestamp = timestamp

    return int(
        ((timestamp - _EPOCH) << 22)
        | (_datacenter_id << 17)
        | (_worker_id << 12)
        | _sequence
    )


def string_to_id(s: str) -> int:
    """
    从字符串生成确定性 BIGINT（替代 uuid.uuid5）。
    对使用场景（从 resource_id-idx 生成题目 ID）足够唯一。
    """
    # 在末尾加盐防止 hash 碰撞
    h = hashlib.md5((s + "_softbei_salt").encode()).hexdigest()
    # 取低 63 位，确保为正
    return int(h[8:24], 16) & 0x7FFFFFFFFFFFFFFF

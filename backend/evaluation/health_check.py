"""
backend/evaluation/health_check.py
Layer 1 健康检查：每次 RAG 请求后采集轻量级指标，不调用 LLM，<5ms 开销。

采集指标：
- 检索是否为空结果
- chunk 分数的 min/p50/max
- 各阶段真实耗时（embedding / DB query 分别计时）
- 生成长度

当指标超出阈值时输出 WARNING 日志，可选接入外部告警。
"""

from __future__ import annotations

import time
from typing import Optional

from loguru import logger

from backend.config import config
from backend.evaluation.models import HealthCheckRecord


# ===========================================================
# 阈值配置
# ===========================================================

# 以下阈值触发 WARNING 日志（不影响请求处理）
_THRESHOLDS = {
    "empty_result_rate": 0.3,          # 最近 100 次中空结果率 >30%
    "score_p50_min": 0.3,              # 中位数分数 <0.3
    "retrieval_latency_p95_ms": 5000,  # 检索 P95 >5s
    "generation_latency_p95_ms": 30000,  # 生成 P95 >30s
}

# 滑动窗口大小（内存中保留最近 N 条记录用于计算比率）
_WINDOW_SIZE = 100


# ===========================================================
# 健康检查采集器
# ===========================================================

class HealthChecker:
    """每次请求的轻量级健康检查（模块级单例）。"""

    def __init__(self):
        self._records: list[HealthCheckRecord] = []
        self._enabled: bool = True

    @property
    def enabled(self) -> bool:
        return self._enabled and config.evaluation.health_check_enabled

    # ----------------------------------------------------------
    # 采集
    # ----------------------------------------------------------

    def record(
        self,
        agent_type: str = "",
        kp_name: str = "",
        n_retrieved: int = 0,
        scores: list[float] | None = None,
        embedding_latency_ms: float = 0.0,
        db_query_latency_ms: float = 0.0,
        total_retrieval_ms: float = 0.0,
        draft_length: int = 0,
        generation_latency_ms: float = 0.0,
    ) -> Optional[HealthCheckRecord]:
        """
        记录一次请求的健康指标。若未启用健康检查则跳过。

        :return: HealthCheckRecord 或 None（未启用时）
        """
        if not self.enabled:
            return None

        _scores = scores or []
        record = HealthCheckRecord(
            agent_type=agent_type,
            kp_name=kp_name,
            n_retrieved=n_retrieved,
            n_empty_results=1 if n_retrieved == 0 else 0,
            score_p50=_percentile(_scores, 50) if _scores else 0.0,
            score_min=min(_scores) if _scores else 0.0,
            score_max=max(_scores) if _scores else 0.0,
            embedding_latency_ms=embedding_latency_ms,
            db_query_latency_ms=db_query_latency_ms,
            total_retrieval_ms=total_retrieval_ms,
            draft_length=draft_length,
            generation_latency_ms=generation_latency_ms,
        )

        self._records.append(record)
        # 保持窗口大小
        if len(self._records) > _WINDOW_SIZE:
            self._records = self._records[-_WINDOW_SIZE:]

        # 记录后立即检查阈值
        self._check_thresholds()

        return record

    # ----------------------------------------------------------
    # 阈值检查
    # ----------------------------------------------------------

    def _check_thresholds(self) -> None:
        """检查最近窗口内的指标是否超出阈值，超出则 WARNING。"""
        if len(self._records) < 10:
            return  # 样本太少，不检查

        window = self._records[-_WINDOW_SIZE:]

        # 空结果率
        empty_count = sum(1 for r in window if r.n_empty_results)
        empty_rate = empty_count / len(window)
        if empty_rate > _THRESHOLDS["empty_result_rate"]:
            logger.warning(
                f"[HealthCheck] 空结果率偏高: {empty_rate:.1%} "
                f"(>{_THRESHOLDS['empty_result_rate']:.0%}), "
                f"最近 {len(window)} 次中 {empty_count} 次返回空"
            )

        # 分数中位数
        p50_scores = [r.score_p50 for r in window if r.score_p50 > 0]
        if p50_scores:
            avg_p50 = sum(p50_scores) / len(p50_scores)
            if avg_p50 < _THRESHOLDS["score_p50_min"]:
                logger.warning(
                    f"[HealthCheck] 检索分数偏低: avg P50={avg_p50:.3f} "
                    f"(<{_THRESHOLDS['score_p50_min']}), 可能存在检索质量退化"
                )

        # 检索延迟 P95
        retrieval_lats = sorted(
            [r.total_retrieval_ms for r in window if r.total_retrieval_ms > 0]
        )
        if retrieval_lats:
            p95_lat = retrieval_lats[int(len(retrieval_lats) * 0.95)]
            if p95_lat > _THRESHOLDS["retrieval_latency_p95_ms"]:
                logger.warning(
                    f"[HealthCheck] 检索延迟偏高: P95={p95_lat:.0f}ms "
                    f"(>{_THRESHOLDS['retrieval_latency_p95_ms']}ms)"
                )

    # ----------------------------------------------------------
    # 查询
    # ----------------------------------------------------------

    def get_recent(self, n: int = 20) -> list[HealthCheckRecord]:
        """获取最近 N 条健康检查记录。"""
        return self._records[-n:]

    def get_summary(self) -> dict:
        """获取当前窗口的汇总指标。"""
        if not self._records:
            return {"status": "no_data", "sample_count": 0}

        window = self._records[-_WINDOW_SIZE:]
        scores_all = [r.score_p50 for r in window if r.score_p50 > 0]
        lats = [r.total_retrieval_ms for r in window if r.total_retrieval_ms > 0]

        return {
            "status": "ok",
            "sample_count": len(window),
            "empty_result_rate": sum(1 for r in window if r.n_empty_results) / len(window),
            "avg_score_p50": sum(scores_all) / len(scores_all) if scores_all else 0.0,
            "p50_retrieval_ms": _percentile(lats, 50) if lats else 0.0,
            "p95_retrieval_ms": _percentile(lats, 95) if lats else 0.0,
        }

    def clear(self) -> None:
        """清空历史记录。"""
        self._records.clear()


# ===========================================================
# 工具函数
# ===========================================================

def _percentile(data: list[float], p: int) -> float:
    """线性插值百分位数计算（不依赖 numpy）。"""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    n = len(sorted_data)
    if n == 1:
        return sorted_data[0]
    k = (p / 100) * (n - 1)
    f = int(k)
    c = k - f
    if f + 1 >= n:
        return sorted_data[-1]
    return sorted_data[f] * (1 - c) + sorted_data[f + 1] * c


# 模块级单例
health_checker = HealthChecker()

"""RAG 评估数据采集器：在 RAG 管线中匿名埋点，采集检索和生成的元数据。"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Optional

from loguru import logger

from backend.config import config
from backend.evaluation.models import RetrievalEvalRecord, GenerationEvalRecord


class RAGEvalCollector:
    """RAG 评估数据采集器（模块级单例）。

    采样率由 config.evaluation.mode 决定：
    - development → evaluation.sampling.development（默认 1.0）
    - production  → evaluation.sampling.production（默认 0.1）

    用法::

        from backend.evaluation.collector import collector

        collector.start_query(query="梯度下降", kp_name="梯度下降", user_id="123", session_id="456")
        collector.record_retrieval(
            scores=[0.87, 0.65, 0.52],
            chunk_ids=["doc_a_0", "doc_a_1"],
            chunk_texts=["梯度下降是一种...", "学习率控制..."],
            doc_ids=["doc_a", "doc_a"],
            embedding_latency_ms=120.0,
            db_query_latency_ms=45.0,
        )
        collector.record_generation(
            agent_type="doc_agent",
            draft_length=1500,
            generation_latency_ms=2500.0,
            safety_passed=True,
            safety_issues_count=0,
            experiment_group="baseline",
        )
        record = collector.flush()
    """

    def __init__(self):
        self._current_retrieval: Optional[RetrievalEvalRecord] = None
        self._current_generation: Optional[GenerationEvalRecord] = None
        self._retrieval_timer: Optional[float] = None
        self._records: list[GenerationEvalRecord] = []
        self._experiment_group: Optional[str] = None

    @property
    def sample_rate(self) -> float:
        """从 config 读取当前模式的采样率。"""
        mode = config.evaluation.mode
        if mode == "development":
            return config.evaluation.sampling.development
        return config.evaluation.sampling.production

    def start_query(
        self,
        query: str,
        kp_name: str = "",
        user_id: str = "",
        session_id: str = "",
    ) -> None:
        """在检索开始时调用，初始化采集状态。"""
        self._current_retrieval = RetrievalEvalRecord(
            query=query,
            kp_name=kp_name,
            user_id=user_id,
            session_id=session_id,
            timestamp=datetime.utcnow(),
        )
        self._current_generation = None
        self._retrieval_timer = time.perf_counter()

    def record_retrieval(
        self,
        scores: list[float],
        chunk_ids: list[str],
        chunk_texts: list[str] | None = None,
        doc_ids: list[str] | None = None,
        embedding_latency_ms: float = 0.0,
        db_query_latency_ms: float = 0.0,
    ) -> None:
        """检索完成后调用，填写分数、延迟和 chunk 信息。"""
        if self._current_retrieval is None:
            return
        rec = self._current_retrieval
        rec.scores = scores
        rec.chunk_ids = chunk_ids
        rec.chunk_texts = chunk_texts or []
        rec.doc_ids = doc_ids or []
        rec.n_results = len(scores)
        rec.embedding_latency_ms = embedding_latency_ms
        rec.db_query_latency_ms = db_query_latency_ms
        rec.n_candidates = max(len(scores), 1)  # 若未设 prefetch 则等于 n_results

    def record_generation(
        self,
        agent_type: str = "",
        draft_length: int = 0,
        generation_latency_ms: float = 0.0,
        safety_passed: bool = True,
        safety_issues_count: int = 0,
        safety_issues: list[str] | None = None,
        experiment_group: Optional[str] = None,
    ) -> None:
        """Agent 生成完成后调用。"""
        total_ms = (
            (time.perf_counter() - self._retrieval_timer) * 1000
            if self._retrieval_timer
            else 0.0
        )
        self._experiment_group = experiment_group
        self._current_generation = GenerationEvalRecord(
            session_id=self._current_retrieval.session_id if self._current_retrieval else "",
            user_id=self._current_retrieval.user_id if self._current_retrieval else "",
            agent_type=agent_type,
            kp_name=self._current_retrieval.kp_name if self._current_retrieval else "",
            query=self._current_retrieval.query if self._current_retrieval else "",
            timestamp=datetime.utcnow(),
            draft_length=draft_length,
            generation_latency_ms=generation_latency_ms,
            has_rag_context=(
                self._current_retrieval is not None
                and len(self._current_retrieval.chunk_ids) > 0
            ),
            n_retrieved=(
                self._current_retrieval.n_results if self._current_retrieval else 0
            ),
            safety_passed=safety_passed,
            safety_issues_count=safety_issues_count,
            safety_issues=safety_issues or [],
            retrieval_record=self._current_retrieval,
        )

    def flush(self) -> Optional[GenerationEvalRecord]:
        """结束当前查询，返回完整的 GenerationEvalRecord，清空内部状态。"""
        record = self._current_generation
        if record is not None:
            self._records.append(record)
            logger.info(
                f"[Collector] flush: agent={record.agent_type}, kp={record.kp_name}, "
                f"n_retrieved={record.n_retrieved}, draft_len={record.draft_length}, "
                f"safety_passed={record.safety_passed}, safety_issues={len(record.safety_issues)}"
            )

            # 健康检查采集（Layer 1）
            self._record_health_check(record)

        self._current_retrieval = None
        self._current_generation = None
        self._retrieval_timer = None
        self._experiment_group = None
        return record

    def decide_sample(self, session_id: str = "") -> bool:
        """
        按 session_id 哈希决定是否对本次请求执行 LLM-as-Judge 评估。

        使用 session_id 的 hash 值取模，确保同一 session 的所有轮次
        要么全评估，要么全不评估。
        """
        rate = self.sample_rate
        if rate >= 1.0:
            return True
        if rate <= 0.0:
            return False

        if not session_id:
            import random
            return random.random() < rate

        seed = hash(session_id) % 100
        return (seed / 100.0) < rate

    def _record_health_check(self, record: GenerationEvalRecord) -> None:
        """将 flush 的记录同步写入健康检查器。"""
        try:
            from backend.evaluation.health_check import health_checker
            retrieval = record.retrieval_record
            health_checker.record(
                agent_type=record.agent_type,
                kp_name=record.kp_name,
                n_retrieved=record.n_retrieved,
                scores=retrieval.scores if retrieval else None,
                embedding_latency_ms=retrieval.embedding_latency_ms if retrieval else 0.0,
                db_query_latency_ms=retrieval.db_query_latency_ms if retrieval else 0.0,
                total_retrieval_ms=(
                    (retrieval.embedding_latency_ms + retrieval.db_query_latency_ms)
                    if retrieval else 0.0
                ),
                draft_length=record.draft_length,
                generation_latency_ms=record.generation_latency_ms,
            )
        except Exception:
            pass  # 健康检查异常不影响主流程

    def get_recent_records(self, n: int = 20) -> list[GenerationEvalRecord]:
        """获取最近 N 条采集记录。"""
        return self._records[-n:]

    def get_records_since(self, since: datetime) -> list[GenerationEvalRecord]:
        """获取指定时间之后的所有记录。"""
        return [r for r in self._records if r.timestamp >= since]

    def clear_records(self) -> None:
        """清空所有历史记录。"""
        self._records.clear()


# 模块级单例（全局共享，采样率由 config.evaluation.mode 自动决定）
collector = RAGEvalCollector()

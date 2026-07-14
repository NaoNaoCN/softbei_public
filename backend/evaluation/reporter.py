"""RAG 评估报告生成器：聚合采集数据，生成日报/周报/Markdown 报告。"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from backend.evaluation.models import (
    RetrievalEvalRecord,
    GenerationEvalRecord,
    RAGEvalReport,
)
from backend.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    mrr,
    ndcg_at_k,
    hit_rate,
    score_distribution,
)


class RAGReporter:
    """RAG 评估报告生成器。

    用法::

        reporter = RAGReporter()
        report = reporter.generate_report(records)
        print(reporter.to_markdown(report))
    """

    def __init__(self, output_dir: str | None = None):
        self._last_report: Optional[RAGEvalReport] = None
        if output_dir is None:
            from pathlib import Path as _Path
            from backend.config import config as _cfg
            output_dir = str(_Path(__file__).parent.parent.parent / _cfg.logging.dir)
        self._output_dir = output_dir

    def generate_report(
        self,
        records: list[GenerationEvalRecord],
        period_start: datetime | None = None,
        period_end: datetime | None = None,
    ) -> RAGEvalReport:
        """
        从采集记录生成评估报告。

        :param records:      GenerationEvalRecord 列表
        :param period_start: 报告周期开始时间
        :param period_end:   报告周期结束时间
        :return:             RAGEvalReport
        """
        if not records:
            return RAGEvalReport(
                period_start=period_start or datetime.utcnow(),
                period_end=period_end or datetime.utcnow(),
                total_queries=0,
            )

        timestamps = [r.timestamp for r in records if r.timestamp]
        start = period_start or (min(timestamps) if timestamps else datetime.utcnow())
        end = period_end or (max(timestamps) if timestamps else datetime.utcnow())

        all_relevance_labels = [
            r.relevance_labels for r in records if r.relevance_labels
        ]

        k = 5
        precision_scores = [precision_at_k(labels, k) for labels in all_relevance_labels if labels]
        recall_scores = []
        for labels in all_relevance_labels:
            if labels:
                total_rel = sum(1 for v in labels if v > 0)
                recall_scores.append(recall_at_k(labels, total_rel, k) if total_rel > 0 else 0.0)

        all_retrieval_scores: list[float] = []
        retrieval_latencies: list[float] = []
        for r in records:
            if r.retrieval_record and r.retrieval_record.scores:
                all_retrieval_scores.extend(r.retrieval_record.scores)
                total_latency = r.retrieval_record.embedding_latency_ms + r.retrieval_record.db_query_latency_ms
                if total_latency > 0:
                    retrieval_latencies.append(total_latency)

        score_dist = score_distribution(all_retrieval_scores)

        faithfulness_scores = [
            r.faithfulness_score for r in records
            if r.faithfulness_score is not None
        ]
        hallucination_rates = [
            r.hallucination_rate_val or 0.0 for r in records
        ]
        concept_coverages = [
            r.concept_coverage for r in records
            if r.concept_coverage is not None
        ]
        completeness_scores = [
            r.completeness_score for r in records
            if r.completeness_score is not None
        ]

        gen_latencies = [
            r.generation_latency_ms for r in records
            if r.generation_latency_ms > 0
        ]
        ret_latencies_sorted = sorted(retrieval_latencies) if retrieval_latencies else []
        gen_latencies_sorted = sorted(gen_latencies) if gen_latencies else []

        def _p50(vals: list[float]) -> float:
            if not vals:
                return 0.0
            return vals[len(vals) // 2]

        def _p95(vals: list[float]) -> float:
            if not vals:
                return 0.0
            return vals[int(len(vals) * 0.95)]

        delta: dict[str, float] = {}
        if self._last_report is not None:
            prev = self._last_report
            if prev.avg_faithfulness > 0:
                cur_faith = (
                    sum(faithfulness_scores) / len(faithfulness_scores)
                    if faithfulness_scores else 0.0
                )
                delta["faithfulness"] = round(cur_faith - prev.avg_faithfulness, 4)
            if prev.p50_retrieval_latency_ms > 0:
                cur_lat = _p50(ret_latencies_sorted)
                delta["p50_retrieval_latency_ms"] = round(cur_lat - prev.p50_retrieval_latency_ms, 1)

        report = RAGEvalReport(
            period_start=start,
            period_end=end,
            total_queries=len(records),
            precision_at_5=round(sum(precision_scores) / len(precision_scores), 4) if precision_scores else 0.0,
            recall_at_5=round(sum(recall_scores) / len(recall_scores), 4) if recall_scores else 0.0,
            mrr_val=round(mrr(all_relevance_labels), 4) if all_relevance_labels else 0.0,
            ndcg_at_5=round(sum(ndcg_at_k(labels, k) for labels in all_relevance_labels) / len(all_relevance_labels), 4) if all_relevance_labels else 0.0,
            hit_rate_val=round(hit_rate(all_relevance_labels, k), 4) if all_relevance_labels else 0.0,
            score_p50=round(score_dist.get("p50", 0.0), 4),
            avg_faithfulness=round(sum(faithfulness_scores) / len(faithfulness_scores), 4) if faithfulness_scores else 0.0,
            avg_hallucination_rate=round(sum(hallucination_rates) / len(hallucination_rates), 4) if hallucination_rates else 0.0,
            avg_concept_coverage=round(sum(concept_coverages) / len(concept_coverages), 4) if concept_coverages else 0.0,
            p50_retrieval_latency_ms=round(_p50(ret_latencies_sorted), 1),
            p95_retrieval_latency_ms=round(_p95(ret_latencies_sorted), 1),
            p50_generation_latency_ms=round(_p50(gen_latencies_sorted), 1),
            delta_vs_previous=delta,
        )

        self._last_report = report
        return report

    def generate_daily_report(
        self,
        records: list[GenerationEvalRecord],
    ) -> RAGEvalReport:
        """生成日报：过去 24 小时的指标。"""
        now = datetime.utcnow()
        start = now - timedelta(hours=24)
        daily = [r for r in records if r.timestamp and r.timestamp >= start]
        return self.generate_report(daily, period_start=start, period_end=now)

    def generate_weekly_report(
        self,
        records: list[GenerationEvalRecord],
    ) -> RAGEvalReport:
        """生成周报：过去 7 天的指标。"""
        now = datetime.utcnow()
        start = now - timedelta(days=7)
        weekly = [r for r in records if r.timestamp and r.timestamp >= start]
        return self.generate_report(weekly, period_start=start, period_end=now)

    def to_markdown(self, report: RAGEvalReport) -> str:
        """将报告渲染为 Markdown 格式字符串。"""
        lines = [
            f"# RAG 评估报告",
            f"",
            f"**周期：** {report.period_start.strftime('%Y-%m-%d %H:%M')} → {report.period_end.strftime('%Y-%m-%d %H:%M')}",
            f"**总查询数：** {report.total_queries}",
            f"",
            f"## 检索质量",
            f"",
            f"| 指标 | 值 | 参考标准 | 达标 |",
            f"|------|-----|---------|------|",
            f"| Precision@5 | {report.precision_at_5:.3f} | > 0.60 | {_check_reference(report.precision_at_5, '> 0.60')} |",
            f"| Recall@5 | {report.recall_at_5:.3f} | > 0.70 | {_check_reference(report.recall_at_5, '> 0.70')} |",
            f"| MRR | {report.mrr_val:.3f} | > 0.50 | {_check_reference(report.mrr_val, '> 0.50')} |",
            f"| NDCG@5 | {report.ndcg_at_5:.3f} | > 0.60 | {_check_reference(report.ndcg_at_5, '> 0.60')} |",
            f"| Hit Rate@5 | {report.hit_rate_val:.3f} | > 0.80 | {_check_reference(report.hit_rate_val, '> 0.80')} |",
            f"| Score P50 | {report.score_p50:.3f} | > 0.65 | {_check_reference(report.score_p50, '> 0.65')} |",
            f"",
            f"## 生成质量",
            f"",
            f"| 指标 | 值 | 参考标准 | 达标 |",
            f"|------|-----|---------|------|",
            f"| Avg Faithfulness | {report.avg_faithfulness:.3f} | > 0.70 | {_check_reference(report.avg_faithfulness, '> 0.70')} |",
            f"| Avg Hallucination Rate | {report.avg_hallucination_rate:.3f} | < 0.30 | {_check_reference(report.avg_hallucination_rate, '< 0.30')} |",
            f"| Avg Concept Coverage | {report.avg_concept_coverage:.3f} | > 0.60 | {_check_reference(report.avg_concept_coverage, '> 0.60')} |",
            f"",
            f"## 系统效率",
            f"",
            f"| 指标 | 值 | 参考标准 | 达标 |",
            f"|------|-----|---------|------|",
            f"| P50 Retrieval Latency | {report.p50_retrieval_latency_ms:.0f} ms | < 500 ms | {_check_reference(report.p50_retrieval_latency_ms, '< 500')} |",
            f"| P95 Retrieval Latency | {report.p95_retrieval_latency_ms:.0f} ms | < 1000 ms | {_check_reference(report.p95_retrieval_latency_ms, '< 1000')} |",
            f"| P50 Generation Latency | {report.p50_generation_latency_ms:.0f} ms | < 5000 ms | {_check_reference(report.p50_generation_latency_ms, '< 5000')} |",
        ]

        if report.delta_vs_previous:
            lines.append("")
            lines.append("## 变化趋势（与上期对比）")
            lines.append("")
            lines.append("| 指标 | Delta |")
            lines.append("|------|-------|")
            for key, val in report.delta_vs_previous.items():
                sign = "+" if val >= 0 else ""
                lines.append(f"| {key} | {sign}{val:.4f} |")

        return "\n".join(lines)

    def to_summary(self, report: RAGEvalReport) -> str:
        """生成单行摘要，适合日志输出。"""
        return (
            f"[RAGReport] queries={report.total_queries} "
            f"P@5={report.precision_at_5:.3f} "
            f"Faith={report.avg_faithfulness:.3f} "
            f"Halluc={report.avg_hallucination_rate:.3f} "
            f"P50_ret={report.p50_retrieval_latency_ms:.0f}ms "
            f"P50_gen={report.p50_generation_latency_ms:.0f}ms"
        )

    def save_to_disk(
        self,
        report: RAGEvalReport,
        filename: str | None = None,
    ) -> str:
        """将报告写入日志目录，返回文件路径。同名文件会覆盖。"""
        from pathlib import Path
        from datetime import datetime

        out_dir = Path(self._output_dir)
        out_dir.mkdir(exist_ok=True)

        if filename is None:
            ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            filename = f"rag_eval_report_{ts}.md"

        filepath = out_dir / filename
        md_content = self.to_markdown(report)
        filepath.write_text(md_content, encoding="utf-8")
        return str(filepath)

    def save_daily_report(
        self,
        records: list[GenerationEvalRecord],
    ) -> str | None:
        """生成日报并写入 logs/ 目录。无记录时返回 None。"""
        if not records:
            return None

        report = self.generate_daily_report(records)
        if report.total_queries == 0:
            return None

        filepath = self.save_to_disk(report)
        from loguru import logger
        logger.info(
            "[RAGReporter] 日报已保存: {} ({} queries, Faith={:.3f})",
            filepath, report.total_queries, report.avg_faithfulness,
        )
        return filepath


def _check_reference(value: float, ref_str: str) -> str:
    """
    对比实际值与参考标准，返回达标状态指示符。

    支持格式:
      - ``> 0.60``  (值 >= 阈值 → 达标)
      - ``< 0.30``  (值 <= 阈值 → 达标)
      - ``-``       (无参考标准，返回空)

    :return: "✅" 达标, "❌" 未达标, "" 无参考
    """
    import re

    ref_str = ref_str.strip()
    if not ref_str or ref_str == "-":
        return ""

    match = re.match(r'([><]=?)\s*([\d.]+)', ref_str)
    if not match:
        return ""

    op, threshold_str = match.groups()
    threshold = float(threshold_str)

    if op in (">", ">="):
        return "✅" if value >= threshold else "❌"
    elif op in ("<", "<="):
        return "✅" if value <= threshold else "❌"
    return ""

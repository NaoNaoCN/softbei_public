"""RAG 评估分析引擎：从评估报告中识别瓶颈、模式、生成改进建议。

用法：
    python -m backend.evaluation.analyzer --run
    python -m backend.evaluation.analyzer --from-db --since 2026-05-01
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

from loguru import logger

from backend.evaluation.models import (
    GoldenRegressionReport,
    GoldenEvalResult,
    RAGEvalAnalysisReport,
    AnalysisSuggestion,
    KPBreakdown,
)

# 质量阈值（与 reporter.py 参考标准对齐）
_THRESHOLDS: dict[str, tuple[str, float]] = {
    "faithfulness":      ("high", 0.70),
    "completeness":      ("high", 0.60),
    "precision_at_5":    ("high", 0.60),
}


class RAGAnalyzer:
    """RAG 评估分析器。

    用法::

        analyzer = RAGAnalyzer()
        analysis = await analyzer.analyze_golden_report(report)
        analysis_id = await analyzer.save_analysis(analysis)
    """

    def __init__(self):
        self._thresholds = _THRESHOLDS.copy()

    async def analyze_golden_report(
        self,
        report: GoldenRegressionReport,
    ) -> RAGEvalAnalysisReport:
        """分析黄金测试集报告，返回结构化分析结果。"""
        if report.total_queries == 0:
            return RAGEvalAnalysisReport(
                total_queries=0,
                key_findings=["黄金测试集为空，无法分析"],
            )

        scores = {
            "faithfulness": report.avg_faithfulness,
            "completeness": report.avg_completeness,
            "precision_at_5": report.avg_precision_at_5,
        }
        pass_rate = report.passed / max(report.total_queries, 1)

        bottlenecks = self._identify_bottlenecks(scores, pass_rate)
        query_categories = self._categorize_queries(report.per_query_results)
        per_kp = self._build_per_kp_breakdown(report.per_query_results)
        correlations = self._check_correlations(scores)
        strengths, weaknesses = self._identify_strengths_weaknesses(
            bottlenecks, query_categories, per_kp
        )
        suggestions = self._generate_suggestions(
            bottlenecks, query_categories, per_kp, correlations
        )
        trend = await self._determine_trend("golden_run", pass_rate)
        key_findings = self._generate_key_findings(
            bottlenecks, query_categories, strengths, weaknesses, correlations
        )

        bottleneck_labels = [b["label"] for b in bottlenecks if b["severity"] != "ok"]

        return RAGEvalAnalysisReport(
            analysis_type="golden_run",
            source_reference=f"golden_run_{report.run_timestamp.isoformat()}",
            created_at=datetime.utcnow(),
            avg_faithfulness=round(report.avg_faithfulness, 4),
            avg_completeness=round(report.avg_completeness, 4),
            avg_precision_at_5=round(report.avg_precision_at_5, 4),
            pass_rate=round(pass_rate, 4),
            total_queries=report.total_queries,
            bottlenecks=bottleneck_labels,
            strengths=strengths,
            weaknesses=weaknesses,
            suggestions=suggestions,
            key_findings=key_findings,
            per_kp_breakdown=per_kp,
            trend=trend,
            extra_metadata={
                "passed": report.passed,
                "failed": report.failed,
                "correlations": correlations,
                "query_categories": {
                    k: v for k, v in query_categories.items()
                    if k != "queries_by_category"
                },
            },
        )

    def _identify_bottlenecks(
        self,
        scores: dict[str, float],
        pass_rate: float,
    ) -> list[dict]:
        """对比各指标与阈值，返回按严重程度排序的瓶颈列表。"""
        bottlenecks = []
        for name, value in scores.items():
            direction, threshold = self._thresholds[name]
            if direction == "high":
                gap = value - threshold
            else:
                gap = threshold - value

            if gap < -0.15:
                severity = "critical"
            elif gap < 0:
                severity = "warning"
            else:
                severity = "ok"

            label = f"{name}: {value:.2f} (阈值={threshold:.2f}, 差距={gap:+.2f})"

            bottlenecks.append({
                "metric": name,
                "current": round(value, 4),
                "threshold": threshold,
                "gap": round(gap, 4),
                "severity": severity,
                "label": label,
            })

        severity_order = {"critical": 0, "warning": 1, "ok": 2}
        bottlenecks.sort(key=lambda b: (severity_order[b["severity"]], b["gap"]))

        return bottlenecks

    def _categorize_queries(
        self,
        results: list[GoldenEvalResult],
    ) -> dict:
        """将查询按性能模式分类。"""
        categories = {
            "strong": 0,
            "retrieval_weak": 0,
            "generation_weak": 0,
            "both_weak": 0,
            "queries_by_category": {
                "strong": [],
                "retrieval_weak": [],
                "generation_weak": [],
                "both_weak": [],
            },
        }

        for r in results:
            fid = r.query_id
            if r.faithfulness_pass and r.completeness_pass:
                categories["strong"] += 1
                categories["queries_by_category"]["strong"].append(fid)
            elif r.faithfulness_pass and not r.completeness_pass:
                categories["retrieval_weak"] += 1
                categories["queries_by_category"]["retrieval_weak"].append(fid)
            elif not r.faithfulness_pass and r.completeness_pass:
                categories["generation_weak"] += 1
                categories["queries_by_category"]["generation_weak"].append(fid)
            else:
                categories["both_weak"] += 1
                categories["queries_by_category"]["both_weak"].append(fid)

        return categories

    def _build_per_kp_breakdown(
        self,
        results: list[GoldenEvalResult],
    ) -> dict[str, KPBreakdown]:
        """按知识点头聚合分析。"""
        by_kp: dict[str, list[GoldenEvalResult]] = {}
        for r in results:
            kp = r.kp_name or "unknown"
            by_kp.setdefault(kp, []).append(r)

        breakdown: dict[str, KPBreakdown] = {}
        for kp_name, kp_results in by_kp.items():
            n = len(kp_results)
            faith_scores = [r.faithfulness_score for r in kp_results]
            comp_scores = [r.completeness_score for r in kp_results]
            prec_scores = [r.precision_at_5 for r in kp_results if r.precision_at_5 > 0]

            passed = sum(1 for r in kp_results if r.faithfulness_pass and r.completeness_pass)
            failed = n - passed

            breakdown[kp_name] = KPBreakdown(
                kp_name=kp_name,
                avg_faithfulness=round(sum(faith_scores) / n, 4) if n else 0.0,
                avg_completeness=round(sum(comp_scores) / n, 4) if n else 0.0,
                avg_precision_at_5=round(sum(prec_scores) / len(prec_scores), 4) if prec_scores else 0.0,
                pass_count=passed,
                fail_count=failed,
                query_ids=[r.query_id for r in kp_results],
            )

        return breakdown

    def _check_correlations(
        self,
        scores: dict[str, float],
    ) -> dict:
        """检测指标间的相关性模式。"""
        faith = scores.get("faithfulness", 0)
        comp = scores.get("completeness", 0)
        prec = scores.get("precision_at_5", 0)

        return {
            "kb_coverage_issue": prec > 0.60 and comp < 0.50,
            "retrieval_quality_issue": prec < 0.40,
            "generation_quality_issue": prec > 0.60 and faith < 0.50,
            "systemic_issue": prec < 0.40 and faith < 0.60 and comp < 0.50,
        }

    def _identify_strengths_weaknesses(
        self,
        bottlenecks: list[dict],
        query_categories: dict,
        per_kp_breakdown: dict[str, KPBreakdown],
    ) -> tuple[list[str], list[str]]:
        """识别系统的优势和劣势。"""
        strengths = []
        weaknesses = []

        for b in bottlenecks:
            if b["severity"] == "ok":
                strengths.append(
                    f"{b['metric']}: {b['current']:.2f} (高于阈值 {b['threshold']:.2f}, +{b['gap']:.2f})"
                )

        healthy_kps = [
            name for name, kp in per_kp_breakdown.items()
            if kp.fail_count == 0
        ]
        if healthy_kps:
            strengths.append(f"{len(healthy_kps)} 个知识点全部通过: {', '.join(healthy_kps)}")

        strong_count = query_categories.get("strong", 0)
        if strong_count > 0:
            strengths.append(f"{strong_count} 条查询在所有指标上均达标 (strong)")

        for b in bottlenecks:
            if b["severity"] in ("critical", "warning"):
                weaknesses.append(b["label"])

        zero_pass_kps = [
            name for name, kp in per_kp_breakdown.items()
            if kp.pass_count == 0
        ]
        if zero_pass_kps:
            weaknesses.append(f"零通过率知识点: {', '.join(zero_pass_kps)}")

        weak_count = query_categories.get("retrieval_weak", 0)
        gen_weak_count = query_categories.get("generation_weak", 0)
        both_count = query_categories.get("both_weak", 0)
        total = sum([
            query_categories.get("strong", 0), weak_count, gen_weak_count, both_count
        ])

        if total > 0:
            if both_count / total > 0.3:
                weaknesses.append(
                    f"{both_count}/{total} ({both_count / total:.0%}) 查询在检索和生成两方面都存在问题"
                )
            if weak_count > gen_weak_count:
                weaknesses.append(
                    f"检索覆盖是主要短板: {weak_count} 条查询 completeness 不达标"
                )
            elif gen_weak_count > weak_count:
                weaknesses.append(
                    f"生成忠实度需要关注: {gen_weak_count} 条查询 faithfulness 不达标"
                )

        return strengths, weaknesses

    def _generate_suggestions(
        self,
        bottlenecks: list[dict],
        query_categories: dict,
        per_kp_breakdown: dict[str, KPBreakdown],
        correlations: dict,
    ) -> list[AnalysisSuggestion]:
        """根据瓶颈和相关性生成可操作的改进建议。"""
        suggestions: list[AnalysisSuggestion] = []

        critical_bottlenecks = [b for b in bottlenecks if b["severity"] == "critical"]
        warning_bottlenecks = [b for b in bottlenecks if b["severity"] == "warning"]

        if correlations.get("systemic_issue"):
            suggestions.append(AnalysisSuggestion(
                priority="high",
                category="retrieval",
                action="检查向量库是否正常初始化、embedding 模型 API 是否可用",
                rationale="所有三个核心指标均显著低于阈值，可能存在基础设施问题",
            ))
            return suggestions

        if correlations.get("kb_coverage_issue"):
            zero_kps = [
                name for name, kp in per_kp_breakdown.items()
                if kp.pass_count == 0
            ]
            suggestions.append(AnalysisSuggestion(
                priority="high",
                category="kb",
                action=f"为零通过率的知识点扩充文档资源: {', '.join(zero_kps[:5])}",
                rationale="检索精度正常(P@5>0.6)但完整度低(<0.5)，知识库中缺少这些知识点的关键方面",
            ))
            suggestions.append(AnalysisSuggestion(
                priority="medium",
                category="kb",
                action="增大 chunk_size 或调整父子切割参数，使每个检索块包含更完整的上下文",
                rationale="当前 chunk 可能粒度太细，丢失了知识点全貌",
            ))

        if correlations.get("retrieval_quality_issue"):
            suggestions.append(AnalysisSuggestion(
                priority="high",
                category="retrieval",
                action="检查 embedding 模型 (BGE-M3) 对当前领域术语的向量表示质量",
                rationale="P@5 低于 0.4，向量检索返回了大量不相关内容",
            ))
            suggestions.append(AnalysisSuggestion(
                priority="medium",
                category="retrieval",
                action="降低 score_threshold 或启用混合检索 (hybrid search) 改善召回",
                rationale="纯向量检索可能不适合所有查询类型",
            ))

        if correlations.get("generation_quality_issue"):
            suggestions.append(AnalysisSuggestion(
                priority="high",
                category="generation",
                action="优化 system prompt，强化'仅使用参考资料回答，标注引用编号'的指令",
                rationale="检索精度正常但 LLM 生成的内容包含无法溯源的信息",
            ))
            suggestions.append(AnalysisSuggestion(
                priority="medium",
                category="generation",
                action="降低 LLM temperature 以减少创造性编造",
                rationale="高温生成更容易产生与参考资料不一致的陈述",
            ))

        if warning_bottlenecks:
            if any(b["metric"] == "completeness" for b in warning_bottlenecks):
                suggestions.append(AnalysisSuggestion(
                    priority="medium",
                    category="retrieval",
                    action="增加 n_results（检索数量）从 5 到 8-10，提升检索广度",
                    rationale="completeness 偏低可能因为检索到的 chunk 数量不足以覆盖全部知识面",
                ))
            if any(b["metric"] == "faithfulness" for b in warning_bottlenecks):
                suggestions.append(AnalysisSuggestion(
                    priority="medium",
                    category="prompt",
                    action="检查 generation prompt 中的温度、top_p 等生成参数",
                    rationale="faithfulness 偏低需要排查 LLM 是否有过度创造性输出的倾向",
                ))

        seen = set()
        unique_suggestions = []
        for s in suggestions:
            if s.action not in seen:
                seen.add(s.action)
                unique_suggestions.append(s)

        return unique_suggestions

    def _generate_key_findings(
        self,
        bottlenecks: list[dict],
        query_categories: dict,
        strengths: list[str],
        weaknesses: list[str],
        correlations: dict,
    ) -> list[str]:
        """生成 1-3 条关键发现。"""
        findings = []

        critical = [b for b in bottlenecks if b["severity"] == "critical"]
        if critical:
            findings.append(
                f"首要瓶颈是 {critical[0]['metric']}: 当前值 {critical[0]['current']:.2f}, "
                f"低于阈值 {critical[0]['threshold']:.2f}"
            )
        elif bottlenecks and bottlenecks[0]["severity"] == "warning":
            findings.append(
                f"系统整体健康，但 {bottlenecks[0]['metric']} 接近阈值边缘 "
                f"({bottlenecks[0]['current']:.2f} vs {bottlenecks[0]['threshold']:.2f})"
            )
        else:
            findings.append("所有核心指标均高于阈值，系统运行优良")

        strong_pct = query_categories.get("strong", 0) / max(
            sum(v for k, v in query_categories.items() if k != "queries_by_category"), 1
        )
        if correlations.get("kb_coverage_issue"):
            findings.append(
                "诊断: 检索精度好但完整度不足 → 知识库内容存在缺口，建议重点扩充文档"
            )
        elif correlations.get("generation_quality_issue"):
            findings.append(
                "诊断: 检索资料质量尚可但 LLM 生成不稳定 → 建议调整 prompt 策略"
            )
        elif strong_pct >= 0.8:
            findings.append(f"整体表现良好，{strong_pct:.0%} 的查询通过所有指标")

        zero_kps = [
            k for k, v in query_categories.get("queries_by_category", {}).items()
        ]
        both_weak = len(query_categories.get("queries_by_category", {}).get("both_weak", []))
        if both_weak > 0:
            findings.append(
                f"{both_weak} 条查询在检索和生成方面双侧不达标，建议优先排查"
            )

        return findings

    async def _determine_trend(
        self,
        analysis_type: str,
        current_pass_rate: float,
    ) -> str:
        """判断趋势（简化版：无历史数据时返回 stable）。"""
        return "stable"


def main():
    parser = argparse.ArgumentParser(description="RAG 评估分析引擎")
    parser.add_argument("--run", action="store_true", help="分析最新的黄金测试集报告")
    parser.add_argument("--output", type=str, default=None, help="分析报告输出路径")
    args = parser.parse_args()

    if args.run:
        import asyncio

        async def _run():
            from backend.evaluation.golden_dataset import load_golden_queries, run_golden_evaluation, format_compact_report
            from backend.db.database import init_db

            await init_db()

            queries = load_golden_queries()
            if not queries:
                print("错误：黄金测试集为空或文件不存在")
                return

            print(f"开始评估 {len(queries)} 条黄金查询...")
            report = await run_golden_evaluation(queries)

            analyzer = RAGAnalyzer()
            analysis = await analyzer.analyze_golden_report(report)

            md = format_compact_report(report, analysis)
            print(md)

            out_dir = Path(__file__).parent.parent.parent / "logs"
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            report_file = out_dir / f"golden_eval_{ts}.md"
            report_file.write_text(md, encoding="utf-8")
            print(f"报告已保存到: {report_file}")

        asyncio.run(_run())
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

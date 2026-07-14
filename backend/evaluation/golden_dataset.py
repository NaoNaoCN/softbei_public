"""Layer 3 黄金测试集：批量离线评估 + 回归检测。"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml
from loguru import logger

from backend.config import config
from backend.evaluation.models import (
    GoldenQuery,
    GoldenEvalResult,
    GoldenRegressionReport,
)
from backend.evaluation.metrics import ndcg_at_k, recall_at_k, mrr, hit_rate


def load_golden_queries(path: str | None = None) -> list[GoldenQuery]:
    """从 YAML 文件加载黄金测试集。"""
    file_path = Path(path or config.evaluation.golden_dataset.path)
    if not file_path.is_absolute():
        file_path = Path(__file__).parent.parent.parent / file_path

    if not file_path.exists():
        logger.warning(f"[GoldenDataset] 黄金测试集文件不存在: {file_path}")
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    queries = []
    for item in raw.get("queries", []):
        queries.append(GoldenQuery(**item))

    logger.info(f"[GoldenDataset] 加载 {len(queries)} 条黄金查询 from {file_path}")
    return queries


async def run_golden_evaluation(
    queries: list[GoldenQuery] | None = None,
    n_results: int = 5,
) -> GoldenRegressionReport:
    """
    对黄金测试集逐条执行 RAG 检索 + LLM Judge 评估。

    :param queries:   黄金查询列表，None=从默认文件加载
    :param n_results: 每条查询的检索条数
    :return:          回归报告
    """
    if queries is None:
        queries = load_golden_queries()

    if not queries:
        return GoldenRegressionReport(
            total_queries=0,
            regression_details=["黄金测试集为空"],
        )

    # 初始化数据库引擎（向量检索依赖）
    from backend.db.database import init_db
    await init_db()

    from backend.rag.retriever import retrieve_by_kp
    from backend.evaluation.judge import get_judge

    judge = get_judge()
    per_query_results: list[GoldenEvalResult] = []
    passed = 0
    failed = 0

    for i, gq in enumerate(queries, 1):
        logger.info(f"[GoldenDataset] [{i}/{len(queries)}] 评估: {gq.id} — {gq.query[:60]}")
        t_start = time.perf_counter()

        try:
            chunks = await retrieve_by_kp(
                gq.kp_name,
                n_results=n_results,
            )
            retrieved_texts = [c.text for c in chunks]

            # 使用与 Agent 管线一致的 system prompt 结构（含反幻觉指令），
            # 确保评估结果能反映 Agent 实际生成质量
            from backend.services.llm import chat_completion

            context = "\n\n---\n\n".join(
                f"[{j+1}] {text}" for j, text in enumerate(retrieved_texts[:5])
            ) if retrieved_texts else "（暂无参考资料）"

            gen_prompt = (
                f"# Role\n"
                f"你是教学资料撰写专家。你的唯一任务是：基于参考资料回答学生的问题。\n"
                f"你**不是**该知识点的权威专家——你的回答必须来自参考资料，不得越界。\n\n"
                f"# Rules（优先级从高到低）\n\n"
                f"## NEVER — 绝对禁止\n"
                f"1. **禁止编造事实**：不得写入参考资料中不存在的数据、公式、定义或结论。\n"
                f"2. **禁止凭记忆补充**：即使你\"知道\"某个相关信息，如果参考资料中没有，就不能写入正文。\n"
                f"3. **禁止跳过空引用检查**：如果参考资料为\"（暂无参考资料）\"，必须在开头声明。\n\n"
                f"## IMPORTANT — 必须做到\n"
                f"4. **引用追溯**：正文中来自参考资料的每一条关键陈述，标注来源编号 [n]。\n"
                f"5. **覆盖诚实**：如果参考资料不足以完整回答，在末尾说明哪些方面未被覆盖。\n\n"
                f"# Output\n"
                f"使用 Markdown，直接回答问题。\n\n"
                f"---\n"
                f"参考资料：\n{context}\n\n"
                f"问题：{gq.query}"
            )
            generated = await chat_completion(
                [{"role": "system", "content": gen_prompt}],
                temperature=0.1,
                max_tokens=2000,
            )

            eval_result = await judge.evaluate_full(
                query=gq.query,
                kp_name=gq.kp_name,
                retrieved_chunks=chunks,
                generated_content=generated,
            )

            eval_time = (time.perf_counter() - t_start) * 1000

            faith_pass = eval_result["faithfulness_score"] >= gq.min_faithfulness
            comp_pass = eval_result["completeness_score"] >= gq.min_completeness
            if faith_pass and comp_pass:
                passed += 1
            else:
                failed += 1
                logger.warning(
                    f"[GoldenDataset] {gq.id} 未通过: "
                    f"faithfulness={eval_result['faithfulness_score']:.2f} (min={gq.min_faithfulness}), "
                    f"completeness={eval_result['completeness_score']:.2f} (min={gq.min_completeness})"
                )

            rel_labels = eval_result.get("relevance_labels", [])
            p5 = eval_result.get("precision_at_5", 0.0)
            # 黄金测试集无 ground truth total_relevant，以 k=5 为分母
            # 即 Recall@5 = 相关 chunk 数 / 5，与 P@5 视角互补
            r5 = recall_at_k(rel_labels, total_relevant=n_results, k=n_results) if rel_labels else 0.0
            n5 = ndcg_at_k(rel_labels, n_results) if rel_labels else 0.0

            per_query_results.append(GoldenEvalResult(
                query_id=gq.id,
                kp_name=gq.kp_name,
                query=gq.query,
                precision_at_5=p5,
                recall_at_5=r5,
                ndcg_at_5=n5,
                relevance_labels=rel_labels,
                faithfulness_score=eval_result["faithfulness_score"],
                completeness_score=eval_result["completeness_score"],
                hallucination_rate=eval_result.get("hallucination_rate", 0.0),
                citation_precision=eval_result.get("citation_precision"),
                faithfulness_pass=faith_pass,
                completeness_pass=comp_pass,
                evaluation_time_ms=round(eval_time, 1),
            ))

        except Exception as e:
            logger.error(f"[GoldenDataset] {gq.id} 评估异常: {e}")
            failed += 1
            per_query_results.append(GoldenEvalResult(
                query_id=gq.id,
                kp_name=gq.kp_name,
                query=gq.query,
                faithfulness_pass=False,
                completeness_pass=False,
            ))

    report = _build_report(per_query_results, passed, failed)
    return report


def _build_report(
    results: list[GoldenEvalResult],
    passed: int,
    failed: int,
) -> GoldenRegressionReport:
    """从逐条结果构建回归报告。"""
    if not results:
        return GoldenRegressionReport(
            total_queries=0,
            regression_details=["无有效评估结果"],
        )

    prec_scores = [r.precision_at_5 for r in results if r.precision_at_5 > 0]
    recall_scores = [r.recall_at_5 for r in results if r.recall_at_5 > 0]
    ndcg_scores = [r.ndcg_at_5 for r in results if r.ndcg_at_5 > 0]

    # MRR / Hit Rate 依赖所有查询的 relevance_labels，需跨查询计算
    all_rel_labels = [r.relevance_labels for r in results if r.relevance_labels]
    mrr_val = mrr(all_rel_labels) if all_rel_labels else 0.0
    hit_val = hit_rate(all_rel_labels, 5) if all_rel_labels else 0.0

    faith_scores = [r.faithfulness_score for r in results if r.faithfulness_score > 0]
    comp_scores = [r.completeness_score for r in results if r.completeness_score > 0]
    hallu_scores = [r.hallucination_rate for r in results]
    citation_scores = [r.citation_precision for r in results if r.citation_precision is not None]

    avg_prec = sum(prec_scores) / len(prec_scores) if prec_scores else 0.0
    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0
    avg_faith = sum(faith_scores) / len(faith_scores) if faith_scores else 0.0
    avg_comp = sum(comp_scores) / len(comp_scores) if comp_scores else 0.0
    avg_hallu = sum(hallu_scores) / len(hallu_scores) if hallu_scores else 0.0
    avg_citation = sum(citation_scores) / len(citation_scores) if citation_scores else None

    report = GoldenRegressionReport(
        total_queries=len(results),
        passed=passed,
        failed=failed,
        avg_precision_at_5=round(avg_prec, 4),
        avg_recall_at_5=round(avg_recall, 4),
        avg_ndcg_at_5=round(avg_ndcg, 4),
        mrr=round(mrr_val, 4),
        hit_rate_at_5=round(hit_val, 4),
        avg_faithfulness=round(avg_faith, 4),
        avg_completeness=round(avg_comp, 4),
        avg_hallucination_rate=round(avg_hallu, 4),
        avg_citation_precision=round(avg_citation, 4) if avg_citation is not None else None,
        per_query_results=results,
    )

    _detect_regression(report)

    return report


_LAST_RUN_FILE = Path(__file__).parent.parent.parent / "logs" / "golden_last_run.json"


def _detect_regression(report: GoldenRegressionReport) -> None:
    """对比上次运行结果，检测质量回归。"""
    import json

    if _LAST_RUN_FILE.exists():
        try:
            with open(_LAST_RUN_FILE, "r", encoding="utf-8") as f:
                last = json.load(f)

            delta_faith = report.avg_faithfulness - last.get("avg_faithfulness", 0)
            delta_comp = report.avg_completeness - last.get("avg_completeness", 0)
            delta_prec = report.avg_precision_at_5 - last.get("avg_precision_at_5", 0)

            report.delta_faithfulness = round(delta_faith, 4)
            report.delta_completeness = round(delta_comp, 4)
            report.delta_precision_at_5 = round(delta_prec, 4)

            # 任一度量下降 >10% → 回归告警
            regressions = []
            for name, delta, last_val in [
                ("faithfulness", delta_faith, last.get("avg_faithfulness", 0)),
                ("completeness", delta_comp, last.get("avg_completeness", 0)),
                ("precision_at_5", delta_prec, last.get("avg_precision_at_5", 0)),
            ]:
                if last_val > 0 and delta < 0 and abs(delta) / last_val > 0.1:
                    regressions.append(
                        f"{name}: {last_val:.3f} → {last_val + delta:.3f} "
                        f"({delta / last_val:+.1%})"
                    )

            if regressions:
                report.regression_detected = True
                report.regression_details = regressions
                logger.warning(
                    f"[GoldenDataset] 回归检测：{len(regressions)} 个指标下降超过 10%！"
                )
            else:
                logger.info("[GoldenDataset] 回归检测通过，无显著退化")
        except Exception as e:
            logger.warning(f"[GoldenDataset] 读取上次运行结果失败: {e}")

    try:
        _LAST_RUN_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_LAST_RUN_FILE, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": report.run_timestamp.isoformat(),
                "avg_faithfulness": report.avg_faithfulness,
                "avg_completeness": report.avg_completeness,
                "avg_precision_at_5": report.avg_precision_at_5,
                "avg_recall_at_5": report.avg_recall_at_5,
                "avg_ndcg_at_5": report.avg_ndcg_at_5,
                "mrr": report.mrr,
                "hit_rate_at_5": report.hit_rate_at_5,
                "avg_hallucination_rate": report.avg_hallucination_rate,
                "avg_citation_precision": report.avg_citation_precision,
                "total_queries": report.total_queries,
                "passed": report.passed,
                "failed": report.failed,
            }, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def format_compact_report(
    report: GoldenRegressionReport,
    analysis,  # RAGEvalAnalysisReport
) -> str:
    """合成一份精简报告：关键数据 + 警告 + 分析结论。"""
    lines = [
        f"# RAG 黄金测试集评估报告",
        f"",
        f"**运行时间**: {report.run_timestamp.isoformat()}",
        f"**测试集规模**: {report.total_queries} 条查询",
        f"**通过率**: {report.passed}/{report.total_queries} ({report.passed / max(report.total_queries, 1):.0%})",
        f"",
        f"## 检索层指标",
        f"",
        f"| 指标 | 当前值 | 参考标准 | 说明 |",
        f"|------|--------|----------|------|",
        f"| P@5 | {report.avg_precision_at_5:.3f} | ≥ 0.60 | 检索精度：Top-5 中相关 chunk 占比 |",
        f"| Recall@5 | {report.avg_recall_at_5:.3f} | ≥ 0.50 | 检索召回：相关 chunk 被检索到的比例 |",
        f"| MRR | {report.mrr:.3f} | ≥ 0.50 | 平均倒数排名：首个相关 chunk 的排名位置 |",
        f"| NDCG@5 | {report.avg_ndcg_at_5:.3f} | ≥ 0.50 | 排序质量：考虑位置加权的相关性得分 |",
        f"| Hit Rate@5 | {report.hit_rate_at_5:.3f} | ≥ 0.80 | 命中率：至少搜到 1 条相关的查询占比 |",
        f"",
    ]

    retrieval_notes = []
    if report.avg_precision_at_5 >= 0.80 and report.hit_rate_at_5 >= 0.80:
        retrieval_notes.append("检索系统运行良好，精度和命中率均达标")
    if report.avg_precision_at_5 < 0.60:
        retrieval_notes.append(f"P@5 偏低 ({report.avg_precision_at_5:.2f})，检索返回了较多不相关内容，建议检查 embedding 模型或降低 score_threshold")
    if report.mrr < 0.50:
        retrieval_notes.append(f"MRR 偏低 ({report.mrr:.2f})，首个相关文档排名靠后，建议优化精排策略")
    if report.hit_rate_at_5 < 0.80:
        retrieval_notes.append(f"Hit Rate 偏低 ({report.hit_rate_at_5:.0%})，{report.total_queries - int(report.hit_rate_at_5 * report.total_queries)} 条查询完全没有搜到相关内容")
    if report.avg_recall_at_5 < 0.50:
        retrieval_notes.append(f"Recall@5 偏低 ({report.avg_recall_at_5:.2f})，仅靠向量检索难以覆盖所有相关知识面")
    if retrieval_notes:
        lines.append("> " + "；".join(retrieval_notes))
        lines.append("")

    lines += [
        f"## 生成层指标",
        f"",
        f"| 指标 | 当前值 | 参考标准 | 说明 |",
        f"|------|--------|----------|------|",
        f"| Faithfulness | {report.avg_faithfulness:.3f} | ≥ 0.70 | 忠实度：生成内容可被参考资料支撑的比例 |",
        f"| Completeness | {report.avg_completeness:.3f} | ≥ 0.60 | 完整度：答案覆盖知识点各方面的程度 |",
    ]

    if report.avg_citation_precision is not None:
        lines.append(
            f"| Citation Accuracy | {report.avg_citation_precision:.3f} | ≥ 0.70 | 引用准确性：[n] 标注与参考资料的一致性 |"
        )
    else:
        lines.append(
            f"| Citation Accuracy | N/A | ≥ 0.70 | 无显式引用标注，无法评估 |"
        )

    hallu_status = "低" if report.avg_hallucination_rate <= 0.15 else ("偏高" if report.avg_hallucination_rate > 0.30 else "中等")
    lines.append(
        f"| Hallucination Rate | {report.avg_hallucination_rate:.3f} | ≤ 0.15 | 幻觉率：无依据内容占比（{hallu_status}） |"
    )
    lines.append("")

    gen_notes = []
    if report.avg_hallucination_rate > 0.20:
        gen_notes.append(f"幻觉率偏高 ({report.avg_hallucination_rate:.1%})，LLM 生成了较多参考资料中不存在的内容")
    if report.avg_faithfulness >= 0.85 and report.avg_hallucination_rate <= 0.15:
        gen_notes.append("生成忠实度良好，LLM 能准确使用参考资料回答问题")
    if report.avg_citation_precision is not None and report.avg_citation_precision < 0.70:
        gen_notes.append(f"引用准确性偏低 ({report.avg_citation_precision:.2f})，[n] 标注与原文匹配度不足，建议检查 prompt 中的引用指令")
    if gen_notes:
        lines.append("> " + "；".join(gen_notes))
        lines.append("")

    if report.regression_detected:
        lines.append("## [WARNING] 回归告警")
        lines.append("")
        for detail in report.regression_details:
            lines.append(f"- {detail}")
        lines.append("")

    if report.failed > 0:
        lines.append("## 未通过查询")
        lines.append("")
        lines.append("| Query ID | Faithfulness | Completeness |")
        lines.append("|----------|-------------|-------------|")
        for r in report.per_query_results:
            if not r.faithfulness_pass or not r.completeness_pass:
                faith_icon = "[PASS]" if r.faithfulness_pass else "[FAIL]"
                comp_icon = "[PASS]" if r.completeness_pass else "[FAIL]"
                lines.append(
                    f"| {r.query_id} | {faith_icon} {r.faithfulness_score:.2f} | "
                    f"{comp_icon} {r.completeness_score:.2f} |"
                )
        lines.append("")

    lines.append("## 分析结论")
    lines.append("")
    lines.append(f"**趋势**: {analysis.trend}")
    lines.append("")
    if analysis.key_findings:
        for f in analysis.key_findings:
            lines.append(f"- {f}")
    lines.append("")
    if analysis.suggestions:
        lines.append("### 改进建议")
        lines.append("")
        for s in sorted(analysis.suggestions, key=lambda x: {"high": 0, "medium": 1, "low": 2}[x.priority]):
            lines.append(f"- **[{s.priority.upper()}]** {s.action}")
        lines.append("")

    return "\n".join(lines)


def show_dataset_info(path: str | None = None) -> str:
    """显示黄金测试集概要信息。"""
    queries = load_golden_queries(path)
    if not queries:
        return "黄金测试集为空或文件不存在"

    tag_counts: dict[str, int] = {}
    for q in queries:
        for tag in q.tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    lines = [
        f"黄金测试集概要",
        f"────────────────",
        f"查询总数: {len(queries)}",
        f"知识点覆盖: {len(set(q.kp_name for q in queries))}",
        f"标签分布:",
    ]
    for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  {tag}: {count}")
    lines.append("")
    lines.append("查询列表:")
    for q in queries:
        lines.append(f"  [{q.id}] {q.kp_name}: {q.query[:50]}...")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="RAG 黄金测试集评估")
    parser.add_argument("--run", action="store_true", help="运行批量评估")
    parser.add_argument("--dataset", type=str, default=None, help="黄金测试集文件路径")
    parser.add_argument("--info", action="store_true", help="显示测试集概要")
    parser.add_argument("--n-results", type=int, default=5, help="每条查询检索条数")
    args = parser.parse_args()

    if args.info:
        print(show_dataset_info(args.dataset))
        return

    if args.run:
        import asyncio
        import io
        import sys

        queries = load_golden_queries(args.dataset)
        if not queries:
            print("Error: golden dataset is empty or file not found")
            return

        print(f"Starting evaluation of {len(queries)} golden queries...")
        report = asyncio.run(run_golden_evaluation(queries, n_results=args.n_results))

        # 分析引擎（仅内存分析，不持久化到 DB）
        from backend.evaluation.analyzer import RAGAnalyzer
        analyzer = RAGAnalyzer()
        analysis = asyncio.run(analyzer.analyze_golden_report(report))

        md = format_compact_report(report, analysis)

        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        print(md)

        out_dir = Path(__file__).parent.parent.parent / "logs"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        report_file = out_dir / f"golden_eval_{ts}.md"
        report_file.write_text(md, encoding="utf-8")
        print(f"报告已保存到: {report_file}")

        if report.regression_detected:
            print(f"\n[WARNING] Regression detected: {len(report.regression_details)} metrics dropped!")
            for d in report.regression_details:
                print(f"  - {d}")
            exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

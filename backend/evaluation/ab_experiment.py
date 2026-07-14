"""Layer 4 A/B 实验框架：对比两个配置组在相同查询上的 RAG 质量。

用法：
    python -m backend.evaluation.ab --group-a baseline --group-b chunk_size_800 \\
        --queries backend/evaluation/golden_queries.yaml

    # 仅对比检索质量（不跑 LLM Judge）
    python -m backend.evaluation.ab --group-a baseline --group-b chunk_size_800 \\
        --queries golden_queries.yaml --retrieval-only
"""

from __future__ import annotations

import argparse
import asyncio
import time
from datetime import datetime
from typing import Optional

from loguru import logger

from backend.config import config
from backend.evaluation.models import ABExperimentResult


async def run_ab_experiment(
    group_a: str,
    group_b: str,
    queries: list[str] | None = None,
    kp_names: list[str] | None = None,
    golden_path: str | None = None,
    retrieval_only: bool = False,
    n_results: int = 5,
) -> ABExperimentResult:
    """
    运行 A/B 实验：对相同的查询集分别用两组配置进行 RAG 评估。

    当前实现：通过 config 切换检索参数（如 n_results、score_threshold），
    在同一进程中依次执行两组评估。

    :param group_a:        对照组标签
    :param group_b:        实验组标签
    :param queries:        查询字符串列表
    :param kp_names:       知识点名称列表（与 queries 一一对应）
    :param golden_path:    黄金测试集路径（与 queries/kp_names 二选一）
    :param retrieval_only: 仅评估检索质量（不跑 LLM Judge）
    :param n_results:      检索条数
    :return:               A/B 对比结果
    """
    if golden_path and not queries:
        from backend.evaluation.golden_dataset import load_golden_queries
        gqs = load_golden_queries(golden_path)
        queries = [gq.query for gq in gqs]
        kp_names = [gq.kp_name for gq in gqs]

    if not queries:
        raise ValueError("必须提供 queries 或 golden_path")

    logger.info(
        f"[ABExperiment] 开始 A/B 实验: {group_a} vs {group_b}, "
        f"{len(queries)} 条查询, retrieval_only={retrieval_only}"
    )

    logger.info(f"[ABExperiment] 运行对照组: {group_a}")
    metrics_a = await _run_group(queries, kp_names or queries, group_a, retrieval_only, n_results)

    logger.info(f"[ABExperiment] 运行实验组: {group_b}")
    metrics_b = await _run_group(queries, kp_names or queries, group_b, retrieval_only, n_results)

    metric_deltas: dict[str, dict] = {}
    for key in metrics_a:
        a_val = metrics_a[key]
        b_val = metrics_b.get(key, 0.0)
        delta = b_val - a_val
        pct = (delta / a_val * 100) if a_val != 0 else 0.0
        metric_deltas[key] = {
            "group_a": round(a_val, 4),
            "group_b": round(b_val, 4),
            "delta": round(delta, 4),
            "pct_change": round(pct, 1),
        }

    improvements = []
    regressions = []
    for name, d in metric_deltas.items():
        if d["pct_change"] > 5:
            improvements.append(f"{name} +{d['pct_change']:.1f}%")
        elif d["pct_change"] < -5:
            regressions.append(f"{name} {d['pct_change']:.1f}%")

    conclusion_parts = []
    if improvements:
        conclusion_parts.append(f"改善: {', '.join(improvements)}")
    if regressions:
        conclusion_parts.append(f"退化: {', '.join(regressions)}")
    if not conclusion_parts:
        conclusion_parts.append("两组无显著差异")

    result = ABExperimentResult(
        experiment_name=f"{group_a}_vs_{group_b}",
        group_a=group_a,
        group_b=group_b,
        group_a_metrics={k: round(v, 4) for k, v in metrics_a.items()},
        group_b_metrics={k: round(v, 4) for k, v in metrics_b.items()},
        metric_deltas=metric_deltas,
        conclusion="; ".join(conclusion_parts),
    )

    logger.info(f"[ABExperiment] 结论: {result.conclusion}")
    return result


async def _run_group(
    queries: list[str],
    kp_names: list[str],
    group_label: str,
    retrieval_only: bool,
    n_results: int,
) -> dict[str, float]:
    """运行一组评估，返回聚合指标。"""
    from backend.rag.retriever import retrieve_by_kp

    all_scores: list[float] = []
    all_n_retrieved: list[int] = []
    faith_scores: list[float] = []
    comp_scores: list[float] = []
    total_latency: list[float] = []

    for query, kp_name in zip(queries, kp_names):
        t_start = time.perf_counter()

        try:
            chunks = await retrieve_by_kp(kp_name, n_results=n_results)
            all_scores.extend([c.score for c in chunks])
            all_n_retrieved.append(len(chunks))

            if not retrieval_only and chunks:
                from backend.services.llm import chat_completion
                from backend.evaluation.judge import get_judge

                context = "\n\n---\n\n".join([c.text for c in chunks[:5]])
                gen_prompt = (
                    f"你是学习助手。请根据以下参考资料回答问题。\n\n"
                    f"参考资料：\n{context}\n\n问题：{query}"
                )
                generated = await chat_completion(
                    [{"role": "user", "content": gen_prompt}],
                    temperature=0.3,
                    max_tokens=2000,
                )

                judge = get_judge()
                eval_result = await judge.evaluate_full(
                    query=query,
                    kp_name=kp_name,
                    retrieved_chunks=chunks,
                    generated_content=generated,
                    experiment_group=group_label,
                )
                faith_scores.append(eval_result["faithfulness_score"])
                comp_scores.append(eval_result["completeness_score"])

            total_latency.append((time.perf_counter() - t_start) * 1000)

        except Exception as e:
            logger.warning(f"[ABExperiment] 查询失败 [{group_label}] {query[:40]}: {e}")

    n = len(queries)
    metrics: dict[str, float] = {
        "avg_n_retrieved": sum(all_n_retrieved) / n if n else 0,
        "empty_rate": sum(1 for x in all_n_retrieved if x == 0) / n if n else 0,
        "avg_score": sum(all_scores) / len(all_scores) if all_scores else 0,
        "avg_latency_ms": sum(total_latency) / n if n else 0,
    }
    if faith_scores:
        metrics["avg_faithfulness"] = sum(faith_scores) / len(faith_scores)
    if comp_scores:
        metrics["avg_completeness"] = sum(comp_scores) / len(comp_scores)

    return metrics


def format_ab_report(result: ABExperimentResult) -> str:
    """将 A/B 实验结果渲染为 Markdown 对比报告。"""
    lines = [
        f"# A/B 实验报告: {result.experiment_name}",
        f"",
        f"**运行时间**: {result.run_timestamp.isoformat()}",
        f"**对照组**: `{result.group_a}` | **实验组**: `{result.group_b}`",
        f"",
        f"## 指标对比",
        f"",
        f"| 指标 | {result.group_a} | {result.group_b} | Delta | 变化% |",
        f"|------|{"-" * len(result.group_a)}|{"-" * len(result.group_b)}|-------|-------|",
    ]

    for name, d in result.metric_deltas.items():
        direction = "📈" if d["pct_change"] > 5 else ("📉" if d["pct_change"] < -5 else "➡")
        lines.append(
            f"| {name} | {d['group_a']:.3f} | {d['group_b']:.3f} | "
            f"{d['delta']:+.3f} | {direction} {d['pct_change']:+.1f}% |"
        )

    lines.append("")
    lines.append(f"## 结论")
    lines.append(f"")
    lines.append(f"{result.conclusion}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="RAG A/B 实验")
    parser.add_argument("--group-a", type=str, required=True, help="对照组标签")
    parser.add_argument("--group-b", type=str, required=True, help="实验组标签")
    parser.add_argument("--queries", type=str, default=None, help="黄金测试集路径")
    parser.add_argument("--retrieval-only", action="store_true", help="仅对比检索质量")
    parser.add_argument("--n-results", type=int, default=5, help="检索条数")
    args = parser.parse_args()

    logger.info(f"A/B 实验: {args.group_a} vs {args.group_b}")

    result = asyncio.run(run_ab_experiment(
        group_a=args.group_a,
        group_b=args.group_b,
        golden_path=args.queries,
        retrieval_only=args.retrieval_only,
        n_results=args.n_results,
    ))

    md = format_ab_report(result)
    print(md)

    from pathlib import Path
    out_dir = Path(__file__).parent.parent.parent / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report_file = out_dir / f"ab_report_{result.experiment_name}_{ts}.md"
    report_file.write_text(md, encoding="utf-8")
    print(f"\n报告已保存到: {report_file}")


if __name__ == "__main__":
    main()


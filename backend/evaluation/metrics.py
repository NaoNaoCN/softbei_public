"""
backend/evaluation/metrics.py
RAG 评估指标计算：Precision@K, Recall@K, MRR, NDCG, 分数分布等。
"""

from __future__ import annotations

import math


def precision_at_k(relevance_labels: list[int], k: int) -> float:
    """
    Precision@K: Top-K 结果中相关结果的占比。

    :param relevance_labels: 按排名顺序的相关度标签列表（0=无关, 1=部分相关, 2=高度相关）
    :param k:                截断位置
    :return:                 Precision@K (0.0 ~ 1.0)
    """
    if k <= 0 or not relevance_labels:
        return 0.0
    top_k = relevance_labels[:k]
    return sum(1 for r in top_k if r > 0) / k


def recall_at_k(relevance_labels: list[int], total_relevant: int, k: int) -> float:
    """
    Recall@K: 所有相关结果中被检索到的比例。

    :param relevance_labels: 按排名顺序的相关度标签列表
    :param total_relevant:   数据集中该查询的相关文档总数
    :param k:                截断位置
    :return:                 Recall@K (0.0 ~ 1.0)
    """
    if k <= 0 or total_relevant <= 0 or not relevance_labels:
        return 0.0
    top_k = relevance_labels[:k]
    return sum(1 for r in top_k if r > 0) / total_relevant


def mrr(relevance_labels_list: list[list[int]]) -> float:
    """
    MRR (Mean Reciprocal Rank): 第一个相关结果排名的倒数均值。

    :param relevance_labels_list: 多条查询的相关度标签列表
    :return:                      MRR (0.0 ~ 1.0)
    """
    if not relevance_labels_list:
        return 0.0
    reciprocal_ranks: list[float] = []
    for labels in relevance_labels_list:
        for i, rel in enumerate(labels, 1):
            if rel > 0:
                reciprocal_ranks.append(1.0 / i)
                break
        else:
            reciprocal_ranks.append(0.0)
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def ndcg_at_k(relevance_labels: list[int], k: int) -> float:
    """
    NDCG@K: 位置加权的归一化折损累积增益。

    DCG@K = Σ(i=1→K) (2^rel_i - 1) / log₂(i + 1)
    NDCG@K = DCG@K / IDCG@K

    :param relevance_labels: 按排名顺序的相关度标签列表
    :param k:                截断位置
    :return:                 NDCG@K (0.0 ~ 1.0)
    """
    if k <= 0 or not relevance_labels:
        return 0.0

    def _dcg(labels: list[int]) -> float:
        return sum(
            (2 ** rel - 1) / math.log2(i + 2)
            for i, rel in enumerate(labels[:k])
        )

    dcg = _dcg(relevance_labels)
    ideal = sorted(relevance_labels, reverse=True)
    idcg = _dcg(ideal)
    return dcg / idcg if idcg > 0 else 0.0


def hit_rate(relevance_labels_list: list[list[int]], k: int) -> float:
    """
    Hit Rate@K: Top-K 中至少有一条相关结果的查询占比。

    :param relevance_labels_list: 多条查询的相关度标签列表
    :param k:                     截断位置
    :return:                      Hit Rate (0.0 ~ 1.0)
    """
    if not relevance_labels_list:
        return 0.0
    hits = sum(1 for labels in relevance_labels_list if any(r > 0 for r in labels[:k]))
    return hits / len(relevance_labels_list)


def hallucination_rate_from_statements(statements: list[dict]) -> float:
    """
    从 Faithfulness Judge 的 statements 列表中计算幻觉率。

    :param statements: [{"text": "...", "verdict": "supported|unsupported"}, ...]
    :return:           幻觉率 (0.0 ~ 1.0)
    """
    if not statements:
        return 0.0
    unsupported = sum(1 for s in statements if s.get("verdict") == "unsupported")
    return unsupported / len(statements)


def hallucination_rate(faithfulness_result: dict) -> float:
    """
    从 Faithfulness Judge 的完整结果中计算幻觉率。

    :param faithfulness_result: Judge 2 返回的完整 dict
    :return:                    幻觉率 (0.0 ~ 1.0)
    """
    statements = faithfulness_result.get("statements", [])
    return hallucination_rate_from_statements(statements)


def score_distribution(scores: list[float]) -> dict[str, float]:
    """
    计算检索分数的分位数分布（不使用 numpy，纯 Python 实现）。

    :param scores: 分数列表
    :return:       {"p25": ..., "p50": ..., "p75": ..., "p90": ..., "min": ..., "max": ...}
    """
    if not scores:
        return {"p25": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0}

    sorted_scores = sorted(scores)
    n = len(sorted_scores)

    def _percentile(p: float) -> float:
        """线性插值计算第 p 百分位数。"""
        k = (n - 1) * p / 100.0
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_scores[int(k)]
        d0 = sorted_scores[f] * (c - k)
        d1 = sorted_scores[c] * (k - f)
        return round(d0 + d1, 4)

    return {
        "min": round(sorted_scores[0], 4),
        "p25": _percentile(25),
        "p50": _percentile(50),
        "p75": _percentile(75),
        "p90": _percentile(90),
        "max": round(sorted_scores[-1], 4),
    }


def avg_score(scores: list[float]) -> float:
    """计算平均分数。"""
    if not scores:
        return 0.0
    return sum(scores) / len(scores)

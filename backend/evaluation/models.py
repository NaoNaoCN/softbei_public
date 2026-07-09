"""
backend/evaluation/models.py
评估数据模型：检索快照、生成快照、评估报告。
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class RetrievalEvalRecord(BaseModel):
    """单次检索的评估快照。"""

    query: str = ""
    kp_name: str = ""
    user_id: str = ""
    session_id: str = ""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    embedding_latency_ms: float = 0.0
    db_query_latency_ms: float = 0.0
    n_candidates: int = 0
    n_results: int = 0
    scores: list[float] = Field(default_factory=list)
    chunk_ids: list[str] = Field(default_factory=list)
    doc_ids: list[str] = Field(default_factory=list)
    chunk_texts: list[str] = Field(default_factory=list)


class GenerationEvalRecord(BaseModel):
    """单次生成的评估快照。"""

    session_id: str = ""
    user_id: str = ""
    agent_type: str = ""
    kp_name: str = ""
    query: str = ""
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # 自动采集字段
    draft_length: int = 0
    generation_latency_ms: float = 0.0
    has_rag_context: bool = False
    n_retrieved: int = 0
    safety_passed: bool = True
    safety_issues_count: int = 0
    safety_issues: list[str] = Field(default_factory=list)  # SafetyAgent 发现的具体问题文本

    # 关联的检索快照
    retrieval_record: Optional[RetrievalEvalRecord] = None

    # LLM-as-Judge 评估结果（异步填充）
    faithfulness_score: Optional[float] = None
    hallucination_rate_val: Optional[float] = Field(default=None, alias="hallucination_rate")
    concept_coverage: Optional[float] = None
    completeness_score: Optional[float] = None

    # 评估详细信息
    relevance_labels: list[int] = Field(default_factory=list)   # Judge 1: 0/1/2
    faithfulness_statements: list[dict] = Field(default_factory=list)  # Judge 2
    completeness_aspects: list[dict] = Field(default_factory=list)     # Judge 3

    model_config = ConfigDict(populate_by_name=True)


class RAGEvalReport(BaseModel):
    """周期性评估报告。"""

    period_start: datetime
    period_end: datetime
    total_queries: int = 0

    # 检索质量汇总
    precision_at_5: float = 0.0
    recall_at_5: float = 0.0
    mrr_val: float = Field(default=0.0, alias="mrr")
    ndcg_at_5: float = 0.0
    hit_rate_val: float = Field(default=0.0, alias="hit_rate")
    score_p50: float = 0.0

    # 生成质量汇总
    avg_faithfulness: float = 0.0
    avg_hallucination_rate: float = 0.0
    avg_concept_coverage: float = 0.0

    # 系统效率
    p50_retrieval_latency_ms: float = 0.0
    p95_retrieval_latency_ms: float = 0.0
    p50_generation_latency_ms: float = 0.0

    # 变化趋势（与上期对比）
    delta_vs_previous: dict[str, float] = Field(default_factory=dict)

    model_config = ConfigDict(populate_by_name=True)


# ===========================================================
# Layer 1: 健康检查
# ===========================================================

class HealthCheckRecord(BaseModel):
    """每次请求的轻量级健康检查指标（不调用 LLM，<5ms）。"""

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_type: str = ""
    kp_name: str = ""

    # 检索健康指标
    n_retrieved: int = 0
    n_empty_results: int = 0          # 1=本次检索为空，0=正常
    score_p50: float = 0.0
    score_min: float = 0.0
    score_max: float = 0.0

    # 延迟（分别真实计时）
    embedding_latency_ms: float = 0.0
    db_query_latency_ms: float = 0.0
    total_retrieval_ms: float = 0.0

    # 生成指标
    draft_length: int = 0
    generation_latency_ms: float = 0.0


# ===========================================================
# Layer 3: 黄金测试集
# ===========================================================

class GoldenQuery(BaseModel):
    """黄金测试集中的单条查询。"""

    id: str = ""                            # 唯一标识
    kp_name: str = ""                       # 知识点名称
    query: str = ""                         # 查询文本
    expected_aspects: list[str] = Field(default_factory=list)  # 应覆盖的知识点方面
    min_faithfulness: float = 0.7           # 最低忠实度期望
    min_completeness: float = 0.6           # 最低完整度期望
    tags: list[str] = Field(default_factory=list)  # 分类标签（如 "definition", "application"）


class GoldenEvalResult(BaseModel):
    """单条黄金查询的评估结果。"""

    query_id: str
    kp_name: str
    query: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Judge 评分 — 检索层
    precision_at_5: float = 0.0
    recall_at_5: float = 0.0
    ndcg_at_5: float = 0.0
    relevance_labels: list[int] = Field(default_factory=list)  # Judge 1: 0/1/2

    # Judge 评分 — 生成层
    faithfulness_score: float = 0.0
    completeness_score: float = 0.0
    hallucination_rate: float = 0.0           # 1.0 - faithfulness
    citation_precision: Optional[float] = None  # Judge 4: 引用准确性

    # 是否通过最低标准
    faithfulness_pass: bool = True
    completeness_pass: bool = True

    # 评估耗时
    evaluation_time_ms: float = 0.0


class GoldenRegressionReport(BaseModel):
    """黄金测试集回归报告。"""

    run_timestamp: datetime = Field(default_factory=datetime.utcnow)
    total_queries: int = 0
    passed: int = 0
    failed: int = 0

    # 各指标均值 — 检索层
    avg_precision_at_5: float = 0.0
    avg_recall_at_5: float = 0.0
    avg_ndcg_at_5: float = 0.0
    mrr: float = 0.0                       # Mean Reciprocal Rank（跨查询计算）
    hit_rate_at_5: float = 0.0              # Hit Rate@5（跨查询计算）

    # 各指标均值 — 生成层
    avg_faithfulness: float = 0.0
    avg_completeness: float = 0.0
    avg_hallucination_rate: float = 0.0
    avg_citation_precision: Optional[float] = None

    # 与上次运行的 delta
    delta_faithfulness: float = 0.0
    delta_completeness: float = 0.0
    delta_precision_at_5: float = 0.0

    # 回归告警（任一度量下降 >10%）
    regression_detected: bool = False
    regression_details: list[str] = Field(default_factory=list)

    # 逐条结果
    per_query_results: list[GoldenEvalResult] = Field(default_factory=list)


# ===========================================================
# Layer 4: A/B 实验
# ===========================================================

class ABExperimentResult(BaseModel):
    """A/B 实验对比结果。"""

    experiment_name: str = ""
    group_a: str = ""                      # 对照组标签
    group_b: str = ""                      # 实验组标签
    run_timestamp: datetime = Field(default_factory=datetime.utcnow)

    # 各组指标
    group_a_metrics: dict[str, float] = Field(default_factory=dict)
    group_b_metrics: dict[str, float] = Field(default_factory=dict)

    # 各指标 delta 及显著性
    metric_deltas: dict[str, dict] = Field(default_factory=dict)
    # 格式: {"faithfulness": {"delta": 0.05, "pct_change": 7.1, "significant": true}}

    conclusion: str = ""                   # 一句话总结


# ===========================================================
# Analysis Models
# ===========================================================

class AnalysisSuggestion(BaseModel):
    """单条改进建议。"""
    priority: str = "medium"        # "high" | "medium" | "low"
    category: str = ""              # "retrieval" | "generation" | "prompt" | "kb" | "latency"
    action: str = ""                # 具体操作建议（一句话）
    rationale: str = ""             # 为什么这条建议（关联到哪个瓶颈）


class KPBreakdown(BaseModel):
    """单个知识点的评估分解。"""
    kp_name: str = ""
    avg_faithfulness: float = 0.0
    avg_completeness: float = 0.0
    avg_precision_at_5: float = 0.0
    pass_count: int = 0
    fail_count: int = 0
    query_ids: list[str] = Field(default_factory=list)


class RAGEvalAnalysisReport(BaseModel):
    """一次完整的评估分析结果。"""

    analysis_type: str = "golden_run"       # "golden_run" | "periodic" | "manual"
    source_reference: str = ""              # 数据来源引用（文件路径或时间范围）
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Aggregate scores
    avg_faithfulness: float = 0.0
    avg_completeness: float = 0.0
    avg_precision_at_5: float = 0.0
    pass_rate: float = 0.0
    total_queries: int = 0

    # Analysis results
    bottlenecks: list[str] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    suggestions: list[AnalysisSuggestion] = Field(default_factory=list)
    key_findings: list[str] = Field(default_factory=list)
    per_kp_breakdown: dict[str, KPBreakdown] = Field(default_factory=dict)
    trend: str = "stable"

    # Flexible overflow
    extra_metadata: dict = Field(default_factory=dict)

    model_config = ConfigDict(populate_by_name=True)
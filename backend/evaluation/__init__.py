"""
backend/evaluation/__init__.py
RAG 效果评估与量化系统（四层递进体系）。
"""

from backend.evaluation.models import (
    RetrievalEvalRecord,
    GenerationEvalRecord,
    RAGEvalReport,
    HealthCheckRecord,
    GoldenQuery,
    GoldenEvalResult,
    GoldenRegressionReport,
    ABExperimentResult,
    AnalysisSuggestion,
    KPBreakdown,
    RAGEvalAnalysisReport,
)
from backend.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    mrr,
    ndcg_at_k,
    hit_rate,
    hallucination_rate,
    score_distribution,
)
from backend.evaluation.judge import RAGJudge
from backend.evaluation.collector import RAGEvalCollector, collector
from backend.evaluation.reporter import RAGReporter
from backend.evaluation.health_check import HealthChecker, health_checker
from backend.evaluation.analyzer import RAGAnalyzer

__all__ = [
    # models
    "RetrievalEvalRecord",
    "GenerationEvalRecord",
    "RAGEvalReport",
    "HealthCheckRecord",
    "GoldenQuery",
    "GoldenEvalResult",
    "GoldenRegressionReport",
    "ABExperimentResult",
    "AnalysisSuggestion",
    "KPBreakdown",
    "RAGEvalAnalysisReport",
    # metrics
    "precision_at_k",
    "recall_at_k",
    "mrr",
    "ndcg_at_k",
    "hit_rate",
    "hallucination_rate",
    "score_distribution",
    # judge
    "RAGJudge",
    # collector
    "RAGEvalCollector",
    "collector",
    # reporter
    "RAGReporter",
    # health check
    "HealthChecker",
    "health_checker",
    # analyzer
    "RAGAnalyzer",
]

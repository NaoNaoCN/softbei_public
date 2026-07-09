"""
tests/conftest.py
Shared fixtures and mock utilities.
"""

from __future__ import annotations

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch


# ============================================================
# Evaluation sample data fixtures
# ============================================================

@pytest.fixture
def sample_scores():
    return [0.95, 0.88, 0.82, 0.55, 0.21]


@pytest.fixture
def sample_relevance_labels():
    return [2, 2, 1, 0, 2]


@pytest.fixture
def sample_relevance_labels_all():
    return [
        [2, 2, 1, 0, 2],
        [1, 0, 0, 2, 0],
        [2, 2, 2, 1, 0],
    ]


@pytest.fixture
def sample_statements():
    return [
        {"text": "gradient descent is an iterative optimization algorithm", "verdict": "supported", "evidence": "gradient descent is..."},
        {"text": "learning rate is usually set to 0.01", "verdict": "supported", "evidence": "learning rate..."},
        {"text": "gradient descent was invented by Newton in 1687", "verdict": "unsupported", "evidence": None},
        {"text": "Adam is the most commonly used optimizer", "verdict": "supported", "evidence": "Adam optimizer..."},
    ]


@pytest.fixture
def sample_faithfulness_result(sample_statements):
    return {
        "statements": sample_statements,
        "faithfulness": 0.75,
        "issues": ["statement about Newton has no evidence in references"],
    }


@pytest.fixture
def sample_completeness_result():
    return {
        "aspects": [
            {"aspect": "definition", "coverage": "covered", "evidence": "gradient descent is a..."},
            {"aspect": "math principles", "coverage": "covered", "evidence": "update formula is..."},
            {"aspect": "learning rate selection", "coverage": "partial", "evidence": "learning rate..."},
            {"aspect": "application scenarios", "coverage": "covered", "evidence": "in deep learning..."},
            {"aspect": "common pitfalls", "coverage": "missing", "evidence": None},
        ],
        "completeness": 0.70,
    }


@pytest.fixture
def sample_retrieval_record(sample_scores):
    from backend.evaluation.models import RetrievalEvalRecord

    return RetrievalEvalRecord(
        query="what is gradient descent",
        kp_name="gradient descent",
        user_id="1001",
        session_id="5001",
        embedding_latency_ms=120.0,
        db_query_latency_ms=45.0,
        n_candidates=10,
        n_results=5,
        scores=sample_scores,
        chunk_ids=["doc_a_0", "doc_a_1", "doc_b_0", "doc_b_1", "doc_c_0"],
        doc_ids=["doc_a", "doc_a", "doc_b", "doc_b", "doc_c"],
        chunk_texts=[
            "gradient descent is a first-order iterative optimization algorithm...",
            "variants of gradient descent include SGD, Adam...",
            "learning rate is a key hyperparameter of gradient descent...",
            "in deep learning, gradient descent is used for...",
            "the difference between Newton's method and gradient descent...",
        ],
    )


@pytest.fixture
def sample_generation_record(sample_retrieval_record, sample_faithfulness_result, sample_completeness_result, sample_relevance_labels):
    from backend.evaluation.models import GenerationEvalRecord

    return GenerationEvalRecord(
        session_id="5001",
        user_id="1001",
        agent_type="doc_agent",
        kp_name="gradient descent",
        query="what is gradient descent",
        draft_length=2500,
        generation_latency_ms=3200.0,
        has_rag_context=True,
        n_retrieved=5,
        safety_passed=True,
        safety_issues_count=0,
        retrieval_record=sample_retrieval_record,
        faithfulness_score=0.85,
        hallucination_rate=0.15,
        concept_coverage=0.70,
        completeness_score=0.70,
        relevance_labels=sample_relevance_labels,
        faithfulness_statements=sample_faithfulness_result["statements"],
        completeness_aspects=sample_completeness_result["aspects"],
    )


@pytest.fixture
def sample_generation_records(sample_generation_record):
    r1 = sample_generation_record
    r2 = r1.model_copy(deep=True)
    r2.timestamp = datetime.utcnow() - timedelta(hours=12)
    r2.faithfulness_score = 0.70
    r2.hallucination_rate_val = 0.30
    r3 = r1.model_copy(deep=True)
    r3.timestamp = datetime.utcnow() - timedelta(hours=6)
    r3.faithfulness_score = 0.92
    r3.hallucination_rate_val = 0.08
    return [r1, r2, r3]


# ============================================================
# Mock LLM fixtures
# ============================================================

class MockLLMResponse:
    """Configurable mock LLM response, simulating chat_completion return values."""

    def __init__(self, responses: list[str] | None = None, default: str = ""):
        self.responses = responses or []
        self.default = default
        self._call_count = 0
        self.calls: list[dict] = []

    def respond(self):
        idx = self._call_count
        self._call_count += 1
        if idx < len(self.responses):
            return self.responses[idx]
        return self.default

    def __call__(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})
        return self.respond()


@pytest.fixture
def mock_chat_completion():
    return AsyncMock()


@pytest.fixture
def patch_chat_completion(mock_chat_completion):
    with patch("backend.services.llm.chat_completion", mock_chat_completion):
        yield mock_chat_completion

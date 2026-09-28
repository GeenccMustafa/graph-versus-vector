"""Unit tests for the answer-quality and retrieval-coverage metrics."""

from __future__ import annotations

import pytest

from rag_compare.data import QAExample
from rag_compare.evaluation import (
    EvalSummary,
    exact_match,
    f1_score,
    normalize_answer,
    score_result,
    support_recall,
)
from rag_compare.schema import RAGResult, RetrievalItem


def _example(**kwargs) -> QAExample:
    base = {"question": "q", "answer": "gold"}
    base.update(kwargs)
    return QAExample(**base)


def test_normalize_answer_strips_case_articles_punctuation() -> None:
    assert normalize_answer("The Quick, Brown Fox!") == "quick brown fox"


def test_exact_match_ignores_case_and_punctuation() -> None:
    assert exact_match("Yes!", "yes") == 1.0
    assert exact_match("no", "yes") == 0.0


def test_f1_score_partial_overlap() -> None:
    assert f1_score("new york city", "new york city") == pytest.approx(1.0)
    assert f1_score("new york", "new york city") == pytest.approx(0.8)
    assert f1_score("", "new york") == 0.0


def test_support_recall_counts_gold_passages() -> None:
    result = RAGResult(
        question="q",
        answer="a",
        retrieval=[RetrievalItem(id="1", text="Alpha passage", score=1.0, source="Alpha")],
    )
    example = _example(supporting=[("Alpha", 0), ("Beta", 0)])
    recall, all_found = support_recall(result, example)
    assert recall == pytest.approx(0.5)
    assert all_found is False


def test_support_recall_without_gold_is_trivially_complete() -> None:
    result = RAGResult(question="q", answer="a")
    recall, all_found = support_recall(result, _example())
    assert recall == 1.0
    assert all_found is True


def test_score_result_populates_record() -> None:
    result = RAGResult(question="q", answer="gold", latency_s=1.5, prompt_tokens=2,
                       completion_tokens=3)
    record = score_result(result, _example())
    assert record.em == 1.0
    assert record.total_tokens == 5
    assert record.num_context_items == 0


def test_eval_summary_aggregates_means() -> None:
    summary = EvalSummary("test")
    summary.add(score_result(RAGResult(question="q", answer="gold"), _example()))
    summary.add(score_result(RAGResult(question="q", answer="wrong"), _example()))
    data = summary.as_dict()
    assert data["n"] == 2
    assert data["exact_match"] == pytest.approx(0.5)
    assert data["f1"] == pytest.approx(0.5)


def test_eval_summary_empty_is_zero() -> None:
    assert EvalSummary("empty").as_dict()["f1"] == 0.0

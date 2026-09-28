"""Offline tests for dense, BM25 and hybrid retrieval in TraditionalRAG."""

from __future__ import annotations

import numpy as np

from rag_compare.bm25_index import BM25Index
from rag_compare.traditional_rag import TraditionalRAG
from rag_compare.vectorstore import VectorIndex

META = [
    {"id": "a", "doc_id": "d", "title": "A", "text": "Ed Wood film directed by Tim Burton"},
    {"id": "b", "doc_id": "d", "title": "B", "text": "The Apollo program landed on the moon"},
    {"id": "c", "doc_id": "d", "title": "C", "text": "unrelated filler words"},
]
MATRIX = np.array(
    [[0.9, 0.1, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0], [0.1, 0.0, 0.0, 0.0]],
    dtype=np.float32,
)


def _rag(settings, fake_llm) -> TraditionalRAG:
    rag = TraditionalRAG(settings, fake_llm)
    rag.index = VectorIndex()
    rag.index.build(MATRIX, META)
    rag.bm25 = BM25Index()
    return rag


def test_dense_retrieval_uses_query_vector(settings, fake_llm) -> None:
    rag = _rag(settings, fake_llm)
    ids = [item.id for item in rag.retrieve("anything", 2, mode="dense")]
    assert ids == ["a", "b"]


def test_bm25_retrieval_uses_terms(settings, fake_llm) -> None:
    rag = _rag(settings, fake_llm)
    ids = [item.id for item in rag.retrieve("Apollo moon", 2, mode="bm25")]
    assert ids[0] == "b"


def test_hybrid_fuses_both_rankings(settings, fake_llm) -> None:
    rag = _rag(settings, fake_llm)
    ids = [item.id for item in rag.retrieve("Apollo moon", 2, mode="hybrid")]
    assert {"a", "b"}.issubset(set(ids))


def test_unknown_mode_falls_back_to_dense(settings, fake_llm) -> None:
    rag = _rag(settings, fake_llm)
    ids = [item.id for item in rag.retrieve("anything", 2, mode="nonsense")]
    assert ids == ["a", "b"]


def test_answer_returns_result(settings, fake_llm) -> None:
    rag = _rag(settings, fake_llm)
    result = rag.answer("Who directed Ed Wood?", 2, mode="bm25")
    assert result.answer == "answer"
    assert result.retrieval

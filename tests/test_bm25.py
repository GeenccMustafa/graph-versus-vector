"""Unit tests for the BM25 lexical index."""

from __future__ import annotations

from rag_compare.bm25_index import BM25Index, tokenize


def test_tokenize_lowercases_and_splits() -> None:
    assert tokenize("Who Directed 'Ed Wood'?") == ["who", "directed", "ed", "wood"]


def test_search_ranks_exact_term_match_first() -> None:
    index = BM25Index()
    index.build(
        [
            "Ed Wood is a 1994 biographical film directed by Tim Burton.",
            "Scott Derrickson is an American film director.",
            "The Saturn V rocket launched the Apollo program.",
        ]
    )
    hits = index.search("Who directed Ed Wood?", k=2)
    assert hits
    assert hits[0].index == 0
    assert hits[0].score > 0


def test_search_before_build_returns_empty() -> None:
    assert BM25Index().search("anything", k=3) == []


def test_search_caps_k_at_corpus_size() -> None:
    index = BM25Index()
    index.build(["only document here"])
    assert len(index.search("document", k=10)) == 1

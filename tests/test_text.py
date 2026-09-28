"""Unit tests for chunking and corpus fingerprinting."""

from __future__ import annotations

from rag_compare.data import Document
from rag_compare.text import build_chunks, chunk_text, corpus_fingerprint


def test_chunk_text_returns_short_text_unchanged() -> None:
    assert chunk_text("Hello world.", size=100, overlap=10) == ["Hello world."]


def test_chunk_text_splits_and_overlaps() -> None:
    sentence = "This is a fairly long sentence with several words in it."
    text = " ".join([sentence] * 8)
    chunks = chunk_text(text, size=120, overlap=40)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)


def test_build_chunks_assigns_stable_ids() -> None:
    docs = [Document(doc_id="doc", title="Doc", text="One. Two. Three.")]
    chunks = build_chunks(docs, size=1000, overlap=0)
    assert [c.chunk_id for c in chunks] == ["doc::0"]
    assert chunks[0].render() == "[Doc] One. Two. Three."


def test_corpus_fingerprint_is_stable_and_order_independent() -> None:
    docs = [Document(doc_id="a", title="A", text="x"), Document(doc_id="b", title="B", text="y")]
    chunks = build_chunks(docs, size=1000, overlap=0)
    assert corpus_fingerprint(chunks) == corpus_fingerprint(list(reversed(chunks)))
    assert len(corpus_fingerprint(chunks)) == 16


def test_corpus_fingerprint_changes_with_corpus() -> None:
    a = build_chunks([Document(doc_id="a", title="A", text="x")], 1000, 0)
    b = build_chunks([Document(doc_id="b", title="B", text="y")], 1000, 0)
    assert corpus_fingerprint(a) != corpus_fingerprint(b)

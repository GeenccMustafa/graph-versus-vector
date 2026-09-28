"""Chunking utilities shared by both pipelines."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from .data import Document


@dataclass
class Chunk:
    """A retrievable slice of a source document.

    Attributes:
        chunk_id: Stable id, ``"{doc_id}::{index}"``.
        doc_id: Identifier of the parent document.
        title: Document title, used as a prefix when rendering.
        text: The chunk's text content.
    """

    chunk_id: str
    doc_id: str
    title: str
    text: str

    def render(self) -> str:
        """Return the chunk as title-prefixed text for embedding or prompting.

        Returns:
            The chunk text prefixed with its bracketed title.
        """
        return f"[{self.title}] {self.text}"


_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    """Split ``text`` into sentences on terminal punctuation.

    Args:
        text: The text to split.

    Returns:
        Non-empty, stripped sentences; the original text if none are found.
    """
    parts = [p.strip() for p in _SENT_RE.split(text) if p.strip()]
    return parts or [text.strip()]


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split text into sentence-aware chunks within a character budget.

    Args:
        text: The text to chunk.
        size: Maximum number of characters per chunk.
        overlap: Approximate number of trailing characters to repeat between
            consecutive chunks.

    Returns:
        The list of chunk strings.
    """
    if len(text) <= size:
        return [text]
    sentences = _split_sentences(text)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for sent in sentences:
        if current_len + len(sent) + 1 > size and current:
            chunks.append(" ".join(current))
            # carry a small overlap of trailing sentences
            carry: list[str] = []
            carry_len = 0
            for s in reversed(current):
                if carry_len + len(s) > overlap:
                    break
                carry.insert(0, s)
                carry_len += len(s) + 1
            current = carry
            current_len = carry_len
        current.append(sent)
        current_len += len(sent) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks


def corpus_fingerprint(chunks: list[Chunk]) -> str:
    """Return a stable hash of a chunk set, used to detect corpus switches.

    Traditional index directories and the Neo4j graph marker are keyed on this,
    so building a different corpus never silently reuses stale indexes.

    Args:
        chunks: The chunks whose ids define the corpus.

    Returns:
        A short (16 hex character) deterministic fingerprint.
    """
    digest = hashlib.sha256()
    for chunk_id in sorted(c.chunk_id for c in chunks):
        digest.update(chunk_id.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()[:16]


def build_chunks(documents: list[Document], size: int, overlap: int) -> list[Chunk]:
    """Split ``documents`` into sentence-aware, overlapping chunks.

    Args:
        documents: Source documents to chunk.
        size: Maximum number of characters per chunk.
        overlap: Approximate overlap between consecutive chunks.

    Returns:
        One :class:`Chunk` per generated piece, in document order.
    """
    chunks: list[Chunk] = []
    for doc in documents:
        pieces = chunk_text(doc.text, size, overlap)
        for i, piece in enumerate(pieces):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}::{i}",
                    doc_id=doc.doc_id,
                    title=doc.title,
                    text=piece,
                )
            )
    return chunks

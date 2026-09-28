"""Shared, offline test fixtures.

Nothing here touches the network, Neo4j or a real LLM provider: the fake client
returns deterministic values so the pure parts of the pipeline can be tested in
isolation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from rag_compare.config import Settings
from rag_compare.llm import ChatResult


class _NullContext:
    """A context manager that yields ``None`` (stands in for a tracing span)."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc) -> bool:
        return False


class FakeTracer:
    """A no-op stand-in for :class:`rag_compare.tracing.Tracer`."""

    @property
    def enabled(self) -> bool:
        """Return ``False``; the fake never traces."""
        return False

    def span(self, *args, **kwargs) -> _NullContext:
        """Return a no-op context manager."""
        return _NullContext()

    def generation(self, *args, **kwargs) -> _NullContext:
        """Return a no-op context manager."""
        return _NullContext()

    def embedding(self, *args, **kwargs) -> _NullContext:
        """Return a no-op context manager."""
        return _NullContext()

    def update(self, *args, **kwargs) -> None:
        """Ignore updates."""

    def score_current(self, *args, **kwargs) -> None:
        """Ignore scores."""

    def flush(self) -> None:
        """Ignore flushes."""


@dataclass
class FakeLLM:
    """A deterministic, offline stand-in for :class:`LLMClient`.

    Attributes:
        query_vector: The vector returned by :meth:`embed_one`.
        answer: The text returned by :meth:`chat`.
    """

    query_vector: np.ndarray
    answer: str = "answer"

    def __post_init__(self) -> None:
        """Attach a no-op tracer and a call counter."""
        self.tracer = FakeTracer()
        self.call_count = 0

    def embed(self, texts, *, batch_size: int = 32, cache: bool = True) -> np.ndarray:
        """Return one deterministic unit vector per input text."""
        rows = []
        for text in texts:
            vec = np.ones(len(self.query_vector), dtype=np.float32)
            vec += (len(text) % 7) / 10.0
            rows.append(vec)
        return np.asarray(rows, dtype=np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        """Return the configured query vector."""
        return np.asarray(self.query_vector, dtype=np.float32)

    def chat(self, messages, **kwargs) -> ChatResult:
        """Return a fixed answer with trivial token counts."""
        return ChatResult(text=self.answer, prompt_tokens=1, completion_tokens=1)

    def chat_json(self, messages, **kwargs):
        """Return an empty JSON object and a chat result."""
        return {}, ChatResult(text="{}", prompt_tokens=1, completion_tokens=1)


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Return settings isolated to a temporary data/cache directory."""
    return Settings(
        DATA_DIR=str(tmp_path / "data"),
        CACHE_DIR=str(tmp_path / "cache"),
        DOCS_DIR=str(tmp_path / "docs"),
        DEEPINFRA_API_KEY="test-key",
    )


@pytest.fixture
def fake_llm() -> FakeLLM:
    """Return a fake LLM whose query vector points along the first axis."""
    return FakeLLM(query_vector=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))

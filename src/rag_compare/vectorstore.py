"""A minimal in-memory vector index for the traditional RAG baseline.

Embeddings are L2-normalised, so cosine similarity is just a dot product.
Persisted to disk so we do not re-embed the corpus on every run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SearchHit:
    """A single nearest-neighbour result from a vector search.

    Attributes:
        index: Row index of the hit in the embedding matrix.
        score: Cosine similarity between the query and the hit.
        meta: The metadata record stored for this hit.
    """

    index: int
    score: float
    meta: dict


class VectorIndex:
    """An in-memory, disk-persisted matrix of L2-normalised embeddings.

    Attributes:
        matrix: The ``(N, D)`` float32 embedding matrix, or ``None`` if empty.
        meta: The per-row metadata records parallel to ``matrix``.
    """

    def __init__(self) -> None:
        """Create an empty index."""
        self.matrix: np.ndarray | None = None
        self.meta: list[dict] = []

    # ------------------------------------------------------------------ build
    def build(self, embeddings: np.ndarray, meta: list[dict]) -> None:
        """Store the embedding matrix and its parallel metadata records.

        Args:
            embeddings: An ``(N, D)`` array of L2-normalised vectors.
            meta: The ``N`` metadata records aligned with ``embeddings``.
        """
        assert len(embeddings) == len(meta)
        self.matrix = np.asarray(embeddings, dtype=np.float32)
        self.meta = meta

    # ----------------------------------------------------------------- search
    def search(self, query: np.ndarray, k: int) -> list[SearchHit]:
        """Return the ``k`` highest-scoring chunks for ``query``.

        Args:
            query: The query embedding.
            k: Maximum number of hits to return.

        Returns:
            Hits sorted by descending cosine similarity (empty if unbuilt).
        """
        if self.matrix is None or len(self.meta) == 0:
            return []
        query = np.asarray(query, dtype=np.float32).reshape(-1)
        scores = self.matrix @ query
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [
            SearchHit(index=int(i), score=float(scores[i]), meta=self.meta[i])
            for i in top
        ]

    # -------------------------------------------------------------- persistence
    def save(self, directory: Path) -> None:
        """Persist the embedding matrix and metadata under ``directory``.

        Args:
            directory: Target directory; created if it does not exist.
        """
        assert self.matrix is not None
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "embeddings.npy", self.matrix)
        (directory / "meta.json").write_text(json.dumps(self.meta))

    @classmethod
    def load(cls, directory: Path) -> "VectorIndex":
        """Load a previously saved index from ``directory``.

        Args:
            directory: Directory written by :meth:`save`.

        Returns:
            The reconstructed :class:`VectorIndex`.
        """
        idx = cls()
        idx.matrix = np.load(directory / "embeddings.npy")
        idx.meta = json.loads((directory / "meta.json").read_text())
        return idx

    def exists(self, directory: Path) -> bool:
        """Return whether a saved index is present in ``directory``.

        Args:
            directory: Directory to check.

        Returns:
            ``True`` if the embeddings file exists.
        """
        return (directory / "embeddings.npy").exists()

    @property
    def size(self) -> int:
        """Return the number of indexed chunks.

        Returns:
            The row count of the embedding matrix (0 if unbuilt).
        """
        return 0 if self.matrix is None else int(self.matrix.shape[0])

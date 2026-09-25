"""A minimal in-memory vector index for the traditional RAG baseline.

Embeddings are L2-normalised, so cosine similarity is just a dot product.
Persisted to disk so we do not re-embed the corpus on every run.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class SearchHit:
    index: int
    score: float
    meta: dict


class VectorIndex:
    def __init__(self) -> None:
        self.matrix: np.ndarray | None = None
        self.meta: list[dict] = []

    # ------------------------------------------------------------------ build
    def build(self, embeddings: np.ndarray, meta: list[dict]) -> None:
        assert len(embeddings) == len(meta)
        self.matrix = np.asarray(embeddings, dtype=np.float32)
        self.meta = meta

    # ----------------------------------------------------------------- search
    def search(self, query: np.ndarray, k: int) -> list[SearchHit]:
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
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "embeddings.npy", self.matrix)
        (directory / "meta.json").write_text(json.dumps(self.meta))

    @classmethod
    def load(cls, directory: Path) -> "VectorIndex":
        idx = cls()
        idx.matrix = np.load(directory / "embeddings.npy")
        idx.meta = json.loads((directory / "meta.json").read_text())
        return idx

    def exists(self, directory: Path) -> bool:
        return (directory / "embeddings.npy").exists()

    @property
    def size(self) -> int:
        return 0 if self.matrix is None else int(self.matrix.shape[0])

"""A minimal BM25 lexical index for sparse retrieval.

BM25 scores documents by exact term overlap (with saturation and length
normalisation), which complements dense embeddings: it excels when the query
and the evidence share rare tokens such as proper nouns, while embeddings are
better at paraphrase. The two are fused in the hybrid retrieval mode.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase ``text`` and split it into alphanumeric tokens.

    Args:
        text: The text to tokenize.

    Returns:
        The list of lowercase alphanumeric tokens.
    """
    return _TOKEN_RE.findall(text.lower())


@dataclass
class SparseHit:
    """A single BM25 search result.

    Attributes:
        index: Row index of the hit in the indexed corpus.
        score: BM25 score for the hit (larger is more relevant).
    """

    index: int
    score: float


class BM25Index:
    """A BM25 index over chunk texts, built in memory from their metadata.

    Attributes:
        model: The underlying ``BM25Okapi`` model, or ``None`` if unbuilt.
    """

    def __init__(self) -> None:
        """Create an empty BM25 index."""
        self.model: BM25Okapi | None = None

    def build(self, texts: list[str]) -> None:
        """Build the BM25 model from the chunk ``texts``.

        Args:
            texts: The chunk texts, in the same order as the retrieval metadata.
        """
        self.model = BM25Okapi([tokenize(t) for t in texts])

    def search(self, query: str, k: int) -> list[SparseHit]:
        """Return the ``k`` highest-scoring documents for ``query``.

        Args:
            query: The query text.
            k: Maximum number of hits to return.

        Returns:
            Hits sorted by descending BM25 score (empty if unbuilt).
        """
        if self.model is None or self.model.corpus_size == 0:
            return []
        scores = np.asarray(self.model.get_scores(tokenize(query)), dtype=np.float32)
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [SparseHit(index=int(i), score=float(scores[i])) for i in top]

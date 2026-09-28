"""Traditional (flat) RAG baseline with dense, BM25 and hybrid retrieval.

Pipeline: chunk corpus -> embed chunks -> top-k retrieval -> answer. This is the
standard "flat" RAG that most people start with: it has no notion of
relationships between entities, so multi-hop questions are hard when the
connecting evidence is spread across passages.

Retrieval modes:

* ``dense``  -- cosine top-k over embeddings (the original baseline).
* ``bm25``   -- lexical top-k over exact term overlap.
* ``hybrid`` -- fuse the two rankings with reciprocal rank fusion (RRF).
"""

from __future__ import annotations

import logging
import time

from .bm25_index import BM25Index, SparseHit
from .config import Settings, get_settings
from .llm import LLMClient
from .schema import RAGResult, RetrievalItem, build_answer_messages
from .text import Chunk, corpus_fingerprint
from .vectorstore import SearchHit, VectorIndex

logger = logging.getLogger(__name__)

# Reciprocal rank fusion constant (Cormack et al. recommend k=60).
RRF_K = 60

DENSE_MODES = {"dense", "vector", "embeddings"}
BM25_MODES = {"bm25", "lexical", "sparse", "keyword"}
HYBRID_MODES = {"hybrid", "dense+bm25", "sparse+dense", "rrf"}


class TraditionalRAG:
    """Flat vector-RAG baseline with dense, BM25 and hybrid retrieval.

    Attributes:
        settings: The resolved settings in use.
        llm: The shared chat/embedding client.
        index: The dense vector index.
        bm25: The BM25 lexical index (built lazily from the vector metadata).
        store_root: Root directory for persisted indexes.
    """

    def __init__(self, settings: Settings | None = None, llm: LLMClient | None = None):
        """Bind the settings, LLM client and cache location for the index.

        Args:
            settings: Optional settings override.
            llm: Optional shared :class:`LLMClient`.
        """
        self.settings = settings or get_settings()
        self.llm = llm or LLMClient(self.settings)
        self.index = VectorIndex()
        self.bm25 = BM25Index()
        self.store_root = self.settings.cache_dir / "traditional_rag"

    # ------------------------------------------------------------------ build
    def build(self, chunks: list[Chunk], *, force: bool = False) -> None:
        """Embed ``chunks`` and persist the index, reusing a matching cache.

        Args:
            chunks: The corpus chunks to index.
            force: Re-embed even if a matching cached index exists.
        """
        self.bm25 = BM25Index()
        # Index per corpus fingerprint so switching corpora never reuses a
        # stale embedding matrix.
        self.store_dir = self.store_root / corpus_fingerprint(chunks)
        meta = [
            {"id": c.chunk_id, "doc_id": c.doc_id, "title": c.title, "text": c.text}
            for c in chunks
        ]
        if not force and self.index.exists(self.store_dir):
            self.index = VectorIndex.load(self.store_dir)
            if self.index.size == len(chunks):
                logger.info("Traditional RAG index cached (%d chunks)", self.index.size)
                return

        logger.info("Embedding %d chunks for traditional RAG ...", len(chunks))
        embeddings = self.llm.embed([c.render() for c in chunks])
        self.index.build(embeddings, meta)
        self.index.save(self.store_dir)

    # --------------------------------------------------------------- retrieve
    def _ensure_bm25(self) -> None:
        """Build the BM25 model from the current index metadata, if needed."""
        if self.bm25.model is None:
            self.bm25.build([m["text"] for m in self.index.meta])

    def _dense_hits(self, question: str, k: int) -> list[SearchHit]:
        """Return dense vector-search hits for the question.

        Args:
            question: The question to embed and match.
            k: Maximum number of hits to return.

        Returns:
            The vector-search hits, ordered by descending similarity.
        """
        qvec = self.llm.embed_one(question)
        return self.index.search(qvec, k)

    def _sparse_hits(self, question: str, k: int) -> list[SparseHit]:
        """Return BM25 lexical-search hits for the question.

        Args:
            question: The question to tokenize and match.
            k: Maximum number of hits to return.

        Returns:
            The BM25 hits, ordered by descending score.
        """
        self._ensure_bm25()
        return self.bm25.search(question, k)

    @staticmethod
    def _item(meta: dict, score: float, kind: str = "chunk") -> RetrievalItem:
        """Build a retrieval item from stored chunk metadata.

        Args:
            meta: A chunk metadata record.
            score: The retrieval score to attach.
            kind: The provenance tag for the item.

        Returns:
            The assembled :class:`RetrievalItem`.
        """
        return RetrievalItem(
            id=meta["id"],
            text=meta["text"],
            score=score,
            source=meta["title"],
            kind=kind,
        )

    def _retrieve_hybrid(self, question: str, k: int) -> list[RetrievalItem]:
        """Fuse dense and BM25 rankings with reciprocal rank fusion (RRF).

        Args:
            question: The question to retrieve for.
            k: Maximum number of fused items to return.

        Returns:
            The top-``k`` items by summed reciprocal rank.
        """
        fused: dict[int, float] = {}
        for dense_rank, dense_hit in enumerate(self._dense_hits(question, k)):
            fused[dense_hit.index] = fused.get(dense_hit.index, 0.0) + 1.0 / (
                RRF_K + dense_rank + 1
            )
        for sparse_rank, sparse_hit in enumerate(self._sparse_hits(question, k)):
            fused[sparse_hit.index] = fused.get(sparse_hit.index, 0.0) + 1.0 / (
                RRF_K + sparse_rank + 1
            )
        order = sorted(fused, key=lambda i: fused[i], reverse=True)[:k]
        return [self._item(self.index.meta[i], fused[i]) for i in order]

    def retrieve(
        self, question: str, k: int | None = None, *, mode: str = "dense"
    ) -> list[RetrievalItem]:
        """Return the top-``k`` chunks for the given retrieval mode.

        Args:
            question: The question to retrieve for.
            k: Number of chunks; defaults to ``settings.top_k``.
            mode: ``dense``, ``bm25`` or ``hybrid`` (aliases accepted); unknown
                values fall back to ``dense``.

        Returns:
            The retrieved items in descending relevance order.
        """
        k = k or self.settings.top_k
        mode = (mode or "dense").lower()
        if mode in BM25_MODES:
            return [
                self._item(self.index.meta[h.index], h.score)
                for h in self._sparse_hits(question, k)
            ]
        if mode in HYBRID_MODES:
            return self._retrieve_hybrid(question, k)
        if mode not in DENSE_MODES:
            logger.warning("Unknown retrieval mode %r; falling back to 'dense'", mode)
        return [self._item(h.meta, h.score) for h in self._dense_hits(question, k)]

    # ----------------------------------------------------------------- answer
    def answer(
        self, question: str, k: int | None = None, *, mode: str = "dense"
    ) -> RAGResult:
        """Answer ``question`` from the retrieved chunks.

        Args:
            question: The question to answer.
            k: Number of chunks to retrieve; defaults to ``settings.top_k``.
            mode: Retrieval mode: ``dense``, ``bm25`` or ``hybrid``.

        Returns:
            The answer, retrieved evidence and usage stats.
        """
        t0 = time.perf_counter()
        with self.llm.tracer.span(
            "traditional_rag.answer", input=question, metadata={"retrieval": mode}
        ) as span:
            retrieval = self.retrieve(question, k, mode=mode)
            context = "\n\n".join(
                f"[{i+1}] ({item.source}) {item.text}" for i, item in enumerate(retrieval)
            )
            chat = self.llm.chat(
                build_answer_messages(question, context), max_tokens=64, temperature=0.0
            )
            result = RAGResult(
                question=question,
                answer=chat.text.strip(),
                retrieval=retrieval,
                latency_s=time.perf_counter() - t0,
                prompt_tokens=chat.prompt_tokens,
                completion_tokens=chat.completion_tokens,
            )
            self.llm.tracer.update(
                span,
                output=result.answer,
                metadata={
                    "pipeline": "traditional_rag",
                    "retrieval": mode,
                    "context_items": len(retrieval),
                    "latency_s": round(result.latency_s, 3),
                },
            )
        return result

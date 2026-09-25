"""Traditional (vector) RAG baseline.

Pipeline: chunk corpus -> embed chunks -> cosine top-k retrieval -> answer.
This is the standard "flat" RAG that most people start with: it has no notion
of relationships between entities, so multi-hop questions are hard when the
connecting evidence is spread across passages.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from .config import Settings, get_settings
from .llm import LLMClient
from .schema import RAGResult, RetrievalItem, build_answer_messages
from .text import Chunk, corpus_fingerprint
from .vectorstore import VectorIndex

logger = logging.getLogger(__name__)


class TraditionalRAG:
    def __init__(self, settings: Settings | None = None, llm: LLMClient | None = None):
        self.settings = settings or get_settings()
        self.llm = llm or LLMClient(self.settings)
        self.index = VectorIndex()
        self.store_root = self.settings.cache_dir / "traditional_rag"

    # ------------------------------------------------------------------ build
    def build(self, chunks: list[Chunk], *, force: bool = False) -> None:
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
    def retrieve(self, question: str, k: int | None = None) -> list[RetrievalItem]:
        k = k or self.settings.top_k
        qvec = self.llm.embed_one(question)
        hits = self.index.search(qvec, k)
        return [
            RetrievalItem(
                id=h.meta["id"],
                text=h.meta["text"],
                score=h.score,
                source=h.meta["title"],
                kind="chunk",
            )
            for h in hits
        ]

    # ----------------------------------------------------------------- answer
    def answer(self, question: str, k: int | None = None) -> RAGResult:
        t0 = time.perf_counter()
        with self.llm.tracer.span(
            "traditional_rag.answer", input=question
        ) as span:
            retrieval = self.retrieve(question, k)
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
                    "context_items": len(retrieval),
                    "latency_s": round(result.latency_s, 3),
                },
            )
        return result

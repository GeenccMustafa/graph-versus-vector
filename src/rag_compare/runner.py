"""High-level orchestration shared by the CLI and the Prefect flows.

Keeping the logic here means ``rag-compare evaluate`` and the Prefect
``evaluate`` flow run exactly the same code.
"""

from __future__ import annotations

import logging

from .config import Settings, get_settings
from .data import load_dataset
from .evaluation import EvalSummary, score_result
from .graph_rag import GraphRAG
from .llm import LLMClient
from .text import build_chunks
from .traditional_rag import TraditionalRAG

logger = logging.getLogger(__name__)


def load_chunks(source: str | None = None, settings: Settings | None = None):
    """Load the dataset and split it into chunks for both pipelines.

    Args:
        source: Corpus source; ``None`` uses the configured dataset.
        settings: Optional settings override.

    Returns:
        A ``(settings, documents, examples, chunks)`` tuple.
    """
    settings = settings or get_settings()
    documents, examples = load_dataset(settings, source=source)
    chunks = build_chunks(documents, settings.chunk_size, settings.chunk_overlap)
    return settings, documents, examples, chunks


def build_all(
    source: str | None = None, *, force: bool = False, settings: Settings | None = None
) -> dict:
    """Build the traditional index and the Neo4j knowledge graph.

    Args:
        source: Corpus source; ``None`` uses the configured dataset.
        force: Rebuild from scratch instead of reusing caches.
        settings: Optional settings override.

    Returns:
        A summary dict with document/question/chunk counts, graph counts and
        the number of provider calls made.
    """
    settings, documents, examples, chunks = load_chunks(source, settings)
    llm = LLMClient(settings)

    traditional = TraditionalRAG(settings, llm)
    traditional.build(chunks, force=force)

    with GraphRAG(settings, llm) as graph:
        graph.build(chunks, force=force, corpus_key=source or settings.dataset_name)
        counts = graph.counts()

    summary = {
        "documents": len(documents),
        "questions": len(examples),
        "chunks": len(chunks),
        "graph": counts,
        "llm_calls": llm.call_count,
    }
    logger.info("Build complete: %s", summary)
    return summary


def run_comparison(
    *,
    limit: int | None = None,
    mode: str = "hybrid",
    retrieval: str = "dense",
    k: int | None = None,
    source: str | None = None,
    settings: Settings | None = None,
) -> dict:
    """Run both pipelines over the examples and build a comparison report.

    Args:
        limit: Maximum number of questions to evaluate.
        mode: GraphRAG retrieval mode (``local``, ``global`` or ``hybrid``).
        retrieval: Traditional RAG retrieval mode (``dense``, ``bm25`` or
            ``hybrid``).
        k: Number of chunks to retrieve per pipeline.
        source: Corpus source; ``None`` uses the configured dataset.
        settings: Optional settings override.

    Returns:
        A report dict with ``config``, aggregate ``results`` and
        ``per_question`` records.

    Raises:
        ValueError: If there are no questions to evaluate.
        RuntimeError: If the Neo4j graph is missing or holds another corpus.
    """
    settings, _, examples, chunks = load_chunks(source, settings)
    if limit:
        examples = examples[:limit]
    if not examples:
        raise ValueError(
            "No questions to evaluate. For a custom corpus add a "
            "'corpus_qa.json' next to your documents (see README)."
        )

    llm = LLMClient(settings)
    traditional = TraditionalRAG(settings, llm)
    traditional.build(chunks)

    graph = GraphRAG(settings, llm)
    graph.connect()
    if not graph.is_built:
        raise RuntimeError("Graph not built. Run `rag-compare build` first.")
    if not graph.matches(chunks):
        raise RuntimeError(
            "The Neo4j graph holds a different corpus. Re-run "
            "`rag-compare build --source ...` before evaluating."
        )

    trad_summary = EvalSummary(f"traditional_rag_{retrieval}")
    graph_summary = EvalSummary(f"graph_rag_{mode}")

    with llm.tracer.span(
        "comparison.run", input={"n": len(examples), "mode": mode, "retrieval": retrieval}
    ) as span:
        for i, example in enumerate(examples, 1):
            logger.info("(%d/%d) %s", i, len(examples), example.question)
            trad_summary.add(
                score_result(
                    traditional.answer(example.question, k, mode=retrieval), example
                )
            )
            graph_summary.add(
                score_result(graph.answer(example.question, mode=mode, k=k), example)
            )
        llm.tracer.update(
            span,
            output={
                "traditional": trad_summary.as_dict(),
                "graph": graph_summary.as_dict(),
            },
        )

    graph.close()

    def delta(a: EvalSummary, b: EvalSummary) -> dict:
        """Return metric-wise differences ``b - a`` excluding name and count.

        Args:
            a: The baseline summary.
            b: The compared summary.

        Returns:
            A dict of rounded ``b - a`` deltas per metric.
        """
        ad, bd = a.as_dict(), b.as_dict()
        return {key: round(bd[key] - ad[key], 4) for key in ad if key not in {"name", "n"}}

    report = {
        "config": {
            "dataset": source or settings.dataset_name,
            "split": settings.dataset_split,
            "num_questions": len(examples),
            "llm_model": settings.llm_model,
            "embedding_model": settings.embedding_model,
            "top_k": k or settings.top_k,
            "retrieval_mode": retrieval,
            "graph_mode": mode,
            "graph_hops": settings.graph_hops,
        },
        "results": {
            "traditional_rag": trad_summary.as_dict(),
            "graph_rag": graph_summary.as_dict(),
            "delta (graph - traditional)": delta(trad_summary, graph_summary),
        },
        "per_question": {
            "traditional_rag": [r.__dict__ for r in trad_summary.records],
            "graph_rag": [r.__dict__ for r in graph_summary.records],
        },
    }
    llm.tracer.flush()
    return report


def run_deepeval(
    *,
    limit: int | None = None,
    source: str | None = None,
    both: bool = True,
    metrics: list[str] | None = None,
    retrieval: str = "dense",
    k: int | None = None,
    settings: Settings | None = None,
) -> dict:
    """Run DeepEval LLM-as-judge metrics for both pipelines.

    Args:
        limit: Maximum number of questions to evaluate.
        source: Corpus source; ``None`` uses the configured dataset.
        both: Evaluate both pipelines, otherwise traditional RAG only.
        metrics: Metric names to run; ``None`` uses the configured defaults.
        retrieval: Traditional RAG retrieval mode (``dense``, ``bm25`` or
            ``hybrid``).
        k: Number of chunks to retrieve per pipeline.
        settings: Optional settings override.

    Returns:
        The DeepEval summary and per-question records.

    Raises:
        ValueError: If there are no questions with gold answers.
        RuntimeError: If the Neo4j graph is missing or holds another corpus.
    """
    from .deepeval_eval import build_test_case, evaluate_test_cases

    settings, _, examples, chunks = load_chunks(source, settings)
    if limit:
        examples = examples[:limit]
    if not examples:
        raise ValueError(
            "DeepEval needs gold answers. Add a 'corpus_qa.json' for a custom "
            "corpus, or use the hotpotqa benchmark."
        )

    llm = LLMClient(settings)
    traditional = TraditionalRAG(settings, llm)
    traditional.build(chunks)

    cases: dict[str, list] = {}
    graph = None
    try:
        if both:
            graph = GraphRAG(settings, llm)
            graph.connect()
            if not graph.is_built:
                raise RuntimeError("Graph not built. Run `rag-compare build` first.")
            if not graph.matches(chunks):
                raise RuntimeError(
                    "The Neo4j graph holds a different corpus. Re-run "
                    "`rag-compare build --source ...` first."
                )
        cases["traditional_rag"] = []
        if both:
            cases["graph_rag"] = []
        for example in examples:
            trad = traditional.answer(example.question, k, mode=retrieval)
            cases["traditional_rag"].append(
                build_test_case(example.question, trad, example.answer)
            )
            if both and graph is not None:
                graph_res = graph.answer(example.question, k=k)
                cases["graph_rag"].append(
                    build_test_case(example.question, graph_res, example.answer)
                )
    finally:
        if graph is not None:
            graph.close()

    result = evaluate_test_cases(cases, settings=settings, metric_names=metrics)
    llm.tracer.flush()
    return result

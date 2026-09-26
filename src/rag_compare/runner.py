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
    """Load the dataset and split it into chunks for both pipelines."""
    settings = settings or get_settings()
    documents, examples = load_dataset(settings, source=source)
    chunks = build_chunks(documents, settings.chunk_size, settings.chunk_overlap)
    return settings, documents, examples, chunks


def build_all(
    source: str | None = None, *, force: bool = False, settings: Settings | None = None
) -> dict:
    """Build the traditional index and the Neo4j graph."""
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
    k: int | None = None,
    source: str | None = None,
    settings: Settings | None = None,
) -> dict:
    """Run both pipelines over the examples and return a report dict."""
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

    trad_summary = EvalSummary("traditional_rag")
    graph_summary = EvalSummary(f"graph_rag_{mode}")

    with llm.tracer.span(
        "comparison.run", input={"n": len(examples), "mode": mode}
    ) as span:
        for i, example in enumerate(examples, 1):
            logger.info("(%d/%d) %s", i, len(examples), example.question)
            trad_summary.add(
                score_result(traditional.answer(example.question, k), example)
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
        """Return metric-wise differences ``b - a`` excluding name and count."""
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
    k: int | None = None,
    settings: Settings | None = None,
) -> dict:
    """Run DeepEval LLM-as-judge metrics for both pipelines."""
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
            trad = traditional.answer(example.question, k)
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

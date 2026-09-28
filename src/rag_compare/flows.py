"""Prefect flows: schedule and observe the GraphRAG-vs-RAG pipeline.

Run a flow directly::

    uv run python -m rag_compare.flows              # full pipeline
    uv run rag-compare flow build                   # build only
    uv run rag-compare flow evaluate                # evaluate only

Start the Prefect UI (http://127.0.0.1:4200) with::

    uv run prefect server start

Flows can also be scheduled/deployed with ``prefect deploy`` — see README.
"""

from __future__ import annotations

import logging

from prefect import flow, task

from .runner import build_all, run_comparison, run_deepeval

logger = logging.getLogger(__name__)


@task(name="build-pipelines", retries=1, retry_delay_seconds=15)
def build_task(source: str | None = None, force: bool = False) -> dict:
    """Prefect task that builds both retrieval pipelines."""
    return build_all(source, force=force)


@task(name="evaluate-comparison", retries=1, retry_delay_seconds=15)
def compare_task(
    limit: int | None = None,
    mode: str = "hybrid",
    retrieval: str = "dense",
    source: str | None = None,
) -> dict:
    """Prefect task that runs the custom-metrics comparison."""
    return run_comparison(limit=limit, mode=mode, retrieval=retrieval, source=source)


@task(name="evaluate-deepeval", retries=1, retry_delay_seconds=15)
def deepeval_task(
    limit: int | None = None,
    source: str | None = None,
    both: bool = True,
    retrieval: str = "dense",
) -> dict:
    """Prefect task that runs the DeepEval LLM-as-judge metrics."""
    return run_deepeval(limit=limit, source=source, both=both, retrieval=retrieval)


@flow(name="rag-compare-build", log_prints=True)
def build_flow(source: str | None = None, force: bool = False) -> dict:
    """Build the traditional index and the Neo4j knowledge graph."""
    summary = build_task(source=source, force=force)
    print(f"Built {summary['chunks']} chunks; graph={summary['graph']}")
    return summary


@flow(name="rag-compare-evaluate", log_prints=True)
def evaluate_flow(
    limit: int | None = None,
    mode: str = "hybrid",
    retrieval: str = "dense",
    source: str | None = None,
) -> dict:
    """Run the head-to-head comparison and print the summary."""
    report = compare_task(limit=limit, mode=mode, retrieval=retrieval, source=source)
    results = report["results"]
    print("Traditional RAG:", results["traditional_rag"])
    print("GraphRAG:       ", results["graph_rag"])
    return report


@flow(name="rag-compare-deepeval", log_prints=True)
def deepeval_flow(
    limit: int | None = None,
    source: str | None = None,
    both: bool = True,
    retrieval: str = "dense",
) -> dict:
    """Run DeepEval LLM-as-judge metrics."""
    result = deepeval_task(limit=limit, source=source, both=both, retrieval=retrieval)
    print("DeepEval summary:", result["summary"])
    return result


@flow(name="rag-compare-pipeline", log_prints=True)
def full_flow(
    source: str | None = None,
    limit: int | None = None,
    mode: str = "hybrid",
    retrieval: str = "dense",
    force: bool = False,
) -> dict:
    """Build both pipelines, then evaluate them (the default end-to-end flow)."""
    build_summary = build_task(source=source, force=force)
    report = compare_task(limit=limit, mode=mode, retrieval=retrieval, source=source)
    return {"build": build_summary, "evaluation": report}


if __name__ == "__main__":
    full_flow()

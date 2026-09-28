"""Command-line interface for the GraphRAG vs traditional-RAG comparison.

Commands
--------
    uv run rag-compare config                 # show resolved settings
    uv run rag-compare download               # fetch + cache the benchmark
    uv run rag-compare build --force          # build both indexes
    uv run rag-compare ask "..."              # ask one question, both answers
    uv run rag-compare evaluate               # custom metrics benchmark
    uv run rag-compare deepeval               # LLM-as-judge metrics
    uv run rag-compare graph                   # inspect the Neo4j graph
    uv run rag-compare flow all               # run via Prefect

Use ``--source hotpotqa`` (default) or ``--source files`` to run against your
own markdown/text documents in ``DOCS_DIR``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import get_settings
from .data import load_dataset
from .graph_rag import GraphRAG
from .llm import LLMClient
from .runner import build_all, run_comparison, run_deepeval
from .text import build_chunks
from .tracing import langfuse_reachable
from .traditional_rag import TraditionalRAG

app = typer.Typer(add_completion=False, help="GraphRAG vs traditional RAG comparison.")
console = Console()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
# Neo4j emits noisy "label does not exist" warnings while a graph is being
# populated for the first time; they are not actionable here.
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

SOURCE_OPTION = typer.Option(
    None, "--source", "-s", help="Corpus source: 'hotpotqa' or 'files'."
)

RETRIEVAL_OPTION = typer.Option(
    "dense",
    "--retrieval",
    "-r",
    help="Traditional RAG retrieval: dense|bm25|hybrid.",
)


def _langfuse_status(s) -> str:
    """Return a human-readable Langfuse tracing status for the config table.

    Args:
        s: The resolved settings.

    Returns:
        A short status string describing whether tracing is active.
    """
    if not s.tracing_enabled:
        return "disabled (no keys)"
    if langfuse_reachable(s.langfuse_host):
        return f"enabled -> {s.langfuse_host}"
    return f"configured, host unreachable ({s.langfuse_host})"


def _load_chunks(source: str | None = None):
    """Load dataset and chunks using the cached global settings.

    Args:
        source: Corpus source; ``None`` uses the configured dataset.

    Returns:
        A ``(settings, documents, examples, chunks)`` tuple.
    """
    settings = get_settings()
    documents, examples = load_dataset(settings, source=source)
    chunks = build_chunks(documents, settings.chunk_size, settings.chunk_overlap)
    return settings, documents, examples, chunks


@app.command()
def config() -> None:
    """Show the resolved configuration and which integrations are active."""
    s = get_settings()
    table = Table(title="rag-compare configuration")
    table.add_column("Setting", style="bold")
    table.add_column("Value")
    rows = [
        ("dataset_name", s.dataset_name),
        ("docs_dir", str(s.docs_dir)),
        ("data_dir", str(s.data_dir)),
        ("cache_dir", str(s.cache_dir)),
        ("llm_model", s.llm_model),
        ("extractor_model", s.extractor_model),
        ("embedding_model", f"{s.embedding_model} ({s.embedding_dim}d)"),
        ("neo4j_uri", s.neo4j_uri),
        ("chunk_size / overlap", f"{s.chunk_size} / {s.chunk_overlap}"),
        ("top_k / graph_hops", f"{s.top_k} / {s.graph_hops}"),
        ("deepinfra key set", "yes" if s.deepinfra_api_key else "NO"),
        ("langfuse tracing", _langfuse_status(s)),
        ("deepeval metrics", ", ".join(s.deepeval_metric_list)),
    ]
    for key, value in rows:
        table.add_row(key, value)
    console.print(table)


@app.command()
def download(source: str = SOURCE_OPTION) -> None:
    """Download and cache the multi-hop QA dataset (or scan local files).

    Args:
        source: Corpus source; ``None`` uses the configured dataset.
    """
    settings = get_settings()
    documents, examples = load_dataset(settings, source=source)
    levels: dict[str, int] = {}
    for ex in examples:
        levels[ex.level] = levels.get(ex.level, 0) + 1
    console.print(
        Panel.fit(
            f"[bold]{len(documents)}[/] documents\n"
            f"[bold]{len(examples)}[/] QA examples ({levels})\n"
            f"source: {source or settings.dataset_name}\n"
            f"data dir: {settings.data_dir}",
            title="Corpus ready",
        )
    )


@app.command()
def build(
    source: str = SOURCE_OPTION,
    force: bool = typer.Option(False, help="Rebuild indexes from scratch."),
) -> None:
    """Build the traditional RAG index and the Neo4j knowledge graph.

    Args:
        source: Corpus source; ``None`` uses the configured dataset.
        force: Rebuild indexes from scratch instead of reusing caches.
    """
    summary = build_all(source, force=force)
    table = Table(title="Neo4j graph")
    table.add_column("Node / Edge")
    table.add_column("Count", justify="right")
    for key, value in summary["graph"].items():
        table.add_row(key, str(value))
    console.print(table)
    console.print(
        f"[green]Done.[/] {summary['documents']} documents, "
        f"{summary['chunks']} chunks, LLM calls: {summary['llm_calls']} "
        f"(cached responses are free)"
    )


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask both systems."),
    source: str = SOURCE_OPTION,
    mode: str = typer.Option("hybrid", help="GraphRAG mode: local|global|hybrid."),
    retrieval: str = RETRIEVAL_OPTION,
    k: int = typer.Option(None, help="Number of chunks to retrieve."),
    show_context: bool = typer.Option(False, "--context", help="Show retrieved context."),
) -> None:
    """Ask a single question and compare the two systems side by side.

    Args:
        question: The question to ask both systems.
        source: Corpus source; ``None`` uses the configured dataset.
        mode: GraphRAG retrieval mode (``local``, ``global`` or ``hybrid``).
        retrieval: Traditional RAG retrieval mode (``dense``, ``bm25`` or
            ``hybrid``).
        k: Number of chunks to retrieve; defaults to ``TOP_K``.
        show_context: Print the retrieved context for both pipelines.
    """
    settings, _, _, chunks = _load_chunks(source)
    llm = LLMClient(settings)

    traditional = TraditionalRAG(settings, llm)
    traditional.build(chunks)
    with GraphRAG(settings, llm) as graph:
        if not graph.is_built:
            console.print("[yellow]Graph not built; run `build` first.[/]")
            raise typer.Exit(1)
        if not graph.matches(chunks):
            console.print(
                "[yellow]Neo4j holds a different corpus; run "
                f"`rag-compare build --source {source or settings.dataset_name}`.[/]"
            )
            raise typer.Exit(1)
        trad = traditional.answer(question, k, mode=retrieval)
        graph_res = graph.answer(question, mode=mode, k=k)
    llm.tracer.flush()

    table = Table(title=f"Q: {question}", show_lines=True)
    table.add_column("", style="bold", no_wrap=True)
    table.add_column(f"Traditional RAG ({retrieval})", overflow="fold")
    table.add_column(f"GraphRAG ({mode})", overflow="fold")
    table.add_row("Answer", trad.answer, graph_res.answer)
    table.add_row("Latency", f"{trad.latency_s:.2f}s", f"{graph_res.latency_s:.2f}s")
    table.add_row("Tokens", str(trad.total_tokens), str(graph_res.total_tokens))
    table.add_row(
        "Context items", str(len(trad.retrieval)), str(len(graph_res.retrieval))
    )
    console.print(table)

    if graph_res.graph_facts:
        console.print(
            Panel(
                "\n".join(f"- {f}" for f in graph_res.graph_facts[:15]),
                title="Graph facts used",
            )
        )
    if show_context:
        console.print(Panel(trad.context_text[:2000], title="Traditional context"))
        console.print(Panel(graph_res.context_text[:2000], title="GraphRAG context"))


@app.command()
def evaluate(
    source: str = SOURCE_OPTION,
    limit: int = typer.Option(None, help="Number of questions to evaluate."),
    mode: str = typer.Option("hybrid", help="GraphRAG mode: local|global|hybrid."),
    retrieval: str = RETRIEVAL_OPTION,
    k: int = typer.Option(None, help="Number of chunks to retrieve."),
    out: Path = typer.Option(Path("results/comparison.json"), help="Report output path."),
) -> None:
    """Run the custom-metrics benchmark and print a comparison report.

    Args:
        source: Corpus source; ``None`` uses the configured dataset.
        limit: Maximum number of questions to evaluate.
        mode: GraphRAG retrieval mode (``local``, ``global`` or ``hybrid``).
        retrieval: Traditional RAG retrieval mode (``dense``, ``bm25`` or
            ``hybrid``).
        k: Number of chunks to retrieve; defaults to ``TOP_K``.
        out: Where to write the JSON report.
    """
    report = run_comparison(limit=limit, mode=mode, retrieval=retrieval, k=k, source=source)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    _print_comparison(report)
    console.print(f"[green]Report written to[/] {out}")


def _print_comparison(report: dict) -> None:
    """Print the side-by-side comparison table for an evaluation report.

    Args:
        report: The report dict returned by :func:`runner.run_comparison`.
    """
    trad = report["results"]["traditional_rag"]
    graph = report["results"]["graph_rag"]
    retrieval = report.get("config", {}).get("retrieval_mode", "dense")
    rows = [
        ("Exact match", "exact_match"),
        ("F1", "f1"),
        ("Supporting-fact recall", "support_recall"),
        ("All supporting facts found", "support_all"),
        ("Latency (s)", "avg_latency_s"),
        ("Tokens", "avg_tokens"),
        ("Context items", "avg_context_items"),
    ]
    table = Table(title=f"GraphRAG vs Traditional RAG [{retrieval}] (n={trad['n']})")
    table.add_column("Metric", style="bold")
    table.add_column("Traditional RAG", justify="right")
    table.add_column("GraphRAG", justify="right")
    table.add_column("Δ", justify="right")
    for label, key in rows:
        a, b = trad[key], graph[key]
        delta = b - a
        color = "green" if delta > 0 else ("red" if delta < 0 else "white")
        table.add_row(label, f"{a:.4f}", f"{b:.4f}", f"[{color}]{delta:+.4f}[/]")
    console.print(table)


@app.command()
def deepeval(
    source: str = SOURCE_OPTION,
    limit: int = typer.Option(5, help="Number of questions (LLM-judge is costly)."),
    metrics: str = typer.Option(
        None, help="Comma-separated metric names (default from .env)."
    ),
    both: bool = typer.Option(True, "--both/--graph-only", help="Evaluate both pipelines."),
    retrieval: str = RETRIEVAL_OPTION,
    k: int = typer.Option(None, help="Number of chunks to retrieve."),
    out: Path = typer.Option(Path("results/deepeval.json"), help="Report output path."),
) -> None:
    """Run DeepEval LLM-as-judge metrics for RAG quality.

    Args:
        source: Corpus source; ``None`` uses the configured dataset.
        limit: Maximum number of questions (the LLM judge is costly).
        metrics: Comma-separated metric names; ``None`` uses the configured
            defaults.
        both: Evaluate both pipelines, otherwise traditional RAG only.
        retrieval: Traditional RAG retrieval mode (``dense``, ``bm25`` or
            ``hybrid``).
        k: Number of chunks to retrieve; defaults to ``TOP_K``.
        out: Where to write the JSON report.
    """
    metric_names = [m.strip() for m in metrics.split(",")] if metrics else None
    result = run_deepeval(
        limit=limit,
        source=source,
        both=both,
        metrics=metric_names,
        retrieval=retrieval,
        k=k,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    table = Table(title="DeepEval LLM-as-judge metrics")
    table.add_column("Pipeline", style="bold")
    table.add_column("Metric")
    table.add_column("Score", justify="right")
    table.add_column("Success rate", justify="right")
    for pipeline, stats in result["summary"].items():
        for metric, values in stats.items():
            table.add_row(
                pipeline,
                metric,
                f"{values['score']:.4f}",
                f"{values['success_rate']:.2%}",
            )
    console.print(table)
    console.print(f"[green]Report written to[/] {out}")


@app.command()
def flow(
    command: str = typer.Argument("all", help="build | evaluate | deepeval | all"),
    source: str = SOURCE_OPTION,
    limit: int = typer.Option(None, help="Number of questions to evaluate."),
    mode: str = typer.Option("hybrid", help="GraphRAG mode: local|global|hybrid."),
    retrieval: str = RETRIEVAL_OPTION,
    force: bool = typer.Option(False, help="Force rebuild."),
) -> None:
    """Run the pipeline through Prefect.

    Start the UI with ``uv run prefect server start``.

    Args:
        command: Which flow to run: ``build``, ``evaluate``, ``deepeval`` or
            ``all``.
        source: Corpus source; ``None`` uses the configured dataset.
        limit: Maximum number of questions to evaluate.
        mode: GraphRAG retrieval mode (``local``, ``global`` or ``hybrid``).
        retrieval: Traditional RAG retrieval mode (``dense``, ``bm25`` or
            ``hybrid``).
        force: Force a rebuild for the build/all flows.
    """
    from . import flows

    command = command.lower()
    if command == "build":
        flows.build_flow(source=source, force=force)
    elif command == "evaluate":
        flows.evaluate_flow(limit=limit, mode=mode, retrieval=retrieval, source=source)
    elif command == "deepeval":
        flows.deepeval_flow(limit=limit, source=source, retrieval=retrieval)
    elif command == "all":
        flows.full_flow(
            source=source, limit=limit, mode=mode, retrieval=retrieval, force=force
        )
    else:
        console.print(f"[red]Unknown flow {command!r}[/] (use build/evaluate/deepeval/all)")
        raise typer.Exit(1)


@app.command()
def graph(
    limit: int = typer.Option(15, help="Max rows to display per section."),
) -> None:
    """Inspect entities, relationships and communities in Neo4j.

    Args:
        limit: Maximum rows/cards to display per section.
    """
    settings = get_settings()
    with GraphRAG(settings) as gr:
        if not gr.is_built:
            console.print("[yellow]Graph not built; run `build` first.[/]")
            raise typer.Exit(1)
        counts = gr.counts()
        console.print(
            Panel.fit(
                "\n".join(f"{k}: [bold]{v}[/]" for k, v in counts.items()),
                title="Graph size",
            )
        )

        rels = gr._run(
            """
            MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
            RETURN a.display_name AS source, r.description AS rel, b.display_name AS target,
                   r.weight AS weight
            ORDER BY r.weight DESC LIMIT $limit
            """,
            limit=limit,
        )
        table = Table(title="Top relationships")
        table.add_column("Source")
        table.add_column("Relationship")
        table.add_column("Target")
        table.add_column("w", justify="right")
        for r in rels:
            table.add_row(
                r["source"] or "", r["rel"] or "", r["target"] or "", str(r["weight"])
            )
        console.print(table)

        comms = gr._run(
            """
            MATCH (cm:Community)
            RETURN cm.title AS title, cm.size AS size, cm.summary AS summary
            ORDER BY cm.size DESC LIMIT $limit
            """,
            limit=min(limit, 5),
        )
        for c in comms:
            console.print(
                Panel(
                    c["summary"] or "",
                    title=f"Community ({c['size']} entities): {c['title']}",
                )
            )


if __name__ == "__main__":
    app()

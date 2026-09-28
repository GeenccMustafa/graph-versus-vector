# GraphRAG vs Traditional RAG

A hands-on comparison of two retrieval-augmented generation pipelines, plus the
production tooling around them:

- **Traditional RAG** — chunk the corpus, then retrieve the top-`k` by **dense**
  (cosine), **BM25** (lexical), or **hybrid** (reciprocal rank fusion) search,
  and feed them to the LLM.
- **GraphRAG** — use an LLM to extract entities and relationships, store them as
  a **knowledge graph in Neo4j** (with vector indexes for chunk/entity/community
  embeddings), then retrieve by *walking the graph* and by *community summaries*.
- **Observability / evaluation / orchestration** — **Langfuse** (tracing),
  **DeepEval** (LLM-as-judge metrics), **Prefect** (flows + UI).

Dataset by default: **HotpotQA** (hard multi-hop QA). You can also drop your own
markdown files into `data/` and run both pipelines over them — see
[Add your own documents](#add-your-own-documents-markdown--text).

---

## Table of contents

1. [What you'll learn](#what-youll-learn)
2. [Architecture](#architecture)
3. [Neo4j schema](#neo4j-schema)
4. [Prerequisites](#prerequisites)
5. [Setup](#setup)
6. [Quickstart (HotpotQA benchmark)](#quickstart-hotpotqa-benchmark)
7. [Add your own documents (markdown / text)](#add-your-own-documents-markdown--text)
8. [CLI reference](#cli-reference)
9. [Integrations: Langfuse, DeepEval, Prefect](#integrations)
10. [File-by-file guide](#file-by-file-guide)
11. [Metrics explained](#metrics-explained)
12. [Results](#results)
13. [Cost](#cost)
14. [Troubleshooting](#troubleshooting)

---

## What you'll learn

Multi-hop questions (e.g. *"Were Scott Derrickson and Ed Wood of the same
nationality?"*) can only be answered by **combining facts from two or more
passages**. Flat top-`k` retrieval often grabs only one of the required
passages, because the question embedding doesn't point at both. GraphRAG bridges
that gap by following relationships between entities. The bundled benchmark
shows exactly this: GraphRAG raises supporting-fact recall from **0.81 → 0.98**.

---

## Architecture

Both pipelines share the same front half (load → chunk) and the same back half
(answer with an LLM, then score the result). They differ only in how they turn a
question into evidence: **Traditional RAG** retrieves text directly, while
**GraphRAG** retrieves over a knowledge graph of entities and relationships.
Everything else is held constant so the comparison is fair.

### Layers

1. **Ingestion** — `data.py` loads the benchmark (HotpotQA) or your own markdown,
   and `text.py` splits documents into sentence-aware, overlapping chunks.
2. **Retrieval** — the two competing pipelines (see below).
3. **Answering** — one shared prompt and chat model turn retrieved evidence into
   a short answer (`schema.py`, `llm.py`).
4. **Cross-cutting** — a single `LLMClient` (on-disk cache + Langfuse hooks),
   `evaluation.py` / `deepeval_eval.py` for scoring, and `runner.py` + `flows.py`
   that let the CLI and Prefect run exactly the same code.

### Request lifecycle

```
                        ┌─────────────────────────────────┐
   corpus ────────────▶ │ load + chunk (sentence-aware)   │
                        └────────────────┬────────────────┘
                                        │  Chunk[]
                ┌───────────────────────┴───────────────────────┐
                ▼                                               ▼
   ┌───────────────────────────┐                   ┌───────────────────────────┐
   │    Traditional RAG        │                   │         GraphRAG          │
   │                           │                   │                           │
   │ BUILD                     │                   │ BUILD                     │
   │  • embed chunks           │                   │  • embed chunks           │
   │  • BM25 index (in-mem)    │                   │  • LLM entity/rel extract │
   │                           │                   │  • write Neo4j triples    │
   │ QUERY                     │                   │  • detect + summarise     │
   │  • dense | BM25 | RRF     │                   │    communities            │
   │  • fuse rankings (k=60)   │                   │                           │
   │                           │                   │ QUERY                     │
   │                           │                   │  • local | global | hybrid│
   └─────────────┬─────────────┘                   └─────────────┬─────────────┘
                │                                               │
                └───────────────────────┬───────────────────────┘
                                        ▼
                        ┌─────────────────────────────────┐
                        │ answer with LLM (same prompt)   │
                        └────────────────┬────────────────┘
                                        ▼
                        ┌─────────────────────────────────┐
                        │ evaluate: EM / F1 / support     │
                        │ recall; DeepEval judges         │
                        │ (Langfuse traces every call)    │
                        └─────────────────────────────────┘
```

### Phase 1 — build (offline, once per corpus)

Both pipelines persist their indexes so repeated runs are cheap and can run
offline:

- **Traditional RAG** embeds every chunk and stores the matrix under
  `.cache/traditional_rag/<corpus-fingerprint>/`. The BM25 index is rebuilt in
  memory from that stored chunk metadata, so no extra artifacts are written.
- **GraphRAG** additionally runs LLM entity/relationship extraction per chunk,
  merges the results, and writes a graph to **Neo4j**: `Document`, `Chunk`,
  `Entity` and `Community` nodes, with `PART_OF`, `MENTIONS`, `RELATES_TO` and
  `IN_COMMUNITY` edges. Communities are detected with networkx and summarised by
  the LLM (used by global search).
- Both are keyed by a **corpus fingerprint** (`text.corpus_fingerprint`), so
  switching corpora never silently reuses a stale index. The Neo4j graph also
  stores a `Corpus` marker node and rebuilds itself when the corpus changes.

### Phase 2 — query (per question)

Each pipeline turns a question into a list of `RetrievalItem`s, which the
**same** prompt and chat model then turn into an answer.

**Traditional RAG retrieval modes** (`--retrieval`):

- `dense` (default) — cosine top-`k` over the embedding matrix.
- `bm25` — lexical top-`k` over exact term overlap (strong on rare tokens and
  proper nouns).
- `hybrid` — fuse dense and BM25 rankings with reciprocal rank fusion
  (`RRF_K = 60`).

**GraphRAG retrieval modes** (`--mode`):

- `local` — vector-seed the top entities, expand `GRAPH_HOPS` hops, and collect
  relationship facts plus the chunks that mention those entities.
- `global` — vector-match LLM-written community summaries (thematic questions).
- `hybrid` (default) — both.

### Where state lives

| Location | What it holds |
|----------|----------------|
| Neo4j | The GraphRAG knowledge graph (chunks, entities, relationships, communities) |
| `.cache/traditional_rag/<fingerprint>/` | Dense embeddings + chunk metadata |
| `.cache/llm/` | Raw chat/embedding responses (content-hash keyed) |
| `data/` | The cached benchmark and your own documents |
| `results/` | JSON evaluation reports |

---

## Neo4j schema

| Node | Key properties | Relationships |
|------|----------------|---------------|
| `Document` | `doc_id`, `title` | |
| `Chunk` | `chunk_id`, `title`, `text`, `embedding` (vector) | `PART_OF`→Document, `MENTIONS`→Entity |
| `Entity` | `name` (normalised), `display_name`, `type`, `description`, `embedding` (vector) | `RELATES_TO`→Entity, `IN_COMMUNITY`→Community |
| `Community` | `community_id`, `title`, `summary`, `embedding` (vector) | |
| `Corpus` | `id`, `fingerprint`, `corpus_key` | marker node used to detect corpus switches |

Vector indexes: `chunk_embeddings`, `entity_embeddings`, `community_embeddings`.

---

## Prerequisites

- [`uv`](https://docs.astral.sh/uv/) (dependency manager)
- A **Neo4j** instance (local via Homebrew, or Docker, or Neo4j Aura)
- An **OpenAI-compatible LLM/embedding API**. This repo defaults to
  **DeepInfra** (`meta-llama/Llama-3.3-70B-Instruct-Turbo` for answers,
  `deepseek-ai/DeepSeek-V4-Flash` for extraction/summaries, `BAAI/bge-m3` for
  embeddings). Any OpenAI-compatible endpoint works.

---

## Setup

### 1. Neo4j

**Option A — Homebrew (macOS):**

```bash
brew install neo4j
neo4j-admin dbms set-initial-password "<your-password>"
brew services start neo4j
```

Or use the helper script (pass your password as the argument, or export
`NEO4J_PASSWORD` first):

```bash
bash scripts/setup_neo4j.sh "<your-password>"
```

**Option B — Docker:**

```bash
# Set NEO4J_PASSWORD in .env first (docker compose reads it from there).
docker compose up -d
```

Verify: open **http://localhost:7474** (user `neo4j`, password: the one you set
as `NEO4J_PASSWORD` in `.env`).

### 2. Environment

```bash
cp .env.example .env
# edit .env: DEEPINFRA_API_KEY, NEO4J_PASSWORD, model names, and optionally
# LANGFUSE_* / DEEPEVAL_* (see Integrations).
# If you use `make up` / `make langfuse` (Docker), also fill the self-hosted
# stack vars listed at the bottom of .env.example.
```

### 3. Install

```bash
make install            # == uv sync --no-editable
```

> **Why `--no-editable`?** On macOS, `uv`'s editable install writes
> `rag_compare.pth` with the `hidden` flag, which CPython's `site` module skips,
> causing `ModuleNotFoundError: No module named 'rag_compare'`. A non-editable
> install avoids this entirely. See [Troubleshooting](#troubleshooting).

After changing source code, refresh the installed copy:

```bash
make reinstall          # == uv sync --no-editable --reinstall-package rag-compare
```

### 4. Start everything with one command

```bash
make up                 # Neo4j + Langfuse (Docker) + Prefect, with health checks
```

It starts each service, waits for it to answer, and prints the URLs:

```
===================== Services =====================
  Neo4j      http://localhost:7474   (credentials in .env)
  Langfuse   http://localhost:3000   (credentials in .env)
  Prefect    http://127.0.0.1:4200
====================================================
```

It's idempotent (already-running services are detected) and graceful about what
it can't start — e.g. if Docker is missing it skips Langfuse and tells you how to
enable it. Stop the whole stack with:

```bash
make down               # stop Prefect + Langfuse + Neo4j
KEEP_NEO4J=1 make down  # leave Neo4j running
```

Individual services: `make neo4j`, `make langfuse`, `make ui`.

---

## Quickstart (HotpotQA benchmark)

```bash
make neo4j            # start Neo4j (if not already running)
make install          # install dependencies
make config           # print resolved settings (sanity check)

make download         # fetch + cache 30 multi-hop questions / 291 passages

make build            # build the traditional index AND the Neo4j graph
                      # (LLM extraction; everything is cached afterwards)

make ask Q="Were Scott Derrickson and Ed Wood of the same nationality?"
make evaluate         # head-to-head benchmark -> results/comparison.json
make graph            # inspect entities / relationships / communities
```

Equivalent without `make` (note `--no-sync` to avoid re-syncing):

```bash
uv run --no-sync rag-compare download
uv run --no-sync rag-compare build
uv run --no-sync rag-compare ask "..."
uv run --no-sync rag-compare evaluate
```

You can also call the installed script directly: `.venv/bin/rag-compare ...`.

---

## Add your own documents (markdown / text)

**This is the workflow for turning your own files into both a GraphRAG and a
traditional-RAG system.**

### 1. Drop files into `data/`

Any of `.md`, `.markdown`, `.mdx`, `.txt`, `.rst` are picked up **recursively**
from `DOCS_DIR` (defaults to `data/`). Each file becomes one `Document`; the
title is the first markdown heading, or the file name if there is none.

```bash
cp my_notes.md data/
# or a whole folder:
cp -r ~/notes/topic data/topic/
```

> The HotpotQA cache (`data/hotpotqa_*.json`) is ignored by the file scanner, so
> both can live side by side.

### 2. Build both pipelines from the files

```bash
make build SOURCE=files
# == uv run --no-sync rag-compare build --source files
```

What happens:

1. Files are read and chunked (`CHUNK_SIZE` / `CHUNK_OVERLAP`).
2. **Traditional RAG**: every chunk is embedded and stored in a cosine index
   keyed by a fingerprint of the corpus (`​.cache/traditional_rag/<fingerprint>/`).
   The BM25 index is rebuilt in memory from that same stored chunk metadata, so
   no extra artifacts are written.
3. **GraphRAG**: entities/relationships are extracted, written to Neo4j, then
   communities are detected and summarised. Because the Neo4j graph is for one
   corpus at a time, switching corpora **automatically clears and rebuilds** it
   (detected via the `Corpus` marker node).

### 3. Ask questions

```bash
make ask Q="Which rocket did the Apollo program use?" SOURCE=files
uv run --no-sync rag-compare ask "Which rocket did the Apollo program use?" --source files
```

### 4. (Optional) Evaluate your own corpus

For `evaluate` / `deepeval` you need gold answers. Add a `corpus_qa.json` next
to your documents (e.g. `data/corpus_qa.json`):

```json
[
  {
    "question": "Which rocket did the Apollo program use, and who led its design?",
    "answer": "Saturn V, led by Wernher von Braun",
    "supporting_titles": ["Apollo Program", "Saturn V"]
  }
]
```

`supporting_titles` are the document titles (or file names) that contain the
evidence — they power the *supporting-fact recall* metric.

```bash
make evaluate SOURCE=files
make deepeval SOURCE=files
uv run --no-sync rag-compare evaluate --source files
```

A ready-made example lives in [`examples/sample_corpus/`](examples/sample_corpus/)
(three markdown docs + `corpus_qa.json`). Copy it to `data/` to try it.

> **Tip:** set `DATASET_NAME=files` in `.env` to make `files` the default so you
> can omit `--source files` everywhere.

---

## CLI reference

All commands: `uv run --no-sync rag-compare <command>` (or `make <target>`).

| Command | Purpose | Key options |
|---------|---------|-------------|
| `config` | Show resolved settings and integration status | |
| `download` | Load/cache the corpus (download benchmark or scan files) | `--source` |
| `build` | Build traditional index + Neo4j graph | `--source`, `--force` |
| `ask "Q"` | Ask one question, compare both answers | `--source`, `--mode`, `--retrieval`, `-k`, `--context` |
| `evaluate` | Custom metrics (EM/F1/support recall) + report | `--source`, `--limit`, `--mode`, `--retrieval`, `-k`, `--out` |
| `deepeval` | LLM-as-judge metrics | `--source`, `--limit`, `--metrics`, `--retrieval`, `--both/--graph-only`, `--out` |
| `flow` | Run through Prefect | `build\|evaluate\|deepeval\|all`, `--source`, `--limit`, `--mode`, `--retrieval`, `--force` |
| `graph` | Inspect Neo4j (top relations, communities) | `--limit` |

`--source` values: `hotpotqa` (default) or `files`.
`--retrieval` values: `dense` (default), `bm25`, or `hybrid` (traditional RAG only).

**Examples**

```bash
# GraphRAG global (community) search vs plain retrieval
uv run --no-sync rag-compare ask "What are the main themes?" --mode global

# Evaluate only 10 questions, smaller context
uv run --no-sync rag-compare evaluate --limit 10 -k 5

# Compare traditional retrieval variants (dense vs BM25 vs hybrid)
uv run --no-sync rag-compare evaluate --retrieval bm25
uv run --no-sync rag-compare evaluate --retrieval hybrid

# DeepEval with selected metrics
uv run --no-sync rag-compare deepeval --limit 5 --metrics faithfulness,answer_relevancy

# Full pipeline through Prefect
uv run --no-sync rag-compare flow all
```

Outputs land in `results/` (`comparison.json`, `deepeval.json`, …).

---

## Integrations

All three are optional and degrade gracefully. API keys live in `.env`.

### Langfuse — tracing

Langfuse needs a **server**; the Python SDK runs in-process and ships traces to
it over HTTPS. Two options:

**Option A — self-hosted locally (Docker).** Requires Docker Desktop (or colima).
The bundled `docker-compose.langfuse.yml` runs the full Langfuse v4 stack
(Postgres + ClickHouse + Redis + MinIO) and auto-creates a project, so there is
no signup. All credentials come from the gitignored `.env` (no secrets are stored
in the compose file), so set them before starting:

```bash
cp .env.example .env
# fill in LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY (any local values) and the
# stack vars: POSTGRES_PASSWORD, NEXTAUTH_SECRET, SALT, ENCRYPTION_KEY,
# CLICKHOUSE_PASSWORD, REDIS_AUTH, MINIO_ROOT_PASSWORD,
# LANGFUSE_INIT_USER_EMAIL, LANGFUSE_INIT_USER_PASSWORD.

make langfuse            # docker compose -f docker-compose.langfuse.yml up -d
# UI: http://localhost:3000   (login with LANGFUSE_INIT_USER_EMAIL/PASSWORD)

make config              # should show "langfuse tracing: enabled"
make ask Q="..."         # any command now emits traces
```

**Option B — Langfuse Cloud.** Put your cloud keys in `.env` and set
`LANGFUSE_HOST=https://cloud.langfuse.com`.

**What gets traced:** every chat call as a Langfuse **generation** (model,
tokens, input/output) and every embedding call as an **embedding**; each
pipeline answer is a **span** (`traditional_rag.answer`, `graph_rag.answer`,
`comparison.run`). Open the Langfuse UI → **Traces** to inspect.

**Graceful degradation:** if the host is unreachable (e.g. Docker not running)
or keys are blank, tracing self-disables with a warning and nothing is sent — the
app keeps working normally. Confirm with `make config`.

Implementation: [`src/rag_compare/tracing.py`](src/rag_compare/tracing.py) +
hooks in [`llm.py`](src/rag_compare/llm.py).

> In this environment Docker is **not installed**, so the Langfuse stack could
> not be started here — the SDK side is wired and tested, and will emit traces
> as soon as the compose stack is up.

### DeepEval — LLM-as-judge metrics

Runs local LLM-as-judge metrics using your DeepInfra model as the judge (no
OpenAI key needed, no data leaves your provider). Configure which metrics run:

```env
DEEPEVAL_METRICS=contextual_recall,contextual_precision,faithfulness,answer_relevancy
DEEPEVAL_THRESHOLD=0.5
```

Supported: `contextual_recall`, `contextual_precision`, `contextual_relevancy`,
`faithfulness`, `answer_relevancy`, `hallucination`.

```bash
make deepeval
uv run --no-sync rag-compare deepeval --limit 5 --metrics contextual_recall,faithfulness
```

Writes `results/deepeval.json` and prints a per-pipeline score table. Because
each metric makes several judge calls, start with a small `--limit`.

Implementation: [`src/rag_compare/deepeval_eval.py`](src/rag_compare/deepeval_eval.py).

### Prefect — orchestration

Flows live in [`src/rag_compare/flows.py`](src/rag_compare/flows.py):
`rag-compare-build`, `rag-compare-evaluate`, `rag-compare-deepeval`,
`rag-compare-pipeline` (build → evaluate).

```bash
# 1. Start the Prefect API + UI (leave running) -> http://127.0.0.1:4200
make ui                      # == uv run --no-sync prefect server start

# 2. Point the CLI at that server so runs show up in the UI (one-time)
uv run --no-sync prefect config set PREFECT_API_URL="http://127.0.0.1:4200/api"

# 3. Run flows (in another terminal)
uv run --no-sync rag-compare flow all
uv run --no-sync rag-compare flow evaluate --limit 10
uv run --no-sync prefect flow-run ls      # list recorded runs
```

You can also deploy/schedule flows with standard Prefect tooling
(`prefect deploy`). The heavy lifting is in
[`runner.py`](src/rag_compare/runner.py), which the CLI and the flows share, so
behaviour is identical either way.

> **Where does it run?** Prefect flows execute in-process; results are recorded
> to the Prefect API server. If you don't start/point to one, Prefect 3 uses an
> ephemeral local server (runs still work, but there's no persistent UI history).

---

## File-by-file guide

### Top level

| Path | What it is |
|------|------------|
| `pyproject.toml` | Project metadata, dependencies (added via `uv add`), console script `rag-compare` |
| `.env` / `.env.example` | All configuration (API keys, Neo4j, models, integrations) |
| `Makefile` | Convenience targets (`install`, `build`, `ask`, `evaluate`, …) |
| `docker-compose.yml` | Neo4j container (alternative to Homebrew) |
| `docker-compose.langfuse.yml` | Self-hosted Langfuse v4 stack (Postgres/ClickHouse/Redis/MinIO) |
| `scripts/setup_neo4j.sh` | Install + start Neo4j via Homebrew |
| `scripts/start_all.sh` | Start Neo4j + Langfuse + Prefect together (`make up`) |
| `scripts/stop_all.sh` | Stop the stack (`make down`) |
| `scripts/fix_editable_pth.sh` | macOS fix for the editable `.pth` hidden-flag issue |
| `cypher/examples.cypher` | Ready-made Cypher queries to explore the graph |
| `examples/sample_corpus/` | Example markdown docs + `corpus_qa.json` |
| `data/` | Your documents + cached benchmark data (gitignored) |
| `.cache/` | LLM response cache + traditional-index cache (gitignored) |
| `results/` | Evaluation reports (gitignored) |

### Python package (`src/rag_compare/`)

| File | Responsibility |
|------|----------------|
| `__init__.py` | Package marker |
| `__main__.py` | Enables `python -m rag_compare ...` |
| `config.py` | `Settings` (pydantic-settings) loaded from `.env`; project-root detection; paths |
| `tracing.py` | Langfuse wrapper — `span`, `generation`, `embedding`; no-op when unconfigured |
| `llm.py` | `LLMClient`: DeepInfra chat + embeddings, retries, **disk cache**, Langfuse hooks |
| `data.py` | `Document`/`QAExample`; HotpotQA downloader; **local markdown/text loader** (`load_corpus`) + `corpus_qa.json` |
| `text.py` | Sentence-aware `chunk_text`, `build_chunks`, `corpus_fingerprint` |
| `vectorstore.py` | Tiny numpy cosine index with save/load (dense retrieval) |
| `bm25_index.py` | `BM25Index` + tokenizer (lexical retrieval via `rank-bm25`) |
| `traditional_rag.py` | `TraditionalRAG`: dense / BM25 / hybrid (RRF) retrieval → answer |
| `extraction.py` | LLM prompts + parsing for entities/relationships |
| `communities.py` | Community detection (networkx) + LLM community summaries |
| `graph_rag.py` | `GraphRAG`: Neo4j schema, indexes, extraction writes, community build, local/global retrieval, corpus-switch detection |
| `evaluation.py` | Exact match, F1, supporting-fact recall, latency/token aggregation |
| `deepeval_eval.py` | `DeepInfraJudge` + DeepEval metrics runner |
| `runner.py` | Shared orchestration: `build_all`, `run_comparison`, `run_deepeval` (used by CLI **and** Prefect) |
| `flows.py` | Prefect `@flow`/`@task` definitions |
| `cli.py` | Typer CLI (`config`, `download`, `build`, `ask`, `evaluate`, `deepeval`, `flow`, `graph`) |

**How a request flows through the code**

```
cli.py / flows.py
   └─> runner.py            (orchestration)
         ├─> data.py        (load + chunk documents)
         ├─> traditional_rag.py ─> vectorstore.py + bm25_index.py ─> llm.py
         ├─> graph_rag.py  ─> extraction.py, communities.py ─> llm.py
         └─> evaluation.py / deepeval_eval.py
                    └─> llm.py  (all model calls go through here → cached + traced)
```

---

## Metrics explained

| Metric | Meaning |
|--------|---------|
| Exact match | SQuAD-style normalised string equality |
| F1 | token-overlap F1 between prediction and gold |
| **Supporting-fact recall** | fraction of the question's gold passages present in the retrieved context (the key multi-hop signal) |
| All supporting facts found | share of questions where *every* gold passage was retrieved |
| Latency / tokens | efficiency |
| DeepEval metrics | LLM-judge scores for context recall/precision, faithfulness, answer relevancy |

---

## Results

Benchmark: **HotpotQA validation, 30 hard multi-hop questions**, 291 passages,
310 chunks. Answer model `meta-llama/Llama-3.3-70B-Instruct-Turbo`, embeddings
`BAAI/bge-m3`. Traditional RAG `--retrieval dense` (the default), GraphRAG
`hybrid`, 2 hops, 5 entity seeds.

| Metric | Traditional RAG | GraphRAG | Δ |
|--------|----------------:|---------:|---:|
| Exact match | 0.6667 | 0.6333 | -0.0334 |
| **F1** | 0.7627 | **0.8092** | **+0.0465** |
| **Supporting-fact recall** | 0.8111 | **0.9778** | **+0.1667** |
| **All supporting facts found** | 0.7000 | **0.9667** | **+0.2667** |
| Latency (s) | ~2.0 | ~2.3 | +0.3 |
| Tokens / question | 1,136 | 2,555 | +1,419 |
| Context items | 8 | 17 | +9 |

**Reading the numbers**

- GraphRAG's graph traversal surfaces the multi-hop evidence far more reliably:
  supporting-fact recall **0.81 → 0.98**, all-gold-passages **0.70 → 0.97**.
- Answer **F1 improves** (+0.05); strict **exact match dips slightly** (−0.03)
  because the graph adds relevant-but-verbose context.
- Cost: ~2.3× more context tokens, and a one-time LLM-heavy indexing pass.
- A first, unpruned GraphRAG run (every expanded entity description) had the same
  recall gains but a larger EM drop — it packed **85 items / 6,164 tokens** into
  the prompt. Restricting context to seed entities + capped facts + chunks kept
  recall and recovered most of the F1 gap. **Graph context design matters.**

Reports: `results/comparison.json` (pruned run) and
`results/comparison_files.json` (custom corpus demo).

### DeepEval (example output)

On the sample corpus, both pipelines scored `Contextual Recall = 1.0` and
`Faithfulness = 1.0` for the tested question — see `results/deepeval_files.json`.

---

## Cost

DeepInfra standard tier. Indexing is a one-time cost and fully cached.

| Stage | Model | Price in/out /1M | Usage (in/out tokens) | Cost |
|-------|-------|------------------|-----------------------|------|
| Entity extraction (315 calls) | DeepSeek-V4-Flash | $0.09 / $0.18 | 127,635 / 133,934 | $0.036 |
| Community summaries (167 calls) | DeepSeek-V4-Flash | $0.09 / $0.18 | 85,089 / 18,224 | $0.011 |
| Answer generation (90 calls) | Llama-3.3-70B-Turbo | $0.10 / $0.32 | 295,260 / 381 | $0.030 |

Embeddings (`BAAI/bge-m3`) are negligible. Re-running `evaluate` only pays for
answer generation (or nothing, thanks to the cache).

---

## Troubleshooting

- **`ModuleNotFoundError: No module named 'rag_compare'`** (macOS) — uv's
  editable `.pth` got the `hidden` flag and CPython skipped it. Fixes:
  ```bash
  make install        # non-editable install (recommended)
  # or, if you prefer editable:
  bash scripts/fix_editable_pth.sh
  ```
  After editing source: `make reinstall`.
- **`uv run` reverts to a broken install** — use `uv run --no-sync ...` (what the
  Makefile does) or call `.venv/bin/rag-compare ...` directly.
- **`Graph not built` / `different corpus`** — run
  `uv run --no-sync rag-compare build --source <hotpotqa|files>`.
- **Neo4j connection refused** — `brew services start neo4j`, or
  `docker compose up -d`, or update `NEO4J_URI`/password in `.env`.
- **Langfuse traces not appearing** — check `make config` shows
  `enabled -> http://localhost:3000`. Open **http://localhost:3000** → Traces.
  Note: Langfuse v4 disables the legacy `/api/public/traces` endpoint; use the UI
  or `/api/public/v2/observations`. Cached-answer runs emit only **spans**;
  **generations**/embeddings appear only when a real LLM call happens.
- **Commands hang for minutes on the first run** — this is machine I/O pressure,
  not the app. Python imports stall when the project lives on an **iCloud-synced
  `~/Desktop`**, when **Spotlight is indexing** a big `.venv`, or when the **disk
  is nearly full**. Fixes: keep the project on a local (non-iCloud) path, free
  disk space, and/or add a no-index marker (`touch .venv/.metadata_never_index`,
  already done here). After the first warm import, subsequent runs are fast.
- **DeepEval is slow/expensive** — lower `--limit`; each metric makes several
  judge calls.
- **Want to explore the graph?** Open **http://localhost:7474** and paste
  queries from [`cypher/examples.cypher`](cypher/examples.cypher).

---

## Reset

```bash
# Clear just the Neo4j graph
uv run --no-sync rag-compare build --force        # rebuild (clears first)

# Clear caches / data / results
rm -rf .cache data/hotpotqa_*.json results
```

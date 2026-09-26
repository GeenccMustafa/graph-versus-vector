"""Dataset loading: a multi-hop QA benchmark (HotpotQA).

We deliberately use a *multi-hop* benchmark. Multi-hop questions require
combining facts found in two or more separate passages, which is exactly the
regime where a relationship-aware GraphRAG system is expected to beat vanilla
top-k vector RAG.

Source: HotpotQA ``distractor`` split, pulled from the HuggingFace
datasets-server HTTP API (no heavyweight parquet dependency).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from .config import Settings, get_settings

logger = logging.getLogger(__name__)

ROWS_API = "https://datasets-server.huggingface.co/rows"
PAGE_SIZE = 100


@dataclass
class Document:
    """A single source passage (a Wikipedia paragraph)."""

    doc_id: str
    title: str
    text: str

    def render(self) -> str:
        """Return the passage as title-prefixed text for embedding."""
        return f"{self.title}\n{self.text}"


@dataclass
class QAExample:
    """One benchmark question with its gold answer and supporting facts."""

    question: str
    answer: str
    qid: str = ""
    level: str = ""
    qtype: str = ""
    # Supporting facts: list of (title, sentence_index) gold evidence.
    supporting: list[tuple[str, int]] = field(default_factory=list)

    @property
    def supporting_titles(self) -> set[str]:
        """Return the set of distinct gold supporting passage titles."""
        return {t for t, _ in self.supporting}


def _fetch_rows(
    dataset: str, config: str, split: str, offset: int, length: int
) -> list[dict]:
    """Fetch one page of rows from the HuggingFace datasets-server API."""
    params = {
        "dataset": dataset,
        "config": config,
        "split": split,
        "offset": offset,
        "length": length,
    }
    resp = httpx.get(ROWS_API, params=params, timeout=60.0)
    resp.raise_for_status()
    payload = resp.json()
    if "error" in payload:
        raise RuntimeError(f"datasets-server error: {payload['error']}")
    return [r["row"] for r in payload["rows"]]


def load_hotpotqa(
    settings: Settings | None = None,
    *,
    num_examples: int | None = None,
) -> tuple[list[Document], list[QAExample]]:
    """Return ``(documents, examples)`` for the requested slice of HotpotQA.

    Results are cached to ``data/`` so repeated runs are instant and offline.
    """
    settings = settings or get_settings()
    num_examples = num_examples or settings.num_examples
    cache_path = settings.data_dir / f"hotpotqa_{settings.dataset_split}_{num_examples}.json"

    if cache_path.exists():
        logger.info("Loading cached dataset from %s", cache_path)
        raw = json.loads(cache_path.read_text())
    else:
        logger.info("Downloading %d HotpotQA examples ...", num_examples)
        raw = []
        offset = 0
        while len(raw) < num_examples:
            want = min(PAGE_SIZE, num_examples - len(raw))
            raw.extend(_fetch_rows("hotpotqa/hotpot_qa", "distractor", settings.dataset_split, offset, want))
            offset += want
        raw = raw[:num_examples]
        cache_path.write_text(json.dumps(raw))

    documents: dict[str, Document] = {}
    examples: list[QAExample] = []

    for row in raw:
        context = row.get("context", {})
        titles = context.get("title", [])
        sentences = context.get("sentences", [])
        for title, sents in zip(titles, sentences):
            text = " ".join(s.strip() for s in sents).strip()
            if not text:
                continue
            doc_id = title
            documents.setdefault(doc_id, Document(doc_id=doc_id, title=title, text=text))

        sf = row.get("supporting_facts", {})
        supporting = list(zip(sf.get("title", []), sf.get("sent_id", [])))
        examples.append(
            QAExample(
                question=row["question"],
                answer=row["answer"],
                qid=row.get("id", ""),
                level=row.get("level", ""),
                qtype=row.get("type", ""),
                supporting=[(t, int(i)) for t, i in supporting],
            )
        )

    docs = sorted(documents.values(), key=lambda d: d.doc_id)
    logger.info(
        "Dataset ready: %d unique documents, %d questions", len(docs), len(examples)
    )
    return docs, examples


# --------------------------------------------------------------------------- #
# Your own documents (markdown / text)                                        #
# --------------------------------------------------------------------------- #

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".mdx", ".rst"}
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.*)$")


def _read_text(path: Path) -> str:
    """Read ``path`` as UTF-8, falling back to latin-1 on decode errors."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def _title_for(path: Path, text: str) -> str:
    """Return the first markdown heading in ``text``, else the file stem."""
    for line in text.splitlines():
        match = _HEADING_RE.match(line)
        if match:
            return match.group(1).strip()
        if line.strip():
            break
    return path.stem


def load_corpus(
    settings: Settings | None = None,
) -> tuple[list[Document], list[QAExample]]:
    """Load ``.md``/``.txt`` documents from ``docs_dir`` (recursively).

    Every file becomes one :class:`Document`. Drop new markdown files in and
    re-run ``build`` to ingest them. Optionally add a ``corpus_qa.json`` in the
    same directory to enable ``evaluate`` on your own corpus::

        [
          {"question": "...", "answer": "...",
           "supporting_titles": ["My Doc", "Another Doc"]}
        ]
    """
    settings = settings or get_settings()
    root = settings.docs_dir
    root.mkdir(parents=True, exist_ok=True)

    files = sorted(
        p
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_SUFFIXES
        and not any(part.startswith(".") for part in p.relative_to(root).parts)
    )

    documents: list[Document] = []
    for path in files:
        text = _read_text(path).strip()
        if not text:
            continue
        rel = path.relative_to(root)
        documents.append(
            Document(
                doc_id=str(rel),
                title=_title_for(path, text),
                text=text,
            )
        )

    # Optional gold QA pairs for evaluation on the custom corpus.
    qa_path = root / "corpus_qa.json"
    examples: list[QAExample] = []
    if qa_path.exists():
        raw = json.loads(qa_path.read_text())
        for i, row in enumerate(raw):
            supporting = row.get("supporting_titles") or row.get("supporting") or []
            examples.append(
                QAExample(
                    question=row["question"],
                    answer=row["answer"],
                    qid=row.get("id", f"custom-{i}"),
                    level=row.get("level", "custom"),
                    qtype=row.get("type", "custom"),
                    supporting=[(t, 0) for t in supporting],
                )
            )

    logger.info(
        "Loaded %d documents and %d QA examples from %s",
        len(documents),
        len(examples),
        root,
    )
    return documents, examples


def load_dataset(
    settings: Settings | None = None,
    *,
    num_examples: int | None = None,
    source: str | None = None,
) -> tuple[list[Document], list[QAExample]]:
    """Dispatch to the benchmark loader or the local-file loader.

    ``source`` overrides ``settings.dataset_name`` (values: ``hotpotqa`` or
    ``files``).
    """
    settings = settings or get_settings()
    name = (source or settings.dataset_name).lower()
    if name in {"hotpotqa", "hotpot_qa", "benchmark"}:
        return load_hotpotqa(settings, num_examples=num_examples)
    if name in {"files", "corpus", "documents", "docs", "local"}:
        return load_corpus(settings)
    raise ValueError(
        f"Unsupported source: {name!r} (use 'hotpotqa' or 'files')"
    )

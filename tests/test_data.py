"""Unit tests for the local markdown/text corpus loader."""

from __future__ import annotations

import json

from rag_compare.data import Document, load_corpus


def test_document_render() -> None:
    assert Document(doc_id="d", title="Title", text="Body").render() == "Title\nBody"


def test_load_corpus_reads_files_and_qa(settings) -> None:
    docs_dir = settings.docs_dir
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "apollo.md").write_text("# Apollo Program\n\nApollo landed on the Moon.")
    (docs_dir / "saturn_v.md").write_text("# Saturn V\n\nThe Saturn V was a rocket.")
    (docs_dir / "corpus_qa.json").write_text(
        json.dumps(
            [
                {
                    "question": "Which rocket?",
                    "answer": "Saturn V",
                    "supporting_titles": ["Saturn V"],
                }
            ]
        )
    )

    documents, examples = load_corpus(settings)
    titles = {d.title for d in documents}
    assert {"Apollo Program", "Saturn V"} <= titles
    assert len(examples) == 1
    assert examples[0].supporting_titles == {"Saturn V"}


def test_load_corpus_ignores_hidden_and_unsupported(settings) -> None:
    docs_dir = settings.docs_dir
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "keep.md").write_text("# Keep\n\ntext")
    (docs_dir / "image.png").write_bytes(b"\x89PNG")
    hidden = docs_dir / ".hidden"
    hidden.mkdir()
    (hidden / "secret.md").write_text("# Secret\n\nx")

    documents, _ = load_corpus(settings)
    assert {d.title for d in documents} == {"Keep"}

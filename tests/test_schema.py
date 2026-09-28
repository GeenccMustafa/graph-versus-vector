"""Unit tests for the shared result types and answer prompt."""

from __future__ import annotations

from rag_compare.schema import RAGResult, RetrievalItem, build_answer_messages


def test_build_answer_messages_shape() -> None:
    messages = build_answer_messages("Who?", "some context")
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "Who?" in messages[1]["content"]
    assert "some context" in messages[1]["content"]


def test_rag_result_token_and_context_helpers() -> None:
    result = RAGResult(
        question="q",
        answer="a",
        prompt_tokens=10,
        completion_tokens=5,
        retrieval=[
            RetrievalItem(id="1", text="first", score=1.0),
            RetrievalItem(id="2", text="second", score=0.5),
        ],
    )
    assert result.total_tokens == 15
    assert result.context_text == "first\n\nsecond"

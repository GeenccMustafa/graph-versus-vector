"""Shared result types and the answer-generation prompt."""

from __future__ import annotations

from dataclasses import dataclass, field

ANSWER_SYSTEM = (
    "You are a careful question-answering assistant. Answer the user's question "
    "using ONLY the provided context. Multi-hop questions require combining "
    "facts across several items. If the context contains the answer, give the "
    "shortest correct answer (a few words at most). If it does not, answer "
    "\"unknown\". Do not explain your reasoning in the final line."
)


@dataclass
class RetrievalItem:
    """One retrieved piece of evidence, regardless of pipeline.

    Attributes:
        id: Stable identifier of the item (chunk id, entity name, ...).
        text: The text shown to the answering model.
        score: Retrieval score (larger is more relevant).
        source: Human-readable origin, usually a document title.
        kind: Provenance tag, e.g. ``chunk``, ``relation`` or ``community``.
    """

    id: str
    text: str
    score: float
    source: str = ""
    kind: str = "chunk"


@dataclass
class RAGResult:
    """The answer, retrieved evidence and usage stats for one question.

    Attributes:
        question: The question that was answered.
        answer: The model's answer string.
        retrieval: The evidence items passed to the model, in prompt order.
        latency_s: End-to-end answer latency in seconds.
        prompt_tokens: Prompt tokens reported by the provider.
        completion_tokens: Completion tokens reported by the provider.
        graph_facts: Human-readable relationship facts (GraphRAG only).
    """

    question: str
    answer: str
    retrieval: list[RetrievalItem] = field(default_factory=list)
    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Human-readable graph facts, only for GraphRAG (for inspection).
    graph_facts: list[str] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        """Return the combined prompt and completion token count."""
        return self.prompt_tokens + self.completion_tokens

    @property
    def context_text(self) -> str:
        """Return the retrieved item texts joined into one context string."""
        return "\n\n".join(item.text for item in self.retrieval)


def build_answer_messages(question: str, context: str) -> list[dict[str, str]]:
    """Build the chat messages that answer ``question`` from ``context``.

    Args:
        question: The user question to answer.
        context: Concatenated retrieved evidence to ground the answer.

    Returns:
        A two-message list (``system`` + ``user``) for the chat completion API.
    """
    user = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer with the shortest correct string only."
    )
    return [
        {"role": "system", "content": ANSWER_SYSTEM},
        {"role": "user", "content": user},
    ]

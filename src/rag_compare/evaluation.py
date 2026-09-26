"""Evaluation: answer quality + retrieval coverage for multi-hop QA.

Metrics
-------
* exact_match / F1 : standard SQuAD-style scoring of the predicted answer.
* support_recall   : fraction of the question's gold supporting passages that
                     actually appear in the retrieved context. This is the key
                     signal for multi-hop questions -- a pipeline that only
                     fetches one of the two required facts cannot answer.
* support_all      : share of questions where *every* gold passage was found.
* latency / tokens : efficiency.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass, field

from .data import QAExample
from .schema import RAGResult


def normalize_answer(text: str) -> str:
    """Lowercase, strip punctuation and articles, and collapse whitespace."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(pred: str, gold: str) -> float:
    """Return 1.0 if the normalised prediction equals the gold answer."""
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred: str, gold: str) -> float:
    """Return the token-level F1 score between the prediction and gold answer."""
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision = same / len(pred_tokens)
    recall = same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def support_recall(result: RAGResult, example: QAExample) -> tuple[float, bool]:
    """Return the gold-passage recall and whether every gold passage was found."""
    context = result.context_text.lower()
    gold_titles = [t for t, _ in example.supporting]
    if not gold_titles:
        return 1.0, True
    found = [t for t in gold_titles if t.lower() in context]
    recall = len(found) / len(gold_titles)
    return recall, len(found) == len(gold_titles)


@dataclass
class EvalRecord:
    """Per-question scores for one pipeline."""

    question: str
    gold: str
    predicted: str
    em: float
    f1: float
    support_recall: float
    support_all: bool
    latency_s: float
    total_tokens: int
    num_context_items: int


@dataclass
class EvalSummary:
    """Aggregated metric means for one pipeline."""

    name: str
    records: list[EvalRecord] = field(default_factory=list)

    def add(self, record: EvalRecord) -> None:
        """Append a scored question to the summary."""
        self.records.append(record)

    @property
    def n(self) -> int:
        """Return the number of scored questions."""
        return len(self.records)

    def _mean(self, attr: str) -> float:
        """Return the mean of ``attr`` across all records (0.0 if empty)."""
        if not self.records:
            return 0.0
        return sum(getattr(r, attr) for r in self.records) / len(self.records)

    def as_dict(self) -> dict:
        """Return the rounded aggregate metrics as a plain dictionary."""
        return {
            "name": self.name,
            "n": self.n,
            "exact_match": round(self._mean("em"), 4),
            "f1": round(self._mean("f1"), 4),
            "support_recall": round(self._mean("support_recall"), 4),
            "support_all": round(
                sum(r.support_all for r in self.records) / self.n if self.n else 0.0, 4
            ),
            "avg_latency_s": round(self._mean("latency_s"), 3),
            "avg_tokens": round(self._mean("total_tokens"), 1),
            "avg_context_items": round(self._mean("num_context_items"), 1),
        }


def score_result(result: RAGResult, example: QAExample) -> EvalRecord:
    """Score one pipeline result against its gold example."""
    recall, all_found = support_recall(result, example)
    return EvalRecord(
        question=example.question,
        gold=example.answer,
        predicted=result.answer,
        em=exact_match(result.answer, example.answer),
        f1=f1_score(result.answer, example.answer),
        support_recall=recall,
        support_all=all_found,
        latency_s=result.latency_s,
        total_tokens=result.total_tokens,
        num_context_items=len(result.retrieval),
    )

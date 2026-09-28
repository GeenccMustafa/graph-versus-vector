"""DeepEval integration: LLM-as-judge metrics for RAG quality.

We plug DeepInfra in as the judge model (via :class:`DeepEvalBaseLLM`) so no
OpenAI key is required. Metrics are computed manually with ``metric.measure``
so nothing is sent to the Confident AI cloud — everything runs locally.

Supported metric names (see ``DEEPEVAL_METRICS``):
``contextual_recall``, ``contextual_precision``, ``contextual_relevancy``,
``faithfulness``, ``answer_relevancy``, ``hallucination``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass

# Must be set before importing deepeval to avoid telemetry/network calls.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

from deepeval.metrics import (  # noqa: E402
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    HallucinationMetric,
)
from deepeval.models import DeepEvalBaseLLM  # noqa: E402
from deepeval.test_case import LLMTestCase  # noqa: E402

from .config import Settings, get_settings  # noqa: E402
from .llm import LLMClient  # noqa: E402
from .schema import RAGResult  # noqa: E402

logger = logging.getLogger(__name__)


class DeepInfraJudge(DeepEvalBaseLLM):
    """DeepEval judge backed by our DeepInfra (OpenAI-compatible) client.

    Attributes:
        settings: The resolved settings in use.
        llm: The shared chat client used to produce judgements.
    """

    def __init__(self, llm: LLMClient | None = None, settings: Settings | None = None):
        """Bind the settings and the shared LLM client used for judging.

        Args:
            llm: Optional shared :class:`LLMClient`.
            settings: Optional settings override.
        """
        self.settings = settings or get_settings()
        self.llm = llm or LLMClient(self.settings)

    def load_model(self):
        """Return ``self``; DeepEval uses this to lazily initialise the model.

        Returns:
            This judge instance.
        """
        return self

    def generate(self, prompt: str, schema=None, **kwargs):
        """Generate a judge response, validating it against ``schema`` if given.

        Args:
            prompt: The judge prompt.
            schema: Optional pydantic model the response must validate against.
            **kwargs: Ignored extra arguments from the DeepEval interface.

        Returns:
            The validated schema instance when ``schema`` is given, otherwise
            the raw response text.

        Raises:
            ValueError: If a structured response cannot be produced.
        """
        if schema is not None:
            data, _ = self.llm.chat_json(
                [
                    {
                        "role": "system",
                        "content": "You are a strict evaluation assistant. Reply with JSON only.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1024,
            )
            if data is not None:
                try:
                    return schema.model_validate(data)
                except Exception:
                    try:
                        return schema(**data)
                    except Exception:
                        pass
            # Ask again without JSON mode as a fallback.
            text = self.llm.chat(
                [{"role": "user", "content": prompt}], max_tokens=1024
            ).text
            try:
                return schema.model_validate_json(text)
            except Exception as exc:  # pragma: no cover
                raise ValueError(
                    f"Judge could not produce valid schema output: {exc}"
                ) from exc
        return self.llm.chat(
            [{"role": "user", "content": prompt}], max_tokens=1024
        ).text

    async def a_generate(self, prompt: str, schema=None, **kwargs):
        """Asynchronous wrapper around :meth:`generate`.

        Args:
            prompt: The judge prompt.
            schema: Optional pydantic model the response must validate against.
            **kwargs: Ignored extra arguments from the DeepEval interface.

        Returns:
            The same value as :meth:`generate`.
        """
        return self.generate(prompt, schema=schema, **kwargs)

    def get_model_name(self) -> str:
        """Return the underlying model name, for reporting.

        Returns:
            The configured chat model name.
        """
        return self.settings.llm_model


METRIC_REGISTRY = {
    "contextual_recall": ContextualRecallMetric,
    "contextual_precision": ContextualPrecisionMetric,
    "contextual_relevancy": ContextualRelevancyMetric,
    "faithfulness": FaithfulnessMetric,
    "answer_relevancy": AnswerRelevancyMetric,
    "hallucination": HallucinationMetric,
}


@dataclass
class DeepevalRecord:
    """One metric score for one question under one pipeline.

    Attributes:
        pipeline: The pipeline the score belongs to.
        question: The evaluated question.
        metric: The DeepEval metric name.
        score: The metric score, or ``None`` if measurement failed.
        success: Whether the score met the configured threshold.
        reason: The judge's explanation, or an error string.
    """

    pipeline: str
    question: str
    metric: str
    score: float | None
    success: bool
    reason: str


def build_test_case(question: str, result: RAGResult, expected: str) -> LLMTestCase:
    """Convert a RAG result into a DeepEval LLM test case.

    Args:
        question: The question that was asked.
        result: The pipeline result to convert.
        expected: The gold answer.

    Returns:
        The assembled :class:`LLMTestCase`.
    """
    return LLMTestCase(
        input=question,
        actual_output=result.answer,
        expected_output=expected,
        retrieval_context=[item.text for item in result.retrieval],
    )


def evaluate_test_cases(
    cases_by_pipeline: dict[str, list[LLMTestCase]],
    settings: Settings | None = None,
    metric_names: list[str] | None = None,
    judge: DeepInfraJudge | None = None,
) -> dict:
    """Run DeepEval metrics over one or more pipelines and summarise.

    Args:
        cases_by_pipeline: Test cases keyed by pipeline name.
        settings: Optional settings override.
        metric_names: Metric names to run; ``None`` uses the configured list.
        judge: Optional judge override.

    Returns:
        A dict ``{"summary": {pipeline: {metric: {...}}}, "records": [...]}``.
    """
    settings = settings or get_settings()
    metric_names = metric_names or settings.deepeval_metric_list
    judge = judge or DeepInfraJudge(settings=settings)

    metrics = []
    for name in metric_names:
        cls = METRIC_REGISTRY.get(name)
        if cls is None:
            logger.warning("Unknown DeepEval metric %r, skipping", name)
            continue
        metrics.append(
            cls(
                model=judge,
                threshold=settings.deepeval_threshold,
                include_reason=True,
                async_mode=False,
            )
        )

    records: list[DeepevalRecord] = []
    summary: dict[str, dict] = {}

    for pipeline, cases in cases_by_pipeline.items():
        summary[pipeline] = {}
        for metric in metrics:
            scores: list[float] = []
            successes = 0
            for case in cases:
                try:
                    metric.measure(case)
                    score = float(metric.score) if metric.score is not None else None
                    success = bool(metric.success)
                    reason = metric.reason or ""
                except Exception as exc:  # pragma: no cover - judge/parse failures
                    logger.warning(
                        "DeepEval %s failed on %r: %s", metric.__name__, case.input[:40], exc
                    )
                    score, success, reason = None, False, f"error: {exc}"
                if score is not None:
                    scores.append(score)
                    successes += int(success)
                records.append(
                    DeepevalRecord(
                        pipeline=pipeline,
                        question=case.input,
                        metric=metric.__name__,
                        score=score,
                        success=success,
                        reason=reason,
                    )
                )
            mean = sum(scores) / len(scores) if scores else 0.0
            success_rate = successes / len(cases) if cases else 0.0
            summary[pipeline][metric.__name__] = {
                "score": round(mean, 4),
                "success_rate": round(success_rate, 4),
                "n": len(cases),
            }

    return {"summary": summary, "records": [asdict(r) for r in records]}

"""Unit tests for the settings-derived helper properties."""

from __future__ import annotations

from rag_compare.config import Settings


def test_extractor_model_falls_back_to_llm_model() -> None:
    assert Settings(LLM_MODEL="big", EXTRACTION_MODEL="").extractor_model == "big"
    assert Settings(LLM_MODEL="big", EXTRACTION_MODEL="small").extractor_model == "small"


def test_tracing_requires_both_keys() -> None:
    assert Settings(LANGFUSE_PUBLIC_KEY="pk", LANGFUSE_SECRET_KEY="").tracing_enabled is False
    assert Settings(LANGFUSE_PUBLIC_KEY="pk", LANGFUSE_SECRET_KEY="sk").tracing_enabled is True
    assert (
        Settings(LANGFUSE_ENABLED=False, LANGFUSE_PUBLIC_KEY="pk", LANGFUSE_SECRET_KEY="sk")
        .tracing_enabled
        is False
    )


def test_deepeval_metric_list_splits_and_strips() -> None:
    settings = Settings(DEEPEVAL_METRICS=" faithfulness, answer_relevancy ,")
    assert settings.deepeval_metric_list == ["faithfulness", "answer_relevancy"]

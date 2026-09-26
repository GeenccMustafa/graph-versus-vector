"""Central configuration, loaded from environment / .env."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _find_project_root() -> Path:
    """Locate the project root (the dir holding pyproject.toml/.env).

    Works whether the package is installed editable or as a regular wheel,
    and regardless of where the source file lives.
    """
    env_root = os.environ.get("RAG_COMPARE_ROOT")
    if env_root:
        return Path(env_root).resolve()
    cwd = Path.cwd().resolve()
    for candidate in [cwd, *cwd.parents]:
        if (candidate / "pyproject.toml").exists() or (candidate / ".env").exists():
            return candidate
    return cwd


PROJECT_ROOT = _find_project_root()


class Settings(BaseSettings):
    """Runtime settings. Values come from the environment or a local .env file."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM / embedding provider (DeepInfra, OpenAI-compatible) ---
    deepinfra_api_key: str = Field(default="", alias="DEEPINFRA_API_KEY")
    deepinfra_base_url: str = Field(
        default="https://api.deepinfra.com/v1/openai",
        alias="DEEPINFRA_BASE_URL",
    )

    # Chat model used for entity extraction, answering and summarisation.
    llm_model: str = Field(
        default="meta-llama/Llama-3.3-70B-Instruct-Turbo", alias="LLM_MODEL"
    )
    # A separate (usually smaller/cheaper/faster) model for extraction is optional.
    extraction_model: str = Field(default="", alias="EXTRACTION_MODEL")
    extraction_workers: int = Field(default=8, alias="EXTRACTION_WORKERS")

    embedding_model: str = Field(default="BAAI/bge-m3", alias="EMBEDDING_MODEL")
    embedding_dim: int = Field(default=1024, alias="EMBEDDING_DIM")

    # --- Neo4j ---
    neo4j_uri: str = Field(default="bolt://localhost:7687", alias="NEO4J_URI")
    neo4j_user: str = Field(default="neo4j", alias="NEO4J_USER")
    neo4j_password: str = Field(default="password", alias="NEO4J_PASSWORD")
    neo4j_database: str = Field(default="neo4j", alias="NEO4J_DATABASE")

    # --- Retrieval / chunking ---
    chunk_size: int = Field(default=1200, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, alias="CHUNK_OVERLAP")
    top_k: int = Field(default=8, alias="TOP_K")
    graph_hops: int = Field(default=2, alias="GRAPH_HOPS")
    graph_entity_seeds: int = Field(default=5, alias="GRAPH_ENTITY_SEEDS")
    graph_max_facts: int = Field(default=40, alias="GRAPH_MAX_FACTS")

    # --- Data / cache ---
    # dataset_name selects the corpus: "hotpotqa" (download benchmark) or
    # "files" (read .md/.txt files from docs_dir).
    dataset_name: str = Field(default="hotpotqa", alias="DATASET_NAME")
    dataset_split: str = Field(default="validation", alias="DATASET_SPLIT")
    num_examples: int = Field(default=30, alias="NUM_EXAMPLES")
    data_dir: Path = Field(default=PROJECT_ROOT / "data", alias="DATA_DIR")
    cache_dir: Path = Field(default=PROJECT_ROOT / ".cache", alias="CACHE_DIR")
    # Where to look for your own markdown/text documents. Scanned recursively.
    docs_dir: Path = Field(default=PROJECT_ROOT / "data", alias="DOCS_DIR")

    # --- Langfuse (LLM observability). Enabled when both keys are present. ---
    langfuse_enabled: bool = Field(default=True, alias="LANGFUSE_ENABLED")
    langfuse_public_key: str = Field(default="", alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str = Field(default="", alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field(
        default="https://cloud.langfuse.com", alias="LANGFUSE_HOST"
    )

    # --- DeepEval (LLM-as-judge metrics) ---
    deepeval_metrics: str = Field(
        default="contextual_recall,contextual_precision,faithfulness,answer_relevancy",
        alias="DEEPEVAL_METRICS",
    )
    deepeval_threshold: float = Field(default=0.5, alias="DEEPEVAL_THRESHOLD")
    deepeval_telemetry_off: bool = Field(default=True, alias="DEEPEVAL_TELEMETRY_OFF")

    @property
    def cache_enabled(self) -> bool:
        """Return whether on-disk response caching is active (always true)."""
        return True

    @property
    def extractor_model(self) -> str:
        """Return the model used for extraction, falling back to the chat model."""
        return self.extraction_model or self.llm_model

    @property
    def tracing_enabled(self) -> bool:
        """Return whether Langfuse tracing has all required keys."""
        return bool(
            self.langfuse_enabled
            and self.langfuse_public_key
            and self.langfuse_secret_key
        )

    @property
    def deepeval_metric_list(self) -> list[str]:
        """Return the configured DeepEval metric names as a list."""
        return [m.strip() for m in self.deepeval_metrics.split(",") if m.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide cached :class:`Settings` and ensure dirs exist."""
    settings = Settings()  # type: ignore[call-arg]
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    return settings

"""Thin wrapper around an OpenAI-compatible provider (DeepInfra) with disk caching.

Everything the pipelines need goes through :class:`LLMClient`:

* ``chat``   -- text generation (extraction, answering, summarisation)
* ``embed``  -- embeddings, batched and cached on disk

Caching matters a lot here: building a graph over the same corpus repeatedly
(entity extraction is many calls) should not burn tokens twice.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import Settings, get_settings
from .tracing import Tracer

logger = logging.getLogger(__name__)


@dataclass
class ChatResult:
    """The text and token usage returned by a chat completion.

    Attributes:
        text: The generated text.
        prompt_tokens: Prompt tokens reported by the provider.
        completion_tokens: Completion tokens reported by the provider.
        cached: Whether the result was served from the disk cache.
    """

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached: bool = False

    @property
    def total_tokens(self) -> int:
        """Return the combined prompt and completion token count."""
        return self.prompt_tokens + self.completion_tokens


def _hash(payload: Any) -> str:
    """Return a stable SHA-256 hex digest of a JSON-serialisable payload.

    Args:
        payload: Any JSON-serialisable object.

    Returns:
        The hex digest, stable across processes and key orderings.
    """
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


class DiskCache:
    """Tiny JSON/NumPy cache keyed by a content hash.

    Attributes:
        root: Root directory holding one subdirectory per cache kind.
    """

    def __init__(self, root: Path):
        """Create the cache rooted at ``root``, creating the directory if needed.

        Args:
            root: Root directory for cached entries.
        """
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, kind: str, key: str) -> Path:
        """Return the cache file path for a key, creating its directory.

        Args:
            kind: The cache namespace, e.g. ``chat`` or ``embed``.
            key: The content-hash key.

        Returns:
            The path to the JSON file for this entry.
        """
        sub = self.root / kind
        sub.mkdir(parents=True, exist_ok=True)
        return sub / f"{key}.json"

    def get(self, kind: str, key: str):
        """Return the cached JSON value for ``key``, or ``None`` on a miss.

        Args:
            kind: The cache namespace.
            key: The content-hash key.

        Returns:
            The deserialised value, or ``None`` if not cached.
        """
        path = self._path(kind, key)
        if path.exists():
            return json.loads(path.read_text())
        return None

    def put(self, kind: str, key: str, value) -> None:
        """Write ``value`` as JSON under the given cache kind and key.

        Args:
            kind: The cache namespace.
            key: The content-hash key.
            value: A JSON-serialisable value to store.
        """
        self._path(kind, key).write_text(json.dumps(value))


class LLMClient:
    """Chat and embedding client for an OpenAI-compatible provider, with caching.

    Attributes:
        settings: The resolved settings in use.
        client: The underlying OpenAI-compatible client.
        cache: The on-disk response cache.
        tracer: The Langfuse tracer (a no-op when unconfigured).
        call_count: Number of uncached provider calls made.
    """

    def __init__(self, settings: Settings | None = None):
        """Create the OpenAI-compatible client, cache and tracer.

        Args:
            settings: Optional settings override.

        Raises:
            RuntimeError: If ``DEEPINFRA_API_KEY`` is not configured.
        """
        self.settings = settings or get_settings()
        if not self.settings.deepinfra_api_key:
            raise RuntimeError(
                "DEEPINFRA_API_KEY is not set. Copy .env.example to .env and fill it in."
            )
        self.client = OpenAI(
            api_key=self.settings.deepinfra_api_key,
            base_url=self.settings.deepinfra_base_url,
        )
        self.cache = DiskCache(self.settings.cache_dir / "llm")
        self.tracer = Tracer(self.settings)
        self.call_count = 0

    # ------------------------------------------------------------------ chat
    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def _chat_raw(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ):
        """Call the provider's chat endpoint with retries on failure.

        Args:
            model: The model to call.
            messages: The chat messages.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            json_mode: Whether to request a JSON-object response format.

        Returns:
            The raw OpenAI-compatible completion object.
        """
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        return self.client.chat.completions.create(**kwargs)

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        json_mode: bool = False,
        cache: bool = True,
    ) -> ChatResult:
        """Return a chat completion, serving from and writing to the disk cache.

        Args:
            messages: The chat messages.
            model: Model override; defaults to ``settings.llm_model``.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            json_mode: Whether to request a JSON-object response format.
            cache: Whether to read from and write to the disk cache.

        Returns:
            The generated text and token usage.
        """
        model = model or self.settings.llm_model
        key = _hash(
            {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "json_mode": json_mode,
            }
        )
        if cache:
            hit = self.cache.get("chat", key)
            if hit is not None:
                return ChatResult(
                    text=hit["text"],
                    prompt_tokens=hit.get("prompt_tokens", 0),
                    completion_tokens=hit.get("completion_tokens", 0),
                    cached=True,
                )

        t0 = time.perf_counter()
        with self.tracer.generation(
            "chat", model=model, input=messages
        ) as generation:
            resp = self._chat_raw(model, messages, temperature, max_tokens, json_mode)
            self.call_count += 1
            text = resp.choices[0].message.content or ""
            usage = getattr(resp, "usage", None)
            prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(usage, "completion_tokens", 0) or 0
            self.tracer.update(
                generation,
                output=text,
                usage_details={"input": prompt_tokens, "output": completion_tokens},
                metadata={
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "json_mode": json_mode,
                    "cache_hit": False,
                },
            )
        logger.debug(
            "chat model=%s took=%.2fs tokens=%s",
            model,
            time.perf_counter() - t0,
            prompt_tokens + completion_tokens,
        )
        if cache:
            self.cache.put(
                "chat",
                key,
                {
                    "text": text,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                },
            )
        return ChatResult(text, prompt_tokens, completion_tokens, cached=False)

    # ----------------------------------------------------------------- embed
    def embed(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 32,
        cache: bool = True,
    ) -> np.ndarray:
        """Embed texts, serving from and writing to the disk cache.

        Args:
            texts: The texts to embed.
            batch_size: Number of texts per provider request.
            cache: Whether to read from and write to the disk cache.

        Returns:
            An ``(N, D)`` float32 array of L2-normalised embeddings.
        """
        out: list[list[float] | None] = [None] * len(texts)
        pending: list[tuple[int, str]] = []
        keys: dict[int, str] = {}

        if cache:
            for i, text in enumerate(texts):
                key = _hash({"model": self.settings.embedding_model, "text": text})
                keys[i] = key
                hit = self.cache.get("embed", key)
                if hit is not None:
                    out[i] = hit
                else:
                    pending.append((i, text))
        else:
            pending = list(enumerate(texts))

        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            batch_texts = [t for _, t in batch]
            with self.tracer.embedding(
                "embed", model=self.settings.embedding_model, input=batch_texts
            ) as observation:
                resp = self._embed_raw(batch_texts)
                self.tracer.update(
                    observation, metadata={"batch_size": len(batch_texts)}
                )
            for (idx, _text), emb in zip(batch, resp):
                out[idx] = emb
                if cache:
                    self.cache.put("embed", keys[idx], emb)

        arr = np.asarray(out, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def _embed_raw(self, texts: list[str]) -> list[list[float]]:
        """Call the provider's embeddings endpoint with retries on failure.

        Args:
            texts: The batch of texts to embed.

        Returns:
            One embedding vector per input text.
        """
        resp = self.client.embeddings.create(
            model=self.settings.embedding_model, input=texts
        )
        self.call_count += 1
        return [d.embedding for d in resp.data]

    def embed_one(self, text: str) -> np.ndarray:
        """Return the embedding vector for a single string.

        Args:
            text: The text to embed.

        Returns:
            The L2-normalised embedding vector.
        """
        return self.embed([text])[0]

    # ------------------------------------------------------------- json helper
    def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> tuple[Any, ChatResult]:
        """Chat that must return JSON; repairs common wrapping issues.

        Args:
            messages: The chat messages.
            model: Model override; defaults to ``settings.llm_model``.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            A ``(parsed_json, chat_result)`` tuple; ``parsed_json`` is ``None``
            when parsing fails.
        """
        result = self.chat(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,
        )
        return _parse_json(result.text), result


def _parse_json(text: str) -> Any:
    """Parse JSON from model output, tolerating code fences and prose wrappers.

    Args:
        text: The raw model output.

    Returns:
        The parsed JSON value, or ``None`` if it cannot be parsed.
    """
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip().rstrip("`").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
        logger.warning("Could not parse JSON from model output: %.200s", text)
        return None

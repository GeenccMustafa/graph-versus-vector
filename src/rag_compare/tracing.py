"""Optional Langfuse tracing.

If ``LANGFUSE_PUBLIC_KEY`` and ``LANGFUSE_SECRET_KEY`` are set (and
``LANGFUSE_ENABLED`` is not false), every chat/embedding call is recorded as a
Langfuse generation, and high-level pipeline steps are recorded as spans.

If Langfuse is not configured, every method here degrades to a no-op so the
application runs exactly the same without it.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Iterator

from .config import Settings, get_settings

logger = logging.getLogger(__name__)


def langfuse_reachable(host: str, timeout: float = 2.0) -> bool:
    """Return whether a Langfuse server answers its health endpoint at ``host``."""
    if not host:
        return False
    try:
        import httpx

        url = host.rstrip("/") + "/api/public/health"
        return httpx.get(url, timeout=timeout).status_code < 500
    except Exception:
        return False


class Tracer:
    """Langfuse tracing facade that degrades to no-ops when unconfigured."""

    def __init__(self, settings: Settings | None = None):
        """Initialise the client if tracing is enabled and the host is reachable."""
        self.settings = settings or get_settings()
        self.client = None
        if not self.settings.tracing_enabled:
            logger.info("Langfuse tracing disabled (no keys configured)")
            return
        if not self._host_reachable():
            logger.warning(
                "Langfuse host %s is unreachable; tracing disabled. "
                "Start it with `docker compose -f docker-compose.langfuse.yml up -d`.",
                self.settings.langfuse_host,
            )
            return
        try:
            from langfuse import Langfuse

            self.client = Langfuse(
                public_key=self.settings.langfuse_public_key,
                secret_key=self.settings.langfuse_secret_key,
                host=self.settings.langfuse_host,
            )
            logger.info("Langfuse tracing enabled -> %s", self.settings.langfuse_host)
        except Exception as exc:  # pragma: no cover - never break the app
            logger.warning("Langfuse init failed, tracing disabled: %s", exc)
            self.client = None

    def _host_reachable(self) -> bool:
        """Return whether the configured Langfuse host answers its health check."""
        return langfuse_reachable(self.settings.langfuse_host)

    @property
    def enabled(self) -> bool:
        """Return whether a live Langfuse client is attached."""
        return self.client is not None

    @contextmanager
    def span(self, name: str, **kwargs: Any) -> Iterator[Any]:
        """Yield a Langfuse span context, or ``None`` when tracing is off."""
        if self.client is None:
            yield None
            return
        try:
            with self.client.start_as_current_observation(
                as_type="span", name=name, **kwargs
            ) as obs:
                yield obs
        except Exception as exc:  # pragma: no cover
            logger.debug("Langfuse span error: %s", exc)
            yield None

    @contextmanager
    def generation(
        self, name: str, *, model: str, input: Any = None, **kwargs: Any
    ) -> Iterator[Any]:
        """Yield a Langfuse generation context, or ``None`` when tracing is off."""
        if self.client is None:
            yield None
            return
        try:
            with self.client.start_as_current_observation(
                as_type="generation", name=name, model=model, input=input, **kwargs
            ) as obs:
                yield obs
        except Exception as exc:  # pragma: no cover
            logger.debug("Langfuse generation error: %s", exc)
            yield None

    @contextmanager
    def embedding(self, name: str, *, model: str, input: Any = None, **kwargs: Any) -> Iterator[Any]:
        """Yield a Langfuse embedding context, or ``None`` when tracing is off."""
        if self.client is None:
            yield None
            return
        try:
            with self.client.start_as_current_observation(
                as_type="embedding", name=name, model=model, input=input, **kwargs
            ) as obs:
                yield obs
        except Exception as exc:  # pragma: no cover
            logger.debug("Langfuse embedding error: %s", exc)
            yield None

    def update(self, obs: Any, **kwargs: Any) -> None:
        """Update a span/generation observation, ignoring no-op observers."""
        if obs is None:
            return
        try:
            obs.update(**kwargs)
        except Exception as exc:  # pragma: no cover
            logger.debug("Langfuse update error: %s", exc)

    def score_current(self, name: str, value: float, comment: str | None = None) -> None:
        """Attach a numeric score to the current trace."""
        if self.client is None:
            return
        try:
            self.client.score_current_trace(name=name, value=value, comment=comment)
        except Exception as exc:  # pragma: no cover
            logger.debug("Langfuse score error: %s", exc)

    def flush(self) -> None:
        """Flush any buffered events to Langfuse."""
        if self.client is None:
            return
        try:
            self.client.flush()
        except Exception:  # pragma: no cover
            pass


_tracer: Tracer | None = None


def get_tracer(settings: Settings | None = None) -> Tracer:
    """Return the process-wide singleton :class:`Tracer`."""
    global _tracer
    if _tracer is None:
        _tracer = Tracer(settings)
    return _tracer

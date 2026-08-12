"""Retry OpenRouter chat completions across a tier's model chain."""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AsyncOpenAI,
    OpenAI,
    RateLimitError,
)

logger = logging.getLogger(__name__)


def should_failover(exc: BaseException) -> bool:
    """
    Whether to try the next model in the chain.

    Fails fast on auth/billing; retries on rate limits, transient HTTP, transport,
    timeouts, and unknown-model (404) responses.
    """
    if isinstance(exc, (RateLimitError, APIConnectionError, APITimeoutError)):
        return True
    code = getattr(exc, "status_code", None)
    if code in (404, 429, 500, 502, 503, 529):
        return True
    if isinstance(exc, APIError):
        c2 = getattr(exc, "status_code", None)
        if c2 in (404, 429, 500, 502, 503, 529):
            return True
    return False


def chat_completions_with_fallbacks(
    client: OpenAI,
    models: list[str],
    **kwargs: Any,
) -> Tuple[Any, str, bool]:
    """
    ``client.chat.completions.create`` using the first model in ``models``,
    then each next entry on failover.

    Returns ``(response, model_used, used_fallback)`` where ``used_fallback`` is
    True if a model after the first succeeded.
    """
    if not models:
        raise ValueError("models list is empty")
    last: Optional[BaseException] = None
    for i, mid in enumerate(models):
        try:
            resp = client.chat.completions.create(model=mid, **kwargs)
            return resp, mid, i > 0
        except BaseException as exc:
            last = exc
            if not should_failover(exc):
                raise
            logger.warning(
                "OpenRouter model %r failed (%s); trying next in chain",
                mid,
                exc,
                exc_info=True,
            )
    if last is None:  # unreachable: `models` is non-empty and every path sets `last`
        raise RuntimeError("no model in the fallback chain was attempted")
    raise last


async def async_chat_completions_with_fallbacks(
    client: AsyncOpenAI,
    models: list[str],
    **kwargs: Any,
) -> Tuple[Any, str, bool]:
    """Try each model in order until one returns, then report which one did.

    Args:
        client: The async OpenAI-compatible client to call.
        models: Model ids in preference order; the first is the primary.
        **kwargs: Forwarded verbatim to `chat.completions.create`.

    Returns:
        `(response, model_used, used_fallback)` — `used_fallback` is True when
        the primary model failed and a later one answered.

    Raises:
        ValueError: If `models` is empty.
        BaseException: The last provider error, when every model failed.
    """
    if not models:
        raise ValueError("models list is empty")
    last: Optional[BaseException] = None
    for i, mid in enumerate(models):
        try:
            resp = await client.chat.completions.create(model=mid, **kwargs)
            return resp, mid, i > 0
        except BaseException as exc:
            last = exc
            if not should_failover(exc):
                raise
            logger.warning(
                "OpenRouter model %r failed (%s); trying next in chain",
                mid,
                exc,
                exc_info=True,
            )
    if last is None:  # unreachable: `models` is non-empty and every path sets `last`
        raise RuntimeError("no model in the fallback chain was attempted")
    raise last


__all__ = [
    "async_chat_completions_with_fallbacks",
    "chat_completions_with_fallbacks",
    "should_failover",
]

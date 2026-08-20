"""Cached OpenAI-compatible clients pointed at OpenRouter."""

from __future__ import annotations

import asyncio
import os
from functools import lru_cache
from typing import Dict, Optional

from decouple import config
from openai import AsyncOpenAI, OpenAI

from . import settings
from .env_names import LEGACY_OPENAI_API_KEY_ENV, OPENROUTER_API_KEY_ENV
from .invocation_context import get_llm_invocation_context
from .tenant_llm_config import TenantLLMConfig, coalesce_tenant_key
from .tenant_openai_client import build_tenant_openai

# ``openai.OpenAI`` sync client with ``base_url`` set to OpenRouter — use for annotations.
OpenRouterSyncClient = OpenAI

_openai_async_lock = asyncio.Lock()
_async_singleton: Optional[AsyncOpenAI] = None


def _default_headers() -> Dict[str, str]:
    h: Dict[str, str] = {}
    site = (settings.OPENROUTER_SITE_URL or "").strip()
    title = (settings.OPENROUTER_APP_NAME or "").strip()
    if site:
        h["HTTP-Referer"] = site
    if title:
        h["X-Title"] = title
    return h


def ensure_openrouter_env_for_openai_sdk() -> bool:
    """
    Set ``OPENAI_API_KEY`` and ``OPENAI_BASE_URL`` from the shared OpenRouter
    key resolution so libraries that use the default OpenAI client (including
    the OpenAI Agents SDK) send traffic to OpenRouter with the same credentials
    as :func:`get_sync_openrouter`.
    """
    from .settings import primary_llm_api_key

    k = primary_llm_api_key()
    if not k:
        return False
    os.environ.setdefault(OPENROUTER_API_KEY_ENV, k)
    os.environ.setdefault(LEGACY_OPENAI_API_KEY_ENV, k)
    os.environ.setdefault("OPENAI_BASE_URL", settings.OPENROUTER_BASE_URL)
    return True


def _resolve_router_key() -> str:
    return (
        (settings.OPENROUTER_API_KEY or "").strip()
        or config(OPENROUTER_API_KEY_ENV, default="").strip()
        or config(LEGACY_OPENAI_API_KEY_ENV, default="").strip()
    )


def _sync_openai_for_key(key: str) -> OpenAI:
    return OpenAI(
        api_key=key,
        base_url=settings.OPENROUTER_BASE_URL,
        default_headers=_default_headers(),
    )


def openrouter_client_for_key(api_key: Optional[str] = None) -> OpenAI:
    """OpenRouter-compatible client; uses env key when api_key is omitted."""
    if api_key and str(api_key).strip():
        return _sync_openai_for_key(str(api_key).strip())
    return get_sync_openrouter()


@lru_cache(maxsize=32)
def _cached_sync_openai_for_key(key: str) -> OpenAI:
    return _sync_openai_for_key(key)


def get_tenant_sync_openrouter() -> OpenAI:
    """
    Primary sync OpenRouter client: tenant key + model policy + usage logging.
    When no :class:`LLMInvocationContext` is set, logs with ``tenant_id=null``.
    """
    inv = get_llm_invocation_context()
    cfg = TenantLLMConfig.for_organization(inv.tenant_id if inv else None)
    key = coalesce_tenant_key(cfg, OPENROUTER_API_KEY_ENV)
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Add it to .env for LLM access."
        )
    base = _cached_sync_openai_for_key(key)
    return build_tenant_openai(base, cfg, inv)


def get_sync_openrouter() -> OpenAI:
    """
    Back-compat alias: same as :func:`get_tenant_sync_openrouter` (tenant-aware + logging).
    Not cached at this layer because :class:`LLMInvocationContext` is per-request.
    """
    return get_tenant_sync_openrouter()


def try_get_sync_openrouter() -> Optional[OpenAI]:
    """Return OpenRouter client, or None if no key (graceful skip for optional LLM)."""
    key = _resolve_router_key()
    if not key or key.startswith("sk-your"):
        return None
    try:
        return get_tenant_sync_openrouter()
    except RuntimeError:
        return None


def sync_openrouter_with_timeout(timeout: float) -> OpenAI:
    """Same as get_tenant_sync_openrouter() but with HTTP timeout for long prompts."""
    return get_tenant_sync_openrouter().with_options(timeout=timeout)


def create_async_openrouter() -> AsyncOpenAI:
    """New AsyncOpenAI pointed at OpenRouter (for scripts; prefer get_async_openrouter in async apps)."""
    key = _resolve_router_key()
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Add it to .env for LLM access."
        )
    return AsyncOpenAI(
        api_key=key,
        base_url=settings.OPENROUTER_BASE_URL,
        default_headers=_default_headers(),
    )


async def get_async_openrouter() -> AsyncOpenAI:
    """Lazily construct a singleton async client (safe for FastAPI)."""
    global _async_singleton
    if _async_singleton is not None:
        return _async_singleton
    async with _openai_async_lock:
        if _async_singleton is None:
            key = _resolve_router_key()
            if not key:
                raise RuntimeError(
                    "OPENROUTER_API_KEY is not set. Add it to .env for LLM access."
                )
            _async_singleton = AsyncOpenAI(
                api_key=key,
                base_url=settings.OPENROUTER_BASE_URL,
                default_headers=_default_headers(),
            )
    return _async_singleton


def reset_async_client_for_tests() -> None:
    """Test helper: drop cached async client."""
    global _async_singleton
    _async_singleton = None
    _cached_sync_openai_for_key.cache_clear()

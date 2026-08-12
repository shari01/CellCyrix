"""
OpenAI Agents SDK + OpenRouter (full provider/model slugs).

The Agents SDK's default :class:`agents.models.multi_provider.MultiProvider` splits
``model`` on the first ``/`` and dispatches by prefix (``openai``, ``litellm``, …).
OpenRouter slugs like ``google/gemini-2.5-flash`` or ``anthropic/claude-sonnet-4.6``
therefore raise ``UserError: Unknown prefix: google``.

Use :func:`openrouter_run_config` as ``run_config=`` for :class:`agents.Runner` so
all traffic goes through :class:`agents.models.openai_chatcompletions.OpenAIChatCompletionsModel`
with the shared OpenRouter :class:`openai.AsyncOpenAI` client (full slug unchanged).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from agents import RunConfig
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from agents.models.openai_provider import OpenAIProvider
from openai import AsyncOpenAI

from . import settings as llm_settings
from .clients import create_async_openrouter


@lru_cache(maxsize=1)
def _async_openrouter_singleton() -> AsyncOpenAI:
    return create_async_openrouter()


def reset_agents_openrouter_client_cache() -> None:
    """Test helper: clear cached AsyncOpenAI for Agents SDK runs."""
    _async_openrouter_singleton.cache_clear()


def openrouter_run_config(
    *,
    model: str | None = None,
    tracing_disabled: bool = False,
    trace_metadata: dict[str, Any] | None = None,
) -> RunConfig:
    """
    Build :class:`agents.RunConfig` that routes any OpenRouter slug through chat completions.

    ``model`` defaults to :data:`agentic_ai_wf.llm.settings.LLM_MODEL_AGENT`.
    """
    m = (model or llm_settings.LLM_MODEL_AGENT or "").strip()
    if not m:
        m = llm_settings.LLM_MODEL_AGENT
    client = _async_openrouter_singleton()
    ocm = OpenAIChatCompletionsModel(model=m, openai_client=client)
    return RunConfig(
        model=ocm,
        model_provider=OpenAIProvider(openai_client=client),
        tracing_disabled=tracing_disabled,
        trace_metadata=trace_metadata,
    )

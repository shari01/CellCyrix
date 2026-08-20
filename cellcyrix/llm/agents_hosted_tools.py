"""
OpenAI Agents SDK — hosted tools (e.g. :class:`agents.tool.WebSearchTool`).

Hosted tools are **not** supported on Chat Completions. They require the
`OpenAI Responses API <https://platform.openai.com/docs/api-reference/responses>`_
via :class:`agents.models.openai_responses.OpenAIResponsesModel`, with an
**OpenAI platform** API key (not OpenRouter).

Pathway literature uses :func:`pathway_literature_run_config` and
:func:`pathway_literature_web_search_tools` so OpenRouter remains the default
for most agents, while web search is enabled when ``OPENAI_RESPONSES_API_KEY``
is set.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, List

from agents import RunConfig, WebSearchTool
from agents.models.openai_provider import OpenAIProvider
from agents.models.openai_responses import OpenAIResponsesModel
from decouple import config
from openai import AsyncOpenAI

from . import settings as llm_settings
from .env_names import OPENAI_RESPONSES_API_KEY_ENV


def openai_responses_api_key_set() -> bool:
    """True when a key for the OpenAI Responses API is configured."""
    return bool(config(OPENAI_RESPONSES_API_KEY_ENV, default="").strip())


@lru_cache(maxsize=1)
def _async_openai_platform_client() -> AsyncOpenAI:
    key = config(OPENAI_RESPONSES_API_KEY_ENV, default="").strip()
    if not key:
        raise RuntimeError(
            f"{OPENAI_RESPONSES_API_KEY_ENV} is not set. "
            "It is required for OpenAI Responses API and hosted tools (WebSearchTool)."
        )
    return AsyncOpenAI(api_key=key)


def reset_agents_hosted_tools_client_cache() -> None:
    """Drop the cached async platform client (tests, or after a key change)."""
    _async_openai_platform_client.cache_clear()


def openai_responses_run_config_for_hosted_tools(
    *,
    model: str | None = None,
    tracing_disabled: bool = False,
    trace_metadata: dict[str, Any] | None = None,
) -> RunConfig:
    """
    :class:`agents.RunConfig` using the Responses API — required for ``WebSearchTool`` and
    other hosted tools. ``model`` must be an **OpenAI platform** id (e.g. ``gpt-4o``), not
    an OpenRouter slug.
    """
    m = (model or llm_settings.LLM_MODEL_OPENAI_RESPONSES or "gpt-4o").strip()
    client = _async_openai_platform_client()
    orm = OpenAIResponsesModel(model=m, openai_client=client)
    return RunConfig(
        model=orm,
        model_provider=OpenAIProvider(openai_client=client, use_responses=True),
        tracing_disabled=tracing_disabled,
        trace_metadata=trace_metadata,
    )


def pathway_literature_web_search_tools() -> List[Any]:
    """Return ``[WebSearchTool()]`` when platform key is set; else ``[]`` (OpenRouter path)."""
    if not openai_responses_api_key_set():
        return []
    return [WebSearchTool()]


def pathway_literature_run_config() -> RunConfig:
    """
    Pathway ``litrature`` agents: use Responses + hosted search when configured; else OpenRouter.
    """
    if openai_responses_api_key_set():
        return openai_responses_run_config_for_hosted_tools(
            model=llm_settings.LLM_MODEL_OPENAI_RESPONSES,
        )
    from .agents_openrouter import openrouter_run_config

    return openrouter_run_config(model=llm_settings.LLM_MODEL_AGENT)

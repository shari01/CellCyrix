"""Canonical process/.env names for LLM credentials.

Import these names for ``os.environ`` / ``decouple.config`` lookups. Do not
duplicate the legacy string elsewhere in the codebase.
"""

from __future__ import annotations

OPENROUTER_API_KEY_ENV: str = "OPENROUTER_API_KEY"
# Deployments that predate OpenRouter may still export this; handled only via
# :func:`cellcyrix.llm.settings.primary_llm_api_key`.
LEGACY_OPENAI_API_KEY_ENV: str = "OPENAI_API_KEY"
# Native OpenAI API (api.openai.com) for Agents SDK hosted tools (WebSearch, etc.)
# — not OpenRouter; required for :class:`agents.tool.WebSearchTool` with Responses API.
OPENAI_RESPONSES_API_KEY_ENV: str = "OPENAI_API_KEY"

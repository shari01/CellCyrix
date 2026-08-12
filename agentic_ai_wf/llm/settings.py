"""OpenRouter and LLM model configuration (decouple / .env)."""

from __future__ import annotations

from decouple import config

from .env_names import LEGACY_OPENAI_API_KEY_ENV, OPENROUTER_API_KEY_ENV

OPENROUTER_API_KEY: str = config(OPENROUTER_API_KEY_ENV, default="")
OPENROUTER_BASE_URL: str = config(
    "OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1"
)
OPENROUTER_SITE_URL: str = config("OPENROUTER_SITE_URL", default="")
OPENROUTER_APP_NAME: str = config("OPENROUTER_APP_NAME", default="agenticaib")


def parse_model_csv(raw: str) -> list[str]:
    """Comma-separated OpenRouter slugs (whitespace-tolerant, empty segments dropped)."""
    return [p.strip() for p in (raw or "").split(",") if p.strip()]


def _csv_model_list(env_key: str, default: str) -> list[str]:
    return parse_model_csv(config(env_key, default=default))


# Logical tiers — values must be OpenRouter model slugs (see openrouter.ai/models)
LLM_MODEL_CHAT: str = config("LLM_MODEL_CHAT", default="openai/gpt-4o-mini")
LLM_MODEL_FAST: str = config("LLM_MODEL_FAST", default="openai/gpt-5-mini")
LLM_MODEL_HEAVY: str = config("LLM_MODEL_HEAVY", default="anthropic/claude-sonnet-4.6")
# Multi-step tool / Agents SDK (cost-efficient default; escalate via fallbacks on failure)
LLM_MODEL_AGENT: str = config("LLM_MODEL_AGENT", default="anthropic/claude-haiku-4.5")
# Short structured routing / classification
LLM_MODEL_ROUTING: str = config("LLM_MODEL_ROUTING", default="openai/gpt-4.1-nano")

# Fallback chains: comma-separated slugs, tried in order when the primary is overloaded / 5xx / 429
# (ADR: chat→4.1-mini, fast→4.1-nano, heavy→2.5-pro, agent→Sonnet→4.1-mini, routing→flash-lite→4o-mini)
LLM_MODEL_CHAT_FALLBACKS: list[str] = _csv_model_list(
    "LLM_MODEL_CHAT_FALLBACKS", "openai/gpt-4.1-mini"
)
LLM_MODEL_FAST_FALLBACKS: list[str] = _csv_model_list(
    "LLM_MODEL_FAST_FALLBACKS", "openai/gpt-4.1-nano"
)
LLM_MODEL_HEAVY_FALLBACKS: list[str] = _csv_model_list(
    "LLM_MODEL_HEAVY_FALLBACKS", "google/gemini-2.5-pro"
)
LLM_MODEL_AGENT_FALLBACKS: list[str] = _csv_model_list(
    "LLM_MODEL_AGENT_FALLBACKS",
    "anthropic/claude-sonnet-4.6,openai/gpt-4.1-mini",
)
LLM_MODEL_ROUTING_FALLBACKS: list[str] = _csv_model_list(
    "LLM_MODEL_ROUTING_FALLBACKS",
    "google/gemini-2.5-flash-lite,openai/gpt-4o-mini",
)

# Specific models
LLM_MODEL_GPT_5_MINI: str = config("LLM_MODEL_GPT_5_MINI", default="openai/gpt-5-mini")
# OpenAI platform model id (not an OpenRouter slug) for Responses API + hosted tools
LLM_MODEL_OPENAI_RESPONSES: str = config(
    "LLM_MODEL_OPENAI_RESPONSES", default="gpt-5-mini"
)

LLM_MODEL_ANTHROPIC_CLAUDE_OPUS_4_7: str = config(
    "LLM_MODEL_ANTHROPIC_CLAUDE_OPUS_4_7", default="anthropic/claude-opus-4.7"
)

USE_OPENROUTER: bool = config("USE_OPENROUTER", default="true", cast=bool)

# General chat / insights synthesis (no pipeline-data path); supervisor + portal
LLM_RESPONSES_MODEL: str = config(
    "LLM_RESPONSES_MODEL", default="anthropic/claude-haiku-4.5"
)  # anthropic/claude-opus-4.7


def litellm_model_for_openrouter(model_slug: str) -> str:
    """
    Model id for litellm `completion()` when `api_base` is OpenRouter.
    Slugs like ``google/gemini-2.5-flash-lite`` are ambiguous without a provider;
    use ``openrouter/<slug>`` so litellm dispatches correctly (PandasAI / pandasai-litellm).
    """
    s = (model_slug or "").strip()
    if not s or not USE_OPENROUTER:
        return s
    if s.startswith("openrouter/") or s.startswith("vercel_ai_gateway/"):
        return s
    if "/" not in s:
        return s
    return f"openrouter/{s}"


def primary_llm_api_key() -> str:
    """Prefer OpenRouter key, then legacy-compatible env (decouple / runtime)."""
    return (
        config(OPENROUTER_API_KEY_ENV, default="").strip()
        or config(LEGACY_OPENAI_API_KEY_ENV, default="").strip()
    )

"""Resolved model IDs for OpenRouter (no hardcoded provider SDKs)."""

from __future__ import annotations

from typing import Literal, Optional, Sequence, get_args

from . import settings

ModelTier = Literal["chat", "fast", "heavy", "agent", "routing"]

_TIERS: tuple[str, ...] = get_args(ModelTier)


def _dedupe_preserve_order(slugs: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for s in slugs:
        m = (s or "").strip()
        if not m or m in seen:
            continue
        seen.add(m)
        out.append(m)
    return out


def models_for_tier(tier: ModelTier) -> list[str]:
    """
    Primary model for ``tier`` plus configured fallbacks (OpenRouter slugs).

    Order: try primary first, then each fallback on retriable provider errors.
    """
    if tier not in _TIERS:
        raise ValueError(f"unknown tier: {tier!r}; expected one of {_TIERS}")
    if tier == "chat":
        return _dedupe_preserve_order(
            [settings.LLM_MODEL_CHAT, *settings.LLM_MODEL_CHAT_FALLBACKS]
        )
    if tier == "fast":
        return _dedupe_preserve_order(
            [settings.LLM_MODEL_FAST, *settings.LLM_MODEL_FAST_FALLBACKS]
        )
    if tier == "heavy":
        return _dedupe_preserve_order(
            [settings.LLM_MODEL_HEAVY, *settings.LLM_MODEL_HEAVY_FALLBACKS]
        )
    if tier == "agent":
        return _dedupe_preserve_order(
            [settings.LLM_MODEL_AGENT, *settings.LLM_MODEL_AGENT_FALLBACKS]
        )
    # routing
    primary = (settings.LLM_MODEL_ROUTING or "").strip() or settings.LLM_MODEL_FAST
    return _dedupe_preserve_order([primary, *settings.LLM_MODEL_ROUTING_FALLBACKS])


def routing_model() -> str:
    """Primary model for intent routing / short JSON (not the full fallback chain)."""
    return (settings.LLM_MODEL_ROUTING or "").strip() or settings.LLM_MODEL_FAST


def resolve_chat_model(explicit: Optional[str] = None) -> str:
    """Model id for interactive chat: `explicit`, else the configured default."""
    if explicit and explicit.strip():
        return explicit.strip()
    return settings.LLM_MODEL_CHAT


def resolve_heavy_model(explicit: Optional[str] = None) -> str:
    """Model id for heavy reasoning: `explicit`, else the configured default."""
    if explicit and explicit.strip():
        return explicit.strip()
    return settings.LLM_MODEL_HEAVY


def resolve_fast_model(explicit: Optional[str] = None) -> str:
    """Model id for cheap/fast calls: `explicit`, else the configured default."""
    if explicit and explicit.strip():
        return explicit.strip()
    return settings.LLM_MODEL_FAST


def resolve_agent_model(explicit: Optional[str] = None) -> str:
    """Model id for agent loops: `explicit`, else the configured default."""
    if explicit and explicit.strip():
        return explicit.strip()
    return settings.LLM_MODEL_AGENT


def resolve_responses_model(explicit: Optional[str] = None) -> str:
    """Primary model for responses (no-pipeline-data) chat synthesis."""
    if explicit and explicit.strip():
        return explicit.strip()
    return settings.LLM_RESPONSES_MODEL


def pick_model_for_call(
    *,
    model_id: Optional[str],
    has_tools: bool,
) -> str:
    """Default single model when caller does not use tier fallback chains."""
    if model_id and model_id.strip():
        return model_id.strip()
    # Tool + agent loops: agent tier (not heavy) for cost/behavior alignment
    if has_tools:
        return resolve_agent_model()
    return resolve_chat_model()


__all__ = [
    "ModelTier",
    "models_for_tier",
    "pick_model_for_call",
    "resolve_agent_model",
    "resolve_chat_model",
    "resolve_fast_model",
    "resolve_responses_model",
    "resolve_heavy_model",
    "routing_model",
]

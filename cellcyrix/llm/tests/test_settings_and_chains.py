"""Unit tests for LLM tier primaries, fallback chains, and resolvers."""

from __future__ import annotations

import re

import pytest

from cellcyrix.llm import settings
from cellcyrix.llm.models import (
    models_for_tier,
    pick_model_for_call,
    resolve_agent_model,
    resolve_chat_model,
    resolve_fast_model,
    resolve_heavy_model,
    routing_model,
)

_SLUG = re.compile(r"^[a-z0-9][a-z0-9._\-]*/[a-z0-9][a-z0-9._\-/]*$", re.IGNORECASE)


def test_parse_model_csv_trims_and_skips_empties() -> None:
    assert settings.parse_model_csv("") == []
    assert settings.parse_model_csv("  a/b  ,  , c/d  ") == ["a/b", "c/d"]


def test_litellm_model_for_openrouter_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "USE_OPENROUTER", True, raising=False)
    assert settings.litellm_model_for_openrouter("google/gemini-2.5-flash-lite") == (
        "openrouter/google/gemini-2.5-flash-lite"
    )
    assert (
        settings.litellm_model_for_openrouter("openrouter/google/gemini-2.5-flash-lite")
        == "openrouter/google/gemini-2.5-flash-lite"
    )
    monkeypatch.setattr(settings, "USE_OPENROUTER", False, raising=False)
    assert settings.litellm_model_for_openrouter("google/gemini-2.5-flash-lite") == (
        "google/gemini-2.5-flash-lite"
    )


def test_settings_loaded_lists_are_strings_and_nonempty() -> None:
    for name in (
        "LLM_MODEL_CHAT",
        "LLM_MODEL_FAST",
        "LLM_MODEL_HEAVY",
        "LLM_MODEL_AGENT",
        "LLM_MODEL_ROUTING",
    ):
        v = getattr(settings, name)
        assert isinstance(v, str) and v.strip() == v and len(v) > 0
    for name in (
        "LLM_MODEL_CHAT_FALLBACKS",
        "LLM_MODEL_FAST_FALLBACKS",
        "LLM_MODEL_HEAVY_FALLBACKS",
        "LLM_MODEL_AGENT_FALLBACKS",
        "LLM_MODEL_ROUTING_FALLBACKS",
    ):
        lst = getattr(settings, name)
        assert isinstance(lst, list) and all(isinstance(x, str) and x for x in lst)


def test_models_for_tier_respects_order_and_dedupes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cellcyrix.llm.settings as s

    monkeypatch.setattr(s, "LLM_MODEL_CHAT", "p/x", raising=False)
    monkeypatch.setattr(
        s, "LLM_MODEL_CHAT_FALLBACKS", ["p/x", "f/1", "f/1", "f/2"], raising=False
    )
    assert models_for_tier("chat") == ["p/x", "f/1", "f/2"]


def test_models_for_tier_per_tier_wiring(monkeypatch: pytest.MonkeyPatch) -> None:
    import cellcyrix.llm.settings as s

    monkeypatch.setattr(s, "LLM_MODEL_CHAT", "c/1", raising=False)
    monkeypatch.setattr(s, "LLM_MODEL_CHAT_FALLBACKS", ["c/2"], raising=False)
    assert models_for_tier("chat") == ["c/1", "c/2"]

    monkeypatch.setattr(s, "LLM_MODEL_FAST", "f/1", raising=False)
    monkeypatch.setattr(s, "LLM_MODEL_FAST_FALLBACKS", ["f/2"], raising=False)
    assert models_for_tier("fast") == ["f/1", "f/2"]

    monkeypatch.setattr(s, "LLM_MODEL_HEAVY", "h/1", raising=False)
    monkeypatch.setattr(s, "LLM_MODEL_HEAVY_FALLBACKS", ["h/2"], raising=False)
    assert models_for_tier("heavy") == ["h/1", "h/2"]

    monkeypatch.setattr(s, "LLM_MODEL_AGENT", "a/1", raising=False)
    monkeypatch.setattr(s, "LLM_MODEL_AGENT_FALLBACKS", ["a/2", "a/3"], raising=False)
    assert models_for_tier("agent") == ["a/1", "a/2", "a/3"]


def test_models_for_tier_routing_uses_fast_if_routing_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cellcyrix.llm.settings as s

    monkeypatch.setattr(s, "LLM_MODEL_ROUTING", "", raising=False)
    monkeypatch.setattr(s, "LLM_MODEL_FAST", "fast/primary", raising=False)
    monkeypatch.setattr(s, "LLM_MODEL_ROUTING_FALLBACKS", ["r/1"], raising=False)
    assert models_for_tier("routing") == ["fast/primary", "r/1"]


def test_routing_model_matches_primary_slug(monkeypatch: pytest.MonkeyPatch) -> None:
    import cellcyrix.llm.settings as s

    monkeypatch.setattr(s, "LLM_MODEL_ROUTING", "r/p", raising=False)
    monkeypatch.setattr(s, "LLM_MODEL_FAST", "f/p", raising=False)
    assert routing_model() == "r/p"
    monkeypatch.setattr(s, "LLM_MODEL_ROUTING", "  ", raising=False)
    assert routing_model() == "f/p"


def test_models_for_tier_invalid() -> None:
    with pytest.raises(ValueError, match="unknown tier"):
        models_for_tier("not_a_tier")  # type: ignore[arg-type]


def test_resolve_model_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    import cellcyrix.llm.settings as s

    monkeypatch.setattr(s, "LLM_MODEL_CHAT", "cc/1", raising=False)
    monkeypatch.setattr(s, "LLM_MODEL_FAST", "ff/1", raising=False)
    assert resolve_chat_model("x/y") == "x/y"
    assert resolve_chat_model(None) == "cc/1"
    assert resolve_fast_model(None) == "ff/1"
    assert resolve_heavy_model(None) == s.LLM_MODEL_HEAVY
    assert resolve_agent_model(None) == s.LLM_MODEL_AGENT


def test_pick_model_for_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cellcyrix.llm.settings as s

    assert pick_model_for_call(model_id="custom/z", has_tools=True) == "custom/z"
    monkeypatch.setattr(s, "LLM_MODEL_CHAT", "c/p", raising=False)
    monkeypatch.setattr(s, "LLM_MODEL_AGENT", "a/p", raising=False)
    assert pick_model_for_call(model_id=None, has_tools=False) == "c/p"
    assert pick_model_for_call(model_id=None, has_tools=True) == "a/p"


def _tier_names() -> tuple[str, ...]:
    from typing import get_args

    from cellcyrix.llm.models import ModelTier as MT

    return get_args(MT)


@pytest.mark.parametrize("tier", _tier_names())
def test_default_chains_for_each_tier(tier: str) -> None:
    chain = models_for_tier(tier)  # type: ignore[arg-type]
    assert len(chain) >= 1
    for m in chain:
        assert _SLUG.match(m), m

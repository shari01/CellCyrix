"""
Live OpenRouter calls (real network, real spend).

What the *other* tests in this package do *not* cover:
  - They mock ``chat.completions.create`` and only check config + failover *logic*.

This module sends a **minimal** user message to each tier’s **primary** model
and asserts a non-empty assistant reply. Use it to verify slugs, keys, and
connectivity.

Run (incurs small token cost; requires ``OPENROUTER_API_KEY`` or legacy key resolution)::

    RUN_LIVE_LLM_TESTS=1 pytest agentic_ai_wf/llm/tests/test_openrouter_live.py -m integration -v

Skip pricier tiers (cheaper smoke)::

    RUN_LIVE_LLM_TESTS=1 RUN_LIVE_LLM_EXCLUDE=heavy,agent pytest agentic_ai_wf/llm/tests/test_openrouter_live.py -m integration -v
"""

from __future__ import annotations

import os
from typing import get_args

import pytest

from agentic_ai_wf.llm.models import ModelTier, models_for_tier
from agentic_ai_wf.llm.settings import primary_llm_api_key

_TIERS: tuple[str, ...] = get_args(ModelTier)

pytestmark = pytest.mark.integration

_PROMPT = "Reply in one short sentence: name one DNA base (A, T, C, or G)."


def _live_enabled() -> bool:
    return os.environ.get("RUN_LIVE_LLM_TESTS", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _api_key_ok() -> bool:
    k = primary_llm_api_key()
    return bool(k) and not k.startswith("sk-your") and "placeholder" not in k.lower()


def _excluded_tiers() -> set[str]:
    raw = os.environ.get("RUN_LIVE_LLM_EXCLUDE", "").strip()
    if not raw:
        return set()
    return {x.strip().lower() for x in raw.split(",") if x.strip()}


skip_unless_live = pytest.mark.skipif(
    not _live_enabled() or not _api_key_ok(),
    reason="Set RUN_LIVE_LLM_TESTS=1 and a valid OPENROUTER_API_KEY (or legacy) to run; costs money",
)


@skip_unless_live
@pytest.mark.parametrize("tier", _TIERS)
def test_tier_primary_model_returns_text(tier: str) -> None:
    if tier.lower() in _excluded_tiers():
        pytest.skip(f"tier in RUN_LIVE_LLM_EXCLUDE: {tier}")
    from agentic_ai_wf.llm.clients import get_sync_openrouter

    model = models_for_tier(tier)[0]  # type: ignore[arg-type]
    client = get_sync_openrouter()
    out = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": _PROMPT}],
        max_tokens=64,
        temperature=0.2,
    )
    text = (out.choices[0].message.content or "").strip()
    assert text, f"empty content from {model!r} ({tier!r})"


@skip_unless_live
def test_tier_routing_default_chain_first_model_matches_settings() -> None:
    """Sanity: routing tier primary matches env defaults (or your .env)."""
    from agentic_ai_wf.llm import settings as s

    primary = models_for_tier("routing")[0]
    expect = (s.LLM_MODEL_ROUTING or "").strip() or s.LLM_MODEL_FAST
    assert primary == expect

"""
test_llm_determinism.py — the LLM voter's reproducibility guarantees.

The knowledge-based voter calls a hosted model, which is the pipeline's one
non-deterministic dependency and the one a reviewer will press hardest on. Three
properties make it defensible, and these tests pin all three:

1. **Greedy decoding is requested.** ``temperature`` 0.0 and ``top_p`` 1.0 are sent,
   plus a fixed ``seed``. Necessary but not sufficient — a provider can change the
   weights behind a floating alias regardless of sampling parameters.
2. **Responses are cached, keyed by everything that affects them.** The second
   identical request makes no network call, so a published run is replayable exactly,
   offline, from the shipped cache directory.
3. **The key is honest.** Changing the prompt, the model, or any sampling parameter is
   a cache miss. A cache that silently returned a response to a different question
   would be worse than no cache at all.

No network and no credentials: the transport is stubbed, and the assertion that matters
most is on the stub's call count.
"""

from __future__ import annotations

import json

import pytest

from cellcyrix.single_cell_pipeline_agent.singlecell_10x.celltype_consensus import (
    llm_cache,
)
from cellcyrix.single_cell_pipeline_agent.singlecell_10x.celltype_consensus.config import (  # noqa: E501
    ConsensusConfig,
)

CACHE_ENV = llm_cache.ENV_CACHE_DIR
ENABLE_ENV = llm_cache.ENV_ENABLED


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    """Point the cache at a temp directory and reset counters around each test."""
    target = tmp_path / "llm_cache"
    monkeypatch.setenv(CACHE_ENV, str(target))
    monkeypatch.delenv(ENABLE_ENV, raising=False)
    llm_cache.reset_stats()
    yield target
    llm_cache.reset_stats()


@pytest.fixture
def cfg():
    """A config with credentials present so the transport path is reachable."""
    return ConsensusConfig(
        openrouter_api_key="test-key-not-real",
        openrouter_model="test/model-v1",
        llm_temperature=0.0,
        llm_max_tokens=800,
        llm_top_p=1.0,
        llm_seed=0,
        llm_max_retries=1,
    )


class _StubResponse:
    """Minimal stand-in for a `requests` response carrying one chat completion."""

    def __init__(self, content: str, served_model: str = "test/model-v1-20260101"):
        self._payload = {
            "model": served_model,
            "choices": [{"message": {"content": content}}],
        }

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


@pytest.fixture
def stub_transport(monkeypatch):
    """Replace `requests.post` with a counting stub; yields the call log."""
    import requests

    calls: list[dict] = []

    def _post(url, headers=None, json=None, timeout=None):  # noqa: A002 - requests' API
        calls.append({"url": url, "headers": headers, "body": json, "timeout": timeout})
        return _StubResponse('{"cell_type":"T cell","confidence":0.9}')

    monkeypatch.setattr(requests, "post", _post)
    return calls


def _chat(cfg, system="SYS", user="USER"):
    """Call the private transport helper under test."""
    from cellcyrix.single_cell_pipeline_agent.singlecell_10x.celltype_consensus.agent import (  # noqa: E501
        _chat as chat,
    )

    return chat(cfg, system, user)


# --------------------------------------------------------------------------------------
# 1. Greedy decoding is actually requested
# --------------------------------------------------------------------------------------


def test_request_pins_temperature_top_p_and_seed(cfg, cache_dir, stub_transport):
    """The wire request carries temperature 0, top_p 1, and a fixed seed."""
    _chat(cfg)
    assert len(stub_transport) == 1
    body = stub_transport[0]["body"]
    assert body["temperature"] == 0.0
    assert body["top_p"] == 1.0
    assert body["seed"] == 0
    assert body["max_tokens"] == 800
    assert body["model"] == "test/model-v1"


def test_config_defaults_are_greedy():
    """A default config is deterministic-by-default, not opt-in."""
    default = ConsensusConfig()
    assert default.llm_temperature == 0.0
    assert default.llm_top_p == 1.0
    assert default.llm_seed == 0


# --------------------------------------------------------------------------------------
# 2. The second identical call makes no network request
# --------------------------------------------------------------------------------------


def test_identical_request_is_served_from_cache(cfg, cache_dir, stub_transport):
    """Two identical calls -> one network call, identical results."""
    first = _chat(cfg)
    second = _chat(cfg)

    assert first == second
    assert len(stub_transport) == 1, (
        "the second identical request hit the network; the run is not reproducible"
    )
    stats = llm_cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["writes"] == 1


def test_cached_run_needs_no_credentials(cfg, cache_dir, stub_transport):
    """A fully-cached run works with no API key — offline reproduction of a result."""
    _chat(cfg)
    assert len(stub_transport) == 1

    keyless = ConsensusConfig(
        openrouter_api_key=None,
        openrouter_model="test/model-v1",
        llm_temperature=0.0,
        llm_max_tokens=800,
        llm_top_p=1.0,
        llm_seed=0,
    )
    # Without the cache this raises OpenRouterError on the missing key.
    assert _chat(keyless) == '{"cell_type":"T cell","confidence":0.9}'
    assert len(stub_transport) == 1, "a cached run must not call out at all"


def test_fully_cached_is_reported_in_stats(cfg, cache_dir, stub_transport):
    """`fully_cached` distinguishes a replayed run from one that queried."""
    _chat(cfg)
    assert llm_cache.stats()["fully_cached"] is False  # first run had a miss

    llm_cache.reset_stats()
    _chat(cfg)
    stats = llm_cache.stats()
    assert stats["fully_cached"] is True
    assert stats["lookups"] == 1
    assert stats["misses"] == 0


def test_served_model_is_recorded(cfg, cache_dir, stub_transport):
    """The entry records which model the endpoint actually served, not just requested.

    With a floating alias those differ, and the served id is what identifies the
    weights that produced a label.
    """
    _chat(cfg)
    entries = list(cache_dir.rglob("*.json"))
    assert len(entries) == 1
    record = json.loads(entries[0].read_text(encoding="utf-8"))
    assert record["model_requested"] == "test/model-v1"
    assert record["model_served"] == "test/model-v1-20260101"
    assert record["temperature"] == 0.0
    assert record["top_p"] == 1.0
    assert record["seed"] == 0
    # A cache entry must never carry the credential that fetched it.
    assert "test-key-not-real" not in entries[0].read_text(encoding="utf-8")


# --------------------------------------------------------------------------------------
# 3. The key covers everything that changes the answer
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field,value",
    [
        ("openrouter_model", "test/model-v2"),
        ("llm_temperature", 0.7),
        ("llm_top_p", 0.9),
        ("llm_seed", 1),
        ("llm_max_tokens", 400),
    ],
)
def test_changing_any_sampling_parameter_misses(
    cfg, cache_dir, stub_transport, field, value
):
    """A different model or sampling parameter is a miss, never a stale hit."""
    _chat(cfg)
    assert len(stub_transport) == 1

    changed = ConsensusConfig(**{**vars(cfg), field: value})
    _chat(changed)
    assert len(stub_transport) == 2, (
        f"changing {field} wrongly reused a cached response"
    )


@pytest.mark.parametrize(
    "system,user",
    [("SYS-CHANGED", "USER"), ("SYS", "USER-CHANGED")],
)
def test_changing_either_prompt_misses(cfg, cache_dir, stub_transport, system, user):
    """Editing a prompt invalidates the entry — the old answer is to a different question."""
    _chat(cfg)
    assert len(stub_transport) == 1

    _chat(cfg, system=system, user=user)
    assert len(stub_transport) == 2, "a changed prompt reused a cached response"


def test_key_is_stable_across_calls():
    """The same inputs always hash to the same key (no dict-ordering dependence)."""
    kwargs = dict(
        model="m",
        system="s",
        user="u",
        temperature=0.0,
        max_tokens=800,
        top_p=1.0,
        seed=0,
    )
    assert llm_cache.cache_key(**kwargs) == llm_cache.cache_key(**kwargs)


def test_schema_version_participates_in_key(monkeypatch):
    """Bumping the record schema is a clean miss, not a misread of old entries."""
    kwargs = dict(
        model="m",
        system="s",
        user="u",
        temperature=0.0,
        max_tokens=800,
        top_p=1.0,
        seed=0,
    )
    before = llm_cache.cache_key(**kwargs)
    monkeypatch.setattr(llm_cache, "CACHE_SCHEMA_VERSION", 999)
    assert llm_cache.cache_key(**kwargs) != before


# --------------------------------------------------------------------------------------
# Cache mechanics: opt-out, and failure modes that must not break a run
# --------------------------------------------------------------------------------------


def test_cache_can_be_disabled(cfg, cache_dir, stub_transport, monkeypatch):
    """SCPIPE_LLM_CACHE=0 restores uncached behaviour: every call goes out."""
    monkeypatch.setenv(ENABLE_ENV, "0")
    assert llm_cache.is_enabled() is False
    _chat(cfg)
    _chat(cfg)
    assert len(stub_transport) == 2
    assert not list(cache_dir.rglob("*.json")), "nothing should be written when off"


def test_corrupt_entry_is_a_miss_not_a_crash(cfg, cache_dir, stub_transport):
    """A truncated cache file re-queries rather than failing the run."""
    _chat(cfg)
    entry = next(iter(cache_dir.rglob("*.json")))
    entry.write_text("{ this is not json", encoding="utf-8")

    llm_cache.reset_stats()
    assert _chat(cfg) == '{"cell_type":"T cell","confidence":0.9}'
    assert len(stub_transport) == 2, (
        "a corrupt entry should fall through to the network"
    )
    assert llm_cache.stats()["errors"] == 1


def test_missing_key_without_cache_still_raises(cache_dir, stub_transport):
    """The credential check is not bypassed on a cache miss."""
    from cellcyrix.single_cell_pipeline_agent.singlecell_10x.celltype_consensus.agent import (  # noqa: E501
        OpenRouterError,
    )

    keyless = ConsensusConfig(
        openrouter_api_key=None,
        openrouter_model="test/model-v1",
        llm_temperature=0.0,
        llm_max_tokens=800,
    )
    with pytest.raises(OpenRouterError, match="not configured"):
        _chat(keyless, system="UNSEEN", user="UNSEEN")
    assert not stub_transport, "no request should be attempted without a key"


def test_stats_reset(cache_dir):
    """reset_stats zeroes the counters."""
    llm_cache.reset_stats()
    stats = llm_cache.stats()
    assert stats["hits"] == stats["misses"] == stats["writes"] == stats["errors"] == 0
    assert stats["fully_cached"] is False

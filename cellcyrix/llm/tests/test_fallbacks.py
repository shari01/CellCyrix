"""Unit tests for OpenRouter failover helpers (no network)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AsyncOpenAI,
    OpenAI,
    RateLimitError,
)

from cellcyrix.llm.fallbacks import (
    async_chat_completions_with_fallbacks,
    chat_completions_with_fallbacks,
    should_failover,
)


class _CodeExc(Exception):
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


def test_should_failover_rate_limit() -> None:
    assert should_failover(
        RateLimitError("r", response=Mock(status_code=429), body=None)
    )


def test_should_failover_by_status_code() -> None:
    for code in (404, 429, 500, 502, 503, 529):
        assert should_failover(_CodeExc(code))
    assert not should_failover(_CodeExc(401))
    assert not should_failover(_CodeExc(400))


def test_should_failover_api_error() -> None:
    e = APIError("x", request=Mock(), body=None)
    e.status_code = 503
    assert should_failover(e)


def test_should_failover_connection_timeout() -> None:
    req = Mock()
    assert should_failover(APIConnectionError(message="c", request=req))
    assert should_failover(APITimeoutError(request=req))


def test_chat_completions_with_fallbacks_first_succeeds() -> None:
    client = Mock(spec=OpenAI)
    ok = SimpleNamespace(choices=[Mock()])
    client.chat.completions.create = Mock(return_value=ok)
    r, m, fb = chat_completions_with_fallbacks(
        client, ["a/1", "a/2"], messages=[{"role": "user", "content": "x"}]
    )
    assert r is ok and m == "a/1" and fb is False
    assert client.chat.completions.create.call_count == 1


def test_chat_completions_with_fallbacks_reties_on_429() -> None:
    client = Mock(spec=OpenAI)
    ok = SimpleNamespace(choices=[Mock()])
    client.chat.completions.create = Mock(
        side_effect=[
            RateLimitError("r", response=Mock(status_code=429), body=None),
            ok,
        ]
    )
    r, m, fb = chat_completions_with_fallbacks(
        client, ["a/1", "a/2"], messages=[{"role": "user", "content": "x"}]
    )
    assert r is ok and m == "a/2" and fb is True
    assert client.chat.completions.create.call_count == 2


def test_chat_completions_with_fallbacks_empty_chain() -> None:
    client = Mock(spec=OpenAI)
    with pytest.raises(ValueError, match="empty"):
        chat_completions_with_fallbacks(client, [], messages=[])


def test_chat_completions_with_fallbacks_does_not_failover_on_401() -> None:
    client = Mock(spec=OpenAI)
    client.chat.completions.create = Mock(side_effect=_CodeExc(401))
    with pytest.raises(_CodeExc):
        chat_completions_with_fallbacks(client, ["a/1", "a/2"], messages=[])
    assert client.chat.completions.create.call_count == 1


def test_async_chat_completions_with_fallbacks() -> None:
    client = Mock(spec=AsyncOpenAI)
    ok = SimpleNamespace(choices=[Mock()])
    call_n = 0

    async def _acreate(*_a: object, **_kw: object) -> SimpleNamespace:
        nonlocal call_n
        call_n += 1
        if call_n == 1:
            raise RateLimitError("r", response=Mock(status_code=429), body=None)
        return ok

    client.chat.completions.create = _acreate

    async def _run() -> None:
        r, m, fb = await async_chat_completions_with_fallbacks(
            client, ["a/1", "a/2"], messages=[{"role": "user", "content": "x"}]
        )
        assert r is ok and m == "a/2" and fb is True

    asyncio.run(_run())

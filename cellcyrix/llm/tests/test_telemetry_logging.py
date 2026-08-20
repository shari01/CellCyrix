"""Tests for LLM telemetry console log gating."""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest


@pytest.mark.parametrize(
    "env,expected",
    [
        (None, "quiet"),
        ("quiet", "quiet"),
        ("true", "true"),
        ("false", "false"),
    ],
)
def test_global_token_log_mode(monkeypatch, env, expected):
    from cellcyrix.llm.telemetry_logging import global_token_log_mode

    if env is None:
        monkeypatch.delenv("SUPERVISOR_GLOBAL_TOKEN_LOG", raising=False)
    else:
        monkeypatch.setenv("SUPERVISOR_GLOBAL_TOKEN_LOG", env)
    assert global_token_log_mode() == expected


def test_llm_usage_logger_skips_console_by_default(monkeypatch, caplog):
    monkeypatch.delenv("SUPERVISOR_GLOBAL_TOKEN_LOG", raising=False)
    monkeypatch.delenv("LLM_USAGE_LOG_ENABLED", raising=False)
    caplog.set_level(logging.INFO, logger="agenticaib.llm_usage")

    from cellcyrix.llm.llm_usage_logger import log_llm_invocation

    with patch("core.services.tenant_audit.record_tenant_audit_event") as mock_audit:
        log_llm_invocation(
            provider="openrouter",
            model="openai/gpt-5-mini",
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
            tenant_id=None,
            user_id=None,
            phi_policy=True,
        )

    assert not [r for r in caplog.records if "llm_invocation" in r.message]
    mock_audit.assert_called_once()


def test_llm_usage_logger_emits_when_enabled(monkeypatch, caplog):
    monkeypatch.setenv("LLM_USAGE_LOG_ENABLED", "true")
    caplog.set_level(logging.INFO, logger="agenticaib.llm_usage")

    from cellcyrix.llm.llm_usage_logger import log_llm_invocation

    with patch("core.services.tenant_audit.record_tenant_audit_event"):
        log_llm_invocation(
            provider="openrouter",
            model="openai/gpt-5-mini",
            input_tokens=1,
            output_tokens=2,
            total_tokens=3,
            tenant_id=None,
            user_id=None,
            phi_policy=True,
        )

    assert any("llm_invocation" in r.message for r in caplog.records)

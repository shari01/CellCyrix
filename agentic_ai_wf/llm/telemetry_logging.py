"""Env-driven controls for high-volume LLM telemetry console logs."""

from __future__ import annotations

import os

_OFF = frozenset({"false", "0", "off", "no", "disabled"})
_VERBOSE = frozenset({"true", "1", "verbose", "full", "yes", "on"})


def global_token_log_mode() -> str:
    """``quiet`` (default): hooks on, no per-call INFO spam. ``false``: hooks off. ``true``: verbose."""
    return (os.environ.get("SUPERVISOR_GLOBAL_TOKEN_LOG") or "quiet").strip().lower()


def global_token_hooks_enabled() -> bool:
    """True when token-accounting hooks should be installed at all."""
    return global_token_log_mode() not in _OFF


def global_token_console_verbose() -> bool:
    """True when token accounting should also be echoed to the console."""
    return global_token_log_mode() in _VERBOSE


def llm_usage_console_enabled() -> bool:
    """True when per-call usage records should be echoed to the console.

    ``LLM_USAGE_LOG_ENABLED`` wins when set; otherwise this follows the
    global token-log mode.
    """
    explicit = os.environ.get("LLM_USAGE_LOG_ENABLED")
    if explicit is not None:
        return explicit.strip().lower() in _VERBOSE
    return global_token_console_verbose()

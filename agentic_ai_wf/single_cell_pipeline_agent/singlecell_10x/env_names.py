"""
env_names.py — this pipeline's environment variables, with a legacy alias each.

Why this exists
---------------
The pipeline read module-owned settings from bare, unprefixed names:
``SHARED_REFERENCE_ROOT``, ``RSCRIPT_EXE``, ``CONSENSUS_ENABLE_SINGLER``,
``CONSENSUS_ENABLE_PUBMED``, ``REPORT_LLM_MODEL``. Those are generic enough to collide
with another module on a shared host or in a container, and a collision here is
silent: the wrong reference root just resolves to a different marker table.

Every name is now prefixed ``SCPIPE_``. The old spelling is still honoured as a
fallback so existing ``.env`` files and server configs keep working, and taking the
legacy path logs once at WARNING so the migration is visible rather than permanent.

Third-party names are NOT renamed and are not listed here: ``OPENROUTER_API_KEY``,
``NCBI_API_KEY``, ``NCBI_EMAIL``, ``R_HOME`` and ``PATH`` belong to their own tools and
must keep the spelling those tools document.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

PREFIX = "SCPIPE_"

# Canonical name -> the legacy names still accepted, in precedence order.
LEGACY_ALIASES: dict[str, tuple[str, ...]] = {
    "SCPIPE_SHARED_REFERENCE_ROOT": (
        "SHARED_REFERENCE_ROOT",
        "AGENTIC_REFERENCE_DATA_ROOT",
    ),
    "SCPIPE_RSCRIPT_EXE": ("RSCRIPT_EXE",),
    "SCPIPE_ENABLE_SINGLER": ("CONSENSUS_ENABLE_SINGLER",),
    "SCPIPE_ENABLE_PUBMED": ("CONSENSUS_ENABLE_PUBMED",),
    "SCPIPE_REPORT_LLM_MODEL": ("REPORT_LLM_MODEL",),
}

SHARED_REFERENCE_ROOT = "SCPIPE_SHARED_REFERENCE_ROOT"
RSCRIPT_EXE = "SCPIPE_RSCRIPT_EXE"
ENABLE_SINGLER = "SCPIPE_ENABLE_SINGLER"
ENABLE_PUBMED = "SCPIPE_ENABLE_PUBMED"
REPORT_LLM_MODEL = "SCPIPE_REPORT_LLM_MODEL"

# Legacy names already warned about, so a per-cluster call site does not emit the
# same migration notice hundreds of times in one run.
_WARNED: set[str] = set()


def get_env(name: str, default: str | None = None) -> str | None:
    """Read a pipeline environment variable, falling back to its legacy spelling.

    Args:
        name: Canonical ``SCPIPE_``-prefixed name from this module.
        default: Returned when neither the canonical nor any legacy name is set.

    Returns:
        The first non-empty value found, else `default`.
    """
    value = os.environ.get(name, "").strip()
    if value:
        return value

    for legacy in LEGACY_ALIASES.get(name, ()):
        value = os.environ.get(legacy, "").strip()
        if not value:
            continue
        if legacy not in _WARNED:
            _WARNED.add(legacy)
            logger.warning(
                "Environment variable %s is deprecated; rename it to %s. The old name "
                "still works for now but is unprefixed and can collide with other "
                "modules on the same host.",
                legacy,
                name,
            )
        return value

    return default


def reset_deprecation_warnings_for_tests() -> None:
    """Re-arm the once-per-process deprecation warnings. Test helper only."""
    _WARNED.clear()


__all__ = [
    "PREFIX",
    "LEGACY_ALIASES",
    "SHARED_REFERENCE_ROOT",
    "RSCRIPT_EXE",
    "ENABLE_SINGLER",
    "ENABLE_PUBMED",
    "REPORT_LLM_MODEL",
    "get_env",
    "reset_deprecation_warnings_for_tests",
]

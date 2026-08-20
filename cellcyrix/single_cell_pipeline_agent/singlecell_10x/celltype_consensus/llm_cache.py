"""
llm_cache.py — content-addressed cache for LLM responses, so a run is reproducible.

Why this exists
---------------
The knowledge-based voter and the PubMed adjudicator call a hosted model over HTTP.
``temperature`` is already 0.0, which is necessary for reproducibility but nowhere near
sufficient: a hosted endpoint can change the weights behind a floating model alias, load
balance across non-identical replicas, or simply return different text for the same
request on a different day. A run whose cell-type labels depend on that is not
reproducible, and "we set temperature to zero" is not an answer a reviewer should accept.

This makes the dependency auditable in the only way that actually holds: the first time
a given request is made its response is written to disk, keyed by a hash of everything
that could change the answer — model id, system prompt, user prompt, temperature,
top_p, seed and max_tokens. Every later run with the same inputs reads the response back
instead of calling out. So a published result can be regenerated exactly, offline, from
the cache directory shipped beside it, and a cache that is present but missing an entry
is a loud signal that an input changed rather than a silent re-query.

The key deliberately covers the prompt text. Editing a prompt invalidates every entry
that used it, which is correct — the old response is no longer an answer to the question
being asked.

Layout::

    <cache_dir>/<first two hex chars>/<full key>.json

Fanning out on the first byte keeps any single directory small enough for a filesystem
to list quickly when a cohort produces thousands of entries.

Set ``SCPIPE_LLM_CACHE=0`` to disable, or ``SCPIPE_LLM_CACHE_DIR`` to relocate it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

#: Env var that turns the cache off entirely (``0``/``false``/``no``).
ENV_ENABLED = "SCPIPE_LLM_CACHE"

#: Env var that relocates the cache directory.
ENV_CACHE_DIR = "SCPIPE_LLM_CACHE_DIR"

#: Directory name used under the package root when ``SCPIPE_LLM_CACHE_DIR`` is unset.
DEFAULT_CACHE_DIRNAME = ".llm_cache"

#: Bumped when the stored record shape changes, so old entries are ignored rather
#: than misread. Part of the key, so a bump is a clean cache miss and not a crash.
CACHE_SCHEMA_VERSION = 1

# Per-run counters, surfaced in the provenance manifest so a run states how much of
# its LLM output came from cache versus the network.
_STATS: dict[str, int] = {"hits": 0, "misses": 0, "writes": 0, "errors": 0}


def _falsey(value: str) -> bool:
    """True when an env string reads as an explicit "off"."""
    return value.strip().lower() in {"0", "false", "no", "off", ""}


def is_enabled() -> bool:
    """Whether caching is on. On by default; ``SCPIPE_LLM_CACHE=0`` turns it off."""
    raw = os.getenv(ENV_ENABLED)
    if raw is None:
        return True
    return not _falsey(raw)


def cache_dir() -> Path:
    """Directory holding the cache.

    Returns:
        ``SCPIPE_LLM_CACHE_DIR`` when set, else ``<package root>/.llm_cache``. The
        package-relative default keeps the cache with the code that produced it, which
        is what makes "ship the cache with the paper" a copy of one directory.
    """
    env = (os.getenv(ENV_CACHE_DIR) or "").strip()
    if env:
        return Path(env).expanduser().resolve()
    # <...>/cellcyrix/single_cell_pipeline_agent/singlecell_10x/celltype_consensus
    # -> up four to the directory that contains `cellcyrix`.
    package_root = Path(__file__).resolve().parents[4]
    return (package_root / DEFAULT_CACHE_DIRNAME).resolve()


def cache_key(
    *,
    model: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
    top_p: Optional[float] = None,
    seed: Optional[int] = None,
) -> str:
    """Hash every input that can change the response.

    Args:
        model: Model identifier as sent to the endpoint.
        system: System prompt.
        user: User prompt.
        temperature: Sampling temperature.
        max_tokens: Reply-length bound.
        top_p: Nucleus-sampling parameter, when sent.
        seed: Sampling seed, when sent.

    Returns:
        Hex SHA-256 over a canonical JSON encoding of the request.
    """
    payload = {
        "schema": CACHE_SCHEMA_VERSION,
        "model": model,
        "system": system,
        "user": user,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "top_p": top_p,
        "seed": seed,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _entry_path(key: str, directory: Optional[Path] = None) -> Path:
    """On-disk path for a key, sharded on its first two hex characters."""
    base = directory or cache_dir()
    return base / key[:2] / f"{key}.json"


def get(key: str, *, directory: Optional[Path] = None) -> Optional[str]:
    """Look up a cached response.

    A malformed or unreadable entry counts as a miss rather than an error: the caller
    can always re-query, and refusing to run because a cache file got truncated would
    trade a cheap network call for a dead pipeline.

    Args:
        key: Value from :func:`cache_key`.
        directory: Override the cache directory (tests).

    Returns:
        The cached response text, or None on a miss.
    """
    if not is_enabled():
        return None
    path = _entry_path(key, directory)
    if not path.is_file():
        _STATS["misses"] += 1
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        response = record["response"]
    except (OSError, ValueError, KeyError) as exc:
        logger.warning("[LLM-CACHE] ignoring unreadable entry %s: %s", path.name, exc)
        _STATS["errors"] += 1
        _STATS["misses"] += 1
        return None
    _STATS["hits"] += 1
    logger.debug("[LLM-CACHE] hit %s", key[:12])
    return response


def put(
    key: str,
    response: str,
    *,
    directory: Optional[Path] = None,
    meta: Optional[dict[str, Any]] = None,
) -> Optional[Path]:
    """Store a response.

    Written to a temp file and renamed, so a crash mid-write cannot leave a truncated
    entry that later parses as a complete one — the same discipline the pipeline's
    table writers use.

    Args:
        key: Value from :func:`cache_key`.
        response: Raw response text to store.
        directory: Override the cache directory (tests).
        meta: Extra fields recorded alongside (model, prompt lengths, and so on).
            Never include credentials.

    Returns:
        The entry path, or None when caching is disabled or the write failed.
    """
    if not is_enabled():
        return None
    path = _entry_path(key, directory)
    record = {
        "key": key,
        "schema": CACHE_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "response": response,
        **(meta or {}),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".part")
        tmp.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
        tmp.replace(path)
    except OSError as exc:
        # A cache that cannot be written must not fail the run.
        logger.warning("[LLM-CACHE] could not write %s: %s", path.name, exc)
        _STATS["errors"] += 1
        return None
    _STATS["writes"] += 1
    logger.debug("[LLM-CACHE] stored %s", key[:12])
    return path


def stats() -> dict[str, Any]:
    """Cache counters for this run, for the provenance manifest.

    Returns:
        Hits, misses, writes, errors, whether caching was enabled, the directory, and
        ``fully_cached`` — True when every lookup hit, meaning the run made no LLM
        network call and is reproducible from the cache alone.
    """
    total = _STATS["hits"] + _STATS["misses"]
    return {
        "enabled": is_enabled(),
        "directory": str(cache_dir()) if is_enabled() else None,
        **dict(_STATS),
        "lookups": total,
        "fully_cached": total > 0 and _STATS["misses"] == 0,
    }


def reset_stats() -> None:
    """Zero the counters. Call between runs in the same process."""
    for name in _STATS:
        _STATS[name] = 0


__all__ = [
    "ENV_ENABLED",
    "ENV_CACHE_DIR",
    "CACHE_SCHEMA_VERSION",
    "DEFAULT_CACHE_DIRNAME",
    "is_enabled",
    "cache_dir",
    "cache_key",
    "get",
    "put",
    "stats",
    "reset_stats",
]

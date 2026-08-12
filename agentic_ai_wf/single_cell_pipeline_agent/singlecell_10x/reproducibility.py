"""
reproducibility.py — seed discipline, provenance manifest, environment capture.

Three small, dependency-light helpers used by the pipeline so every run is
reproducible and self-documenting:

  * set_global_seed(seed)  — seed Python / NumPy (and PYTHONHASHSEED). Combined
    with random_state=SEED threaded through PCA/neighbors/UMAP/tSNE/Leiden in the
    pipeline, this makes a run deterministic.
  * write_run_manifest(...) — write provenance/manifest.json: seed, params,
    dataset shape, timestamp, git commit, and captured package versions.
"""

from __future__ import annotations

import importlib.metadata
import json
import logging
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Global default seed. Threaded through every stochastic scanpy step.
DEFAULT_SEED = 0

# Packages whose versions are worth pinning to a run for reproducibility.
_VERSION_PKGS = [
    "scanpy",
    "anndata",
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "leidenalg",
    "igraph",
    "bbknn",
    "harmonypy",
    "celltypist",
    "umap-learn",
    "pydeseq2",
    "scrublet",
    "matplotlib",
    "seaborn",
]


def set_global_seed(seed: int = DEFAULT_SEED) -> int:
    """Seed Python `random`, NumPy, and PYTHONHASHSEED. Returns the seed used."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:  # scanpy reads this in a few places; harmless if absent
        import scanpy as sc

        sc.settings.verbosity = sc.settings.verbosity
        if hasattr(sc.settings, "seed"):
            sc.settings.seed = seed
    except (ImportError, AttributeError) as exc:
        logger.debug("scanpy seed settings unavailable: %s", exc)
    return seed


def capture_versions() -> dict[str, str]:
    """Return {package: version} for the analysis-relevant packages + python."""
    out: dict[str, str] = {"python": sys.version.split()[0]}
    for p in _VERSION_PKGS:
        try:
            out[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            # Not installed in this environment; simply absent from the manifest.
            pass
    return out


def _git_commit() -> Optional[str]:
    """Best-effort current git commit of the working tree (None if unavailable)."""
    try:
        r = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(Path(__file__).resolve().parent),
            check=False,  # not a git checkout is a supported outcome -> None
        )
        return r.stdout.strip() or None
    except Exception as exc:  # noqa: BLE001 - best-effort boundary; falls back below
        logger.debug("%s: falling back after %r", __name__, exc)
        return None


def write_run_manifest(
    out_dir: str | Path,
    *,
    analysis_name: str,
    seed: int,
    params: dict[str, Any],
    n_obs: int,
    n_vars: int,
    extra: Optional[dict[str, Any]] = None,
) -> Path:
    """Write provenance/manifest.json capturing everything needed to reproduce a run."""
    out_dir = Path(out_dir)
    prov = out_dir / "provenance"
    prov.mkdir(parents=True, exist_ok=True)

    manifest = {
        "analysis_name": analysis_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "seed": seed,
        "dataset": {"n_obs": int(n_obs), "n_vars": int(n_vars)},
        "params": params,
        "git_commit": _git_commit(),
        "package_versions": capture_versions(),
    }
    if extra:
        manifest.update(extra)

    path = prov / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return path

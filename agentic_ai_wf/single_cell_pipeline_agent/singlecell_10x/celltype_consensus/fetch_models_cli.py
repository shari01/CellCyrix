"""
fetch_models_cli.py — pre-warm the whole CellTypist model bundle.

Why this exists
---------------
The 41 CellTypist models are 54 MB of binary pickles, one of them over the 5 MB
per-file ceiling, so they are fetched rather than committed. The pipeline already
fetches whichever model a run selects on demand (``model_integrity.ensure_model_file``),
which makes this command OPTIONAL — use it to get all 41 in place at once before going
offline, when baking an image, or to re-verify what is already on disk.

It lives inside the package (rather than only as ``scripts/fetch_celltypist_models.py``)
so it ships with a ``pip install`` and is reachable as the ``fetch-celltypist-models``
console script. Someone who installed the wheel instead of cloning the repo can still
run what the README documents.

All download and verification logic is in ``model_integrity``; this module is only the
command-line front end, so the on-demand path and this one cannot diverge.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .lineage_panels import shared_reference_root
from .model_integrity import (
    MANIFEST_NAME,
    ModelIntegrityError,
    fetch_model,
    load_manifest,
    model_urls,
    verify_model_file,
)
from .tools import _BUNDLED_CELLTYPIST_DIRNAME

logger = logging.getLogger(__name__)


def default_models_dir() -> Path:
    """The bundled model directory, resolved the same way the pipeline resolves it.

    Returns:
        ``<shared reference root>/celltypist_models/data/models``.
    """
    return shared_reference_root() / _BUNDLED_CELLTYPIST_DIRNAME / "data" / "models"


def main(argv: list[str] | None = None) -> int:
    """Fetch every model listed in the manifest that is missing or unverified.

    Args:
        argv: Command-line arguments; defaults to `sys.argv[1:]`.

    Returns:
        0 when every manifest entry is present and verified, 1 otherwise.
    """
    parser = argparse.ArgumentParser(
        description="Fetch and verify the bundled CellTypist models."
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="Destination directory (default: the bundled model directory).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even models that are already present and verified.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
        stream=sys.stdout,
    )

    models_dir: Path = args.models_dir or default_models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)

    expected = load_manifest(models_dir)
    if not expected:
        logger.error(
            "no %s in %s; it is tracked in git and lists the expected digests",
            MANIFEST_NAME,
            models_dir,
        )
        return 1

    urls = model_urls(models_dir)
    skipped = fetched = failed = 0

    for name in sorted(expected):
        dest = models_dir / name
        if dest.is_file() and not args.force:
            try:
                verify_model_file(dest, required=True)
                skipped += 1
                continue
            except ModelIntegrityError as exc:
                logger.warning("%s present but not valid (%s); re-fetching", name, exc)
        url = urls.get(name)
        if not url:
            logger.error("%s has no URL in models.json; cannot fetch", name)
            failed += 1
            continue
        if fetch_model(name, url, models_dir):
            fetched += 1
        else:
            failed += 1

    logger.info(
        "done: %d already verified, %d fetched, %d failed (%d in manifest)",
        skipped,
        fetched,
        failed,
        len(expected),
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

"""
atomic_io.py — write a file completely or not at all.

Why this exists
---------------
Every table in this pipeline was written straight to its final path. A run that dies
mid-write — killed worker, full disk, network share hiccup — therefore left a
TRUNCATED file sitting at the name downstream code reads. That failure mode is
particularly bad here because the files are DE tables and marker tables: a CSV cut
off after 400 of 20,000 genes is still valid CSV. It parses, it has the right header,
and it reads as "these are the genes that came out" rather than as an error.

The fix is the standard temp-then-rename: write to ``<name>.tmp`` in the SAME
directory (so the rename stays on one filesystem and is therefore atomic), then
``Path.replace`` it into position. A reader sees either the previous file or the
complete new one, never a half-written one. On failure the partial temp file is
removed and the exception propagates, so the run fails loudly instead of leaving
plausible-looking output behind.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import pandas as pd

# rank_genes_subprocess.py imports this module BY DIRECTORY (it is executed as a
# standalone script via subprocess.run on its path), and in that mode a
# package-relative import raises "attempted relative import with no known parent
# package". Both spellings are therefore supported. column_names is leaf-level
# (pandas only), so importing it by directory is safe.
try:
    from .column_names import to_canonical_columns
except ImportError:  # pragma: no cover - the standalone-script path
    from column_names import to_canonical_columns

logger = logging.getLogger(__name__)


def atomic_write(path: str | Path, write: Callable[[Path], None]) -> Path:
    """Run `write` against a temp path, then atomically move it into place.

    Args:
        path: Final destination. Its parent directory is created if missing.
        write: Callback that writes the complete file to the path it is given.

    Returns:
        The final path.

    Raises:
        Exception: Whatever `write` raises, after the partial temp file is removed.
    """
    final = Path(path)
    final.parent.mkdir(parents=True, exist_ok=True)
    # Same directory as the destination: Path.replace is only atomic within one
    # filesystem, so a temp file in the system temp dir would not be safe here.
    tmp = final.with_name(final.name + ".tmp")
    try:
        write(tmp)
    except Exception:
        # Leaving a stale .tmp behind would be mistaken for a real output next run.
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(final)
    return final


def atomic_to_csv(
    frame: pd.DataFrame | pd.Series,
    path: str | Path,
    *,
    index: bool = False,
    **to_csv_kwargs: object,
) -> Path:
    """Write a DataFrame/Series to CSV atomically.

    Args:
        frame: The table to write.
        path: Final destination path.
        index: Passed to `to_csv`. Defaults to False per the output standard; pass
            True only when the index is real data.
        **to_csv_kwargs: Any other `DataFrame.to_csv` keyword.

    Returns:
        The final path.
    """
    return atomic_write(
        path, lambda tmp: frame.to_csv(tmp, index=index, **to_csv_kwargs)
    )


def atomic_write_text(path: str | Path, text: str, *, encoding: str = "utf-8") -> Path:
    """Write text to `path` atomically as UTF-8.

    Args:
        path: Final destination path.
        text: Full file contents.
        encoding: Text encoding. Defaults to UTF-8 per the I/O standard.

    Returns:
        The final path.
    """
    return atomic_write(path, lambda tmp: tmp.write_text(text, encoding=encoding))


def read_table(path: str | Path, **read_csv_kwargs: object) -> pd.DataFrame:
    """Read a table written by this pipeline, normalising its headers.

    The counterpart to `write_table`: headers are mapped through
    `column_names.to_canonical_columns` on the way IN as well as out, so a reader only
    ever has to know the canonical names. That also makes tables written before the
    Rule 5.4 rename readable unchanged — an old file's ``padj`` arrives as
    ``p_value_adj`` exactly like a new file's.

    Args:
        path: CSV to read.
        **read_csv_kwargs: Any `pandas.read_csv` keyword.

    Returns:
        The table with canonical headers.
    """
    return to_canonical_columns(pd.read_csv(path, **read_csv_kwargs))


def write_table(
    frame: pd.DataFrame,
    path: str | Path,
    *,
    index: bool = False,
    **to_csv_kwargs: object,
) -> Path:
    """Write a DATA table atomically, with canonical lower_snake_case headers.

    Use this for tables another person or tool will read — DE results, marker tables,
    pathway enrichment, summaries. It applies `column_names.to_canonical_columns` so
    every emitted file names the same quantity the same way, regardless of which
    library produced the frame. Values are untouched; only headers are renamed.

    Use `atomic_to_csv` instead for internal bookkeeping tables whose headers are
    already the pipeline's own vocabulary (audit rows, cluster counts, manifests) —
    the rename would be a no-op there anyway, but the distinction documents intent.

    Args:
        frame: The table to write.
        path: Final destination path.
        index: Passed to `to_csv`. Defaults to False per the output standard.
        **to_csv_kwargs: Any other `DataFrame.to_csv` keyword.

    Returns:
        The final path.
    """
    return atomic_to_csv(
        to_canonical_columns(frame), path, index=index, **to_csv_kwargs
    )


__all__ = [
    "atomic_write",
    "atomic_to_csv",
    "atomic_write_text",
    "read_table",
    "write_table",
]

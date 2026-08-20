"""
output_paths.py — resolve a run's output directory deterministically.

Why this exists
---------------
Both drivers used to do ``combined_out_dir = Path(out_name)`` with ``out_name``
defaulting to a bare ``"SC_RESULTS"``. A bare relative name is resolved by the OS
against the *current working directory*, so where a run's results landed depended
on where the process happened to be launched from — the same config produced
``./SC_RESULTS`` under the repo root, under a service's working directory, or under
whatever directory a scheduler chose. Nothing logged the difference, because from
the pipeline's point of view the write succeeded.

Resolution is therefore centralised here, and an ambiguous request fails loudly
rather than guessing:

* absolute ``out_name``            -> used as-is
* relative ``out_name`` + root     -> ``root / out_name``
* relative ``out_name``, no root   -> ``OutputPathError``

Callers that genuinely want "next to the input data" pass that directory as the
root explicitly, which keeps the decision in the caller and out of the CWD.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .exceptions import PipelineInputError

logger = logging.getLogger(__name__)


class OutputPathError(PipelineInputError):
    """The requested output location cannot be resolved without guessing."""


def resolve_output_dir(
    out_name: str | Path,
    output_root: str | Path | None,
    *,
    create: bool = True,
) -> Path:
    """Resolve a run's output directory to an absolute path.

    Args:
        out_name: Output folder for this run. May be absolute, or relative to
            `output_root`.
        output_root: Directory a relative `out_name` is resolved against. May be
            None only when `out_name` is absolute.
        create: Create the directory (and parents) before returning it.

    Returns:
        The absolute output directory.

    Raises:
        OutputPathError: If `out_name` is relative and `output_root` is None, or if
            the resolved path escapes `output_root`.
    """
    out_path = Path(out_name)

    if out_path.is_absolute():
        resolved = out_path.resolve()
    else:
        if output_root is None:
            raise OutputPathError(
                f"out_name={str(out_name)!r} is a relative path and no output_root "
                "was given, so the results would be written relative to the current "
                "working directory. Pass output_root=<directory> (the top-level "
                "main.py passes its `outputs/` folder), or make out_name absolute."
            )
        root = Path(output_root).resolve()
        resolved = (root / out_path).resolve()
        # A configured out_name of "../../elsewhere" would silently write outside the
        # directory the caller nominated; refuse instead of honouring the traversal.
        if not resolved.is_relative_to(root):
            raise OutputPathError(
                f"out_name={str(out_name)!r} resolves to {resolved}, which is outside "
                f"output_root={root}. Use a name inside the output root, or pass an "
                "absolute out_name if writing elsewhere is intended."
            )

    if create:
        resolved.mkdir(parents=True, exist_ok=True)
    logger.debug("Resolved output directory: %s", resolved)
    return resolved

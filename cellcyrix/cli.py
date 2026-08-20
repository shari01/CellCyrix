"""
cli.py — the ``cellcyrix`` console script.

An argument parser over :func:`cellcyrix.runner.run_from_config`, and nothing else. No
analysis logic lives here: the flags map one-to-one onto the two parameters that
function already takes, so the CLI and the repository's root ``main.py`` cannot drift
apart or disagree about what a config means.

Why this exists at all, given the project shipped without it for a long time: the
position was that the pipeline is a library plus one official script, the root
``main.py``. That is coherent for a repository its own authors run, but ``pip install``
places the package on the path without placing ``main.py`` anywhere, so an installed
copy had no way to start a run. Anyone reproducing the work from a released artifact
rather than a clone — a reviewer, a container build, a fresh environment — needs one
command that works, which is what this provides::

    cellcyrix --config config.yaml
    cellcyrix --config my_config.yaml --output-root /data/runs
    cellcyrix --config config.yaml --print-config    # resolve and exit, run nothing

Unlike ``main.py``, the defaults here are resolved against the current working
directory, because an installed package has no repository root to be relative to.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

from cellcyrix.runner import flatten_sections, load_config, run_from_config

logger = logging.getLogger(__name__)

#: Config filename looked for in the working directory when ``--config`` is omitted.
DEFAULT_CONFIG_NAME = "config.yaml"

#: Output directory used when ``--output-root`` is omitted.
DEFAULT_OUTPUT_DIRNAME = "outputs"


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser.

    Returns:
        The parser. Exposed separately so tests can exercise parsing without running.
    """
    parser = argparse.ArgumentParser(
        prog="cellcyrix",
        description=(
            "Run the CellCyrix single-cell pipeline from a YAML config: QC, doublet "
            "detection, HVG selection, PCA/UMAP, Leiden clustering, multi-voter "
            "consensus cell-type annotation, donor-level pseudobulk differential "
            "expression, pathway enrichment, and an HTML/PDF report."
        ),
        epilog=(
            "The config drives everything, including whether the run is single-sample "
            "or a cohort (mode: single | multi). See docs/"
            "PIPELINE_MASTER_DOCUMENTATION.md for the full config reference."
        ),
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=None,
        help=(
            f"YAML config driving the run (default: ./{DEFAULT_CONFIG_NAME} in the "
            "current directory)."
        ),
    )
    parser.add_argument(
        "-o",
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Directory that the config's out_name is resolved against, so outputs "
            f"never land wherever the process happens to be (default: ./"
            f"{DEFAULT_OUTPUT_DIRNAME})."
        ),
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help=(
            "Resolve the config — including nested sections flattened onto driver "
            "arguments — print it as JSON, and exit without running. Use this to "
            "check what a config actually resolves to before spending the compute."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Log at DEBUG instead of INFO.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_version()}",
    )
    return parser


def _version() -> str:
    """Installed package version, or a marker when running from a source tree."""
    try:
        from importlib.metadata import PackageNotFoundError, version

        return version("cellcyrix")
    except PackageNotFoundError:
        return "0.0.0+source"
    except ImportError:  # pragma: no cover - importlib.metadata is stdlib on 3.11+
        return "unknown"


def _configure_console_logging(verbose: bool) -> None:
    """Send this CLI's own messages to stdout without claiming the root logger.

    Deliberately not ``logging.basicConfig()``. The drivers call ``basicConfig``
    themselves to install the run's FileHandler for ``<output_dir>/pipeline.log``, and
    ``basicConfig`` is a no-op once root has any handler — so an entry point that
    configures root first silently costs the run its log file, leaving pipeline.log at
    0 bytes while the run looks healthy. This mirrors ``main.py``'s handling for the
    same reason.

    Args:
        verbose: Log at DEBUG rather than INFO.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    for target in (logger, logging.getLogger("cellcyrix.runner")):
        target.addHandler(handler)
        target.setLevel(logging.DEBUG if verbose else logging.INFO)
        target.propagate = False


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for the ``cellcyrix`` console script.

    Args:
        argv: Argument list. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit status: 0 on success, 1 on a config or input error, 130 if
        interrupted.
    """
    args = build_parser().parse_args(argv)
    _configure_console_logging(args.verbose)

    config_path = (args.config or Path.cwd() / DEFAULT_CONFIG_NAME).expanduser()
    output_root = (args.output_root or Path.cwd() / DEFAULT_OUTPUT_DIRNAME).expanduser()

    try:
        if args.print_config:
            config = load_config(config_path)
            resolved = {
                "config_path": str(config_path.resolve()),
                "output_root": str(output_root.resolve()),
                "mode": config.get("mode"),
                "flattened_sections": flatten_sections(config),
                "config": config,
            }
            # The point of --print-config is a machine-readable dump on stdout, so
            # this is the one place a print is correct rather than a log record.
            print(json.dumps(resolved, indent=2, default=str))  # noqa: T201
            return 0

        run_from_config(config_path, output_root)
        return 0

    except KeyboardInterrupt:
        logger.warning("Interrupted.")
        return 130
    except (FileNotFoundError, ValueError) as exc:
        # A missing config or a malformed mode: is user error, not a crash. Report it
        # as a message and a status code rather than a traceback.
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":  # pragma: no cover - exercised via the console script
    raise SystemExit(main())

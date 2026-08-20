"""Official entry point for the single-cell pipeline, for a clone of this repository.

Reads a YAML config (``config.yaml`` by default) and dispatches to the single-sample or
cohort driver::

    python main.py

The config loading, section flattening and driver dispatch all live in
:mod:`cellcyrix.runner`; this module is the repository-root front end over it and holds
only the repo-relative defaults, because they are the one thing the packaged code cannot
know. The installed console script ``cellcyrix`` (see :mod:`cellcyrix.cli`) is the other
front end over the same function, so the two cannot disagree about what a config means.

To run a different configuration without editing this file::

    import main
    main.main(config_path=Path("my_config.yaml"))
    main.main(config_path=Path("my_config.yaml"), output_root=Path("/data/runs"))

Outputs land under the directory named by ``out_name`` in the config, resolved against
``OUTPUT_DIR`` so a run never writes into the current working directory.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Re-exported so `main.load_config`, `main.flatten_sections` and `main._build_kwargs`
# keep resolving for callers and tests that reach for them through this module.
from cellcyrix.runner import (  # noqa: F401
    _SECTION_KEY_MAP,
    _build_kwargs,
    flatten_sections,
    load_config,
    run_from_config,
)

ROOT = Path(__file__).resolve().parent
logger = logging.getLogger(__name__)

# Root that every run's output directory is resolved against. Passed to the runner
# so `out_name` can never be interpreted relative to the process CWD.
OUTPUT_DIR = ROOT / "outputs"

# Default YAML that drives the pipeline. Override by passing `config_path` to
# `main()`; the constant is only the default.
CONFIG_FILE = ROOT / "config.yaml"


def main(config_path: Path = CONFIG_FILE, output_root: Path = OUTPUT_DIR) -> None:
    """Run the pipeline described by a YAML config.

    Args:
        config_path: YAML config to drive the run. Defaults to `CONFIG_FILE`.
        output_root: Directory that the config's `out_name` is resolved against, so
            outputs never land in the process working directory. Defaults to
            `OUTPUT_DIR`.

    Raises:
        ValueError: If `mode:` is missing/unknown, or the mode's required input
            directory key is absent.
    """
    run_from_config(config_path, output_root)


def _configure_console_logging() -> None:
    """Route THIS module's messages to stdout without claiming the root logger.

    Deliberately not ``logging.basicConfig()``. ``run_pipeline`` /
    ``run_pipeline_multi`` call ``basicConfig`` themselves (main_single.py:225,
    main_multi.py:330) to install the run's FileHandler for
    ``<output_dir>/pipeline.log`` alongside a stdout StreamHandler — and
    ``basicConfig`` is a NO-OP once the root logger has any handler. So an entry
    point that configures root first silently costs the run its log FILE: the
    FileHandler is constructed (creating the file) and then discarded, leaving
    pipeline.log at 0 bytes while the run looks completely healthy.

    Measured, same pipeline log line: root untouched -> 57 bytes; root claimed
    here -> 0 bytes.

    Attaching to this module's own logger with ``propagate=False`` keeps root
    free for the pipeline, and reproduces exactly what the previous ``print``
    calls did — these messages go to stdout and nowhere else. The runner's logger
    is attached too, since that is where the run's own progress lines come from now.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    for target in (logger, logging.getLogger("cellcyrix.runner")):
        target.addHandler(handler)
        target.setLevel(logging.INFO)
        target.propagate = False


if __name__ == "__main__":
    _configure_console_logging()
    main()

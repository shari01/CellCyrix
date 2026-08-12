#!/usr/bin/env python3
"""
Subprocess wrapper for entire single-cell pipeline (Celery fork-safe).

This module runs the complete single-cell pipeline in an isolated subprocess
to avoid Celery prefork deadlocks with matplotlib/R resources.

Usage:
    python -m agentic_ai_wf.single_cell_pipeline_agent.singlecell_10x.pipeline_subprocess \
        --single-10x-dir <input_dir> \
        --output-dir <output_dir> \
        [--sample-label <label>] \
        [--group-label <label>] \
        [--do-pathway-clustering] \
        [--do-groupwise-de] \
        [--do-dpt] \
        [--batch-key <key>] \
        [--integration-method <method>] \
        [--geo-json-path <path>] \
        [--logos-dir <dir>] \
        [--generate-report] \
        [--prepare-for-bisque]
"""

import os

os.environ["MPLBACKEND"] = "Agg"
import matplotlib

matplotlib.use("Agg")

import argparse
import json
import logging
import sys
from pathlib import Path

# Configure logging to ensure output is flushed immediately.
#
# The handler targets STDOUT deliberately. This module's status output used to be
# bare print() calls on stdout, and both parents capture that stream —
# single_cell_pipeline_agent/main.py streams it line by line (call site 2, with
# stderr folded in) or captures it wholesale (call site 1). Sending records to
# logging's default stderr instead would still be captured, but it would reorder
# this module's progress relative to the pipeline's own stdout writes.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
# Force unbuffered output + UTF-8 so status records don't crash on Windows cp1252.
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

from .main_single import run_pipeline

logger = logging.getLogger(__name__)


def main() -> None:
    """Parse CLI args, run the single-cell pipeline, and record a JSON status line.

    Exits 0 on success and 1 on failure; on both paths a machine-readable result
    dict (``status``/``ok``/``output_dir`` or ``error``) is written to
    ``--result-json`` so a parent process (e.g. Celery) can pick it up. The same
    dict is also logged, for a human reading the run output — but the FILE is the
    contract: both callers in single_cell_pipeline_agent/main.py read
    ``result_json_path`` and neither parses this process's stdout.
    """
    parser = argparse.ArgumentParser(
        description="Run single-cell pipeline in isolated subprocess (Celery fork-safe)"
    )
    parser.add_argument(
        "--single-10x-dir",
        required=True,
        help="Path to 10x Genomics data directory",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory path",
    )
    parser.add_argument(
        "--sample-label",
        default=None,
        help="Sample label (optional)",
    )
    parser.add_argument(
        "--group-label",
        default=None,
        help="Group label (optional)",
    )
    parser.add_argument(
        "--do-pathway-clustering",
        action="store_true",
        help="Enable pathway clustering",
    )
    parser.add_argument(
        "--do-groupwise-de",
        action="store_true",
        help="Enable group-wise differential expression",
    )
    parser.add_argument(
        "--do-dpt",
        action="store_true",
        help="Enable diffusion pseudotime computation",
    )
    parser.add_argument(
        "--batch-key",
        default=None,
        help="Batch key for integration (optional)",
    )
    parser.add_argument(
        "--integration-method",
        default=None,
        help="Integration method (e.g., 'bbknn')",
    )
    parser.add_argument(
        "--geo-json-path",
        default=None,
        help="Path to GEO JSON metadata file (optional)",
    )
    parser.add_argument(
        "--logos-dir",
        default=None,
        help="Directory containing logo files (optional)",
    )
    parser.add_argument(
        "--generate-report",
        action="store_true",
        help="Generate HTML/PDF report",
    )
    parser.add_argument(
        "--prepare-for-bisque",
        action="store_true",
        help="Prepare output for Bisque deconvolution",
    )
    parser.add_argument(
        "--result-json",
        default=None,
        help="Path to write result JSON (optional)",
    )

    args = parser.parse_args()

    try:
        logger.info("Starting single-cell pipeline subprocess...")
        logger.info("Input directory: %s", args.single_10x_dir)
        logger.info("Output directory: %s", args.output_dir)
        logger.info("Python executable: %s", sys.executable)
        logger.info("Working directory: %s", Path.cwd())

        # Resolve --output-dir here rather than letting run_pipeline interpret a
        # relative value: this process is spawned with an inherited working directory
        # the caller does not control, so "results/" must be pinned to an absolute
        # path before it reaches the driver.
        output_dir = Path(args.output_dir).resolve()

        # Run pipeline
        logger.info("Calling run_pipeline()...")
        result_path = run_pipeline(
            single_10x_dir=args.single_10x_dir,
            sample_label=args.sample_label,
            group_label=args.group_label,
            out_name=str(output_dir),
            do_pathway_clustering=args.do_pathway_clustering,
            do_groupwise_de=args.do_groupwise_de,
            do_dpt=args.do_dpt,
            batch_key=args.batch_key,
            integration_method=args.integration_method,
            geo_json_path=args.geo_json_path,
            logos_dir=args.logos_dir,
            generate_report=args.generate_report,
            prepare_for_bisque=args.prepare_for_bisque,
        )

        result = {
            "status": "completed",
            "output_dir": str(result_path),
            "ok": True,
        }

        # Write result JSON if requested
        if args.result_json:
            with open(args.result_json, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            logger.info("Result JSON written to: %s", args.result_json)

        logger.info("Pipeline completed successfully. Output: %s", result_path)
        logger.info("%s", json.dumps(result))
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(0)

    except Exception as e:
        error_result = {
            "status": "failed",
            "error": str(e),
            "ok": False,
        }

        # logger.exception attaches the traceback, replacing the previous
        # print + traceback.print_exc(file=sys.stderr) pair.
        logger.exception("Pipeline failed: %s", e)
        sys.stderr.flush()

        # Write error result JSON if requested
        if args.result_json:
            try:
                with open(args.result_json, "w", encoding="utf-8") as f:
                    json.dump(error_result, f, indent=2)
            except OSError:
                # The result FILE is the contract with the parent, so losing it means
                # the parent sees "subprocess died" with no reason. It must not mask
                # the original failure (we still exit 1 below), but it cannot be
                # swallowed silently either.
                logger.exception(
                    "Could not write error result JSON to %s; the parent will have no "
                    "machine-readable failure reason.",
                    args.result_json,
                )

        logger.error("%s", json.dumps(error_result))
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(1)


if __name__ == "__main__":
    main()

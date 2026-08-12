# config_cli.py
"""
CLI configuration parsing utilities.
This module only provides functions for parsing CLI arguments - it does not execute
any parsing on import to avoid interfering with programmatic usage.
"""

import argparse
import logging

# Default logger - gets root logger but doesn't configure it
# Logging will be configured by the caller (either CLI main() or run_pipeline())
logger = logging.getLogger()

# Default pathway clustering setting (can be overridden)
DO_PATHWAY_CLUSTERING = True


def parse_cli_args() -> argparse.Namespace:
    """
    Parse CLI arguments for the single-cell pipeline.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="10x-only single-cell pipeline (single dataset)."
    )
    parser.add_argument(
        "--single-10x-dir",
        type=str,
        required=True,
        help="Path to single 10x folder (required).",
    )
    parser.add_argument(
        "--single-sample-label",
        type=str,
        required=True,
        help="Sample label to store in obs['sample'] (required).",
    )
    parser.add_argument(
        "--single-group-label",
        type=str,
        default=None,
        help="Group label to store in obs['group'] (e.g. CASE/CONTROL). Optional.",
    )
    parser.add_argument(
        "--out-name",
        type=str,
        default="SC_RESULTS",
        help="Custom output dir name (default: SC_RESULTS).",
    )
    parser.add_argument(
        "--no-pathway-clustering",
        action="store_true",
        help="Disable pathway clustering/enrichment.",
    )

    return parser.parse_args()

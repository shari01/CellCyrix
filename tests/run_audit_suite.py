"""
run_audit_suite.py — run the non-invasive scientific audit test suite.

Package convention: the official entry point is the root ``main.py``; extra
audit/personal test scripts live here under ``tests/`` with NO CLI / argparse /
shell runner / ``python -m`` entry point. Just:

    python tests/run_audit_suite.py

It discovers and runs audit_tests.py (small synthetic AnnData with known biology,
positive + negative controls) and prints a pass/fail summary. Exit code is
non-zero if any audit assertion fails, so it can gate CI without a CLI.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import audit_tests  # noqa: E402


def main() -> None:
    suite = unittest.defaultTestLoader.loadTestsFromModule(audit_tests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()

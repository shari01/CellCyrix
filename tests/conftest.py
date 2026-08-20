"""Shared pytest setup for the suite.

Why this exists
---------------
``test_cell_hierarchy_resolver.py`` ships verbatim from the subtype-ref package and
imports ``cell_hierarchy`` as a top-level name. That package is not at the repo root —
it lives beside the module that consumes it, under
``cellcyrix/single_cell_pipeline_agent/singlecell_10x/celltype_consensus/``. The
upstream file's own ``sys.path.insert`` points one level too high, so the import only
resolved under ``tests/run_resolver_tests.py``, which patched the path itself.

Putting the path setup here makes plain ``pytest`` work, which is what the engineering
standard's gate runs. ``run_resolver_tests.py`` keeps working unchanged — it is still
the way to run those tests without pytest installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PKG_ROOT = _HERE.parent
_CELLTYPE_CONSENSUS = (
    _PKG_ROOT
    / "cellcyrix"
    / "single_cell_pipeline_agent"
    / "singlecell_10x"
    / "celltype_consensus"
)

# Package root first so ``cellcyrix.*`` resolves; then the directory that holds
# ``cell_hierarchy`` so the upstream test's top-level import resolves too.
for _path in (str(_PKG_ROOT), str(_CELLTYPE_CONSENSUS)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

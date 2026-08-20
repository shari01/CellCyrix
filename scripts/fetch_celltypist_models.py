"""Pre-warm the CellTypist model bundle, from a clone (no install required).

Running this is OPTIONAL: the pipeline fetches whichever model a run selects on first
use and verifies it against ``SHA256SUMS.txt``, so an offline run works from the second
run onwards with no manual step. Use this to get all 41 in place at once — before going
offline, or when baking an image:

    python scripts/fetch_celltypist_models.py            # fetch/verify all 41
    python scripts/fetch_celltypist_models.py --force    # re-download all 41

After ``pip install`` the same command is on PATH as ``fetch-celltypist-models``.

Everything real lives in the package
(``singlecell_10x/celltype_consensus/fetch_models_cli.py``, over
``celltype_consensus/model_integrity.py``) so this file and the pipeline's own
on-demand fetch cannot drift apart. It exists only so the command works from a clone
that has not been installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cellcyrix.single_cell_pipeline_agent.singlecell_10x.celltype_consensus.fetch_models_cli import (  # noqa: E402 - needs the sys.path line above
    main,
)

if __name__ == "__main__":
    sys.exit(main())

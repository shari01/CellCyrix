"""Scanpy 10x single-cell pipeline — public API.

Two entry points, one per input shape:

* `run_pipeline` — one 10x feature-barcode folder.
* `run_pipeline_multi` — a `<group>/<sample>/` cohort tree, combined into one
  AnnData so batch integration and donor-level pseudobulk DE are available.

Both take an `out_name` plus an `output_root` it is resolved against, and return the
absolute output directory. Everything else in this package is internal: the stage
modules (`qc_filters`, `hvg_selection`, `clustering`, `celltype_consensus`,
`pseudobulk_de`, ...) are orchestrated by `pipeline.run_scanpy_pipeline`, which both
entry points call.
"""

from .main_multi import run_pipeline_multi
from .main_single import run_pipeline

__all__ = ["run_pipeline", "run_pipeline_multi"]

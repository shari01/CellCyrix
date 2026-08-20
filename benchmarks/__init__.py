"""
Annotation benchmark: harmonisation, metrics, and the driver that produces the tables.

Separate from ``tests/benchmarks/``, which measures runtime and memory. This package
measures whether the annotation is *correct*, and is what the paper's numbers come from.

  * :mod:`benchmarks.harmonise` — map every label vocabulary onto one comparison level
    with a single function applied identically to truth and predictions.
  * :mod:`benchmarks.metrics` — macro-F1 with bootstrap CIs, risk-coverage curves,
    calibration/ECE, voter-disagreement entropy.
  * :mod:`benchmarks.run_annotation_benchmark` — the driver.

See ``benchmarks/README.md`` for the datasets to run against and the order to do it in.
"""

__all__ = ["harmonise", "metrics", "run_annotation_benchmark"]

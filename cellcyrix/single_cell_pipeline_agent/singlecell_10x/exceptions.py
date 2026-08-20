"""
exceptions.py — the one base class every error raised by this pipeline shares.

Why this exists
---------------
The pipeline defined ten error types across as many modules, each inheriting from a
different builtin: ``ConsensusConfigError(RuntimeError)``, ``QCConfigError(ValueError)``,
``PseudobulkInputError(ValueError)``, ``CellConservationError(RuntimeError)``, and so
on. Every one of them was individually catchable, but there was no way to express
"handle any failure that came from the pipeline" — a caller had to import and name all
ten, and would silently miss the eleventh when it was added.

``PipelineError`` is that missing base. The existing builtin bases are KEPT as a
second base on each subclass, so code that already catches ``ValueError`` for a bad
config value, or ``RuntimeError`` for a failed computation, keeps working unchanged —
this adds a way to catch, it does not take one away.

Two shared subclasses express the distinction that matters when reading a failure:

* ``PipelineInputError``     — the data or configuration handed in is unusable.
* ``PipelineComputationError`` — the analysis ran but produced an invalid result.
"""

from __future__ import annotations


class PipelineError(Exception):
    """Base class for every error raised by the single-cell pipeline."""


class PipelineInputError(PipelineError, ValueError):
    """Input data or configuration is missing, malformed, or inconsistent."""


class PipelineComputationError(PipelineError, RuntimeError):
    """The analysis ran but produced an invalid or implausible result."""


__all__ = ["PipelineError", "PipelineInputError", "PipelineComputationError"]

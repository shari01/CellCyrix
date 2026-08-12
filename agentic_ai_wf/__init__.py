"""Agentic analysis workflows.

A namespace package for the analysis agents. Each sub-package declares its own
public API — import from those rather than from here:

* `agentic_ai_wf.single_cell_pipeline_agent` — the single-cell 10x pipeline
* `agentic_ai_wf.llm` — shared LLM client / settings layer

Nothing is re-exported at this level on purpose: pulling a sub-package in here would
make importing any agent pay for every agent's dependency chain.
"""

__all__: list[str] = []
__version__ = "1.0.0"

"""Single-cell 10x pipeline agent — public API.

`run_pipeline` (one sample) and `run_pipeline_multi` (a cohort) are the pipeline
entry points. The remaining names are the agent-facing wrappers that run the
pipeline in an isolated subprocess; import them from here rather than reaching into
`main` or `singlecell_10x` internals.
"""

from agentic_ai_wf.single_cell_pipeline_agent.singlecell_10x import (
    run_pipeline,
    run_pipeline_multi,
)

__all__ = [
    "run_pipeline",
    "run_pipeline_multi",
    "single_cell_pipeline_direct",
    "run_single_cell_agent_with_args",
    "run_single_cell_agent_with_args_sync",
    "build_single_cell_runner_agent",
    "SingleCellPipelineArgs",
    "SingleCellPipelineToolResult",
]
__version__ = "1.0.0"


def __getattr__(name: str) -> object:
    """Resolve the agent-layer names lazily.

    `main` imports the `agents` SDK and the LLM settings at module scope, which is a
    heavy and network-configured dependency chain. Importing this package to reach
    only `run_pipeline` must not pay for it, so the agent names are resolved on first
    attribute access instead of at import time.

    Args:
        name: Attribute being looked up.

    Returns:
        The requested attribute from the agent layer.

    Raises:
        AttributeError: If `name` is not part of this package's public API.
    """
    if name in __all__:
        from agentic_ai_wf.single_cell_pipeline_agent import main as _main

        return getattr(_main, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

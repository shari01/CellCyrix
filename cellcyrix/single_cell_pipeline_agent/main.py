"""Agent-facing wrapper around the single-cell pipeline.

Exposes the pipeline three ways: `single_cell_pipeline_direct` (blocking call that
captures the subprocess output wholesale), `run_single_cell_pipeline_tool` (the
`function_tool` the agent invokes, which streams progress line by line), and
`build_single_cell_runner_agent` / `run_single_cell_agent_with_args` for the agent
loop itself. Every path runs the pipeline in an isolated subprocess via
`singlecell_10x.pipeline_subprocess` (Celery prefork safety) and reads the outcome
from that subprocess's `--result-json` file, never from its stdout.
"""

from __future__ import annotations

import ast
import asyncio
import json
import logging
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents import (
    Agent,
    FunctionToolResult,
    ModelSettings,
    RunContextWrapper,
    Runner,
    ToolsToFinalOutputResult,
    function_tool,
)
from pydantic import BaseModel, ConfigDict, Field

from cellcyrix.llm import settings as llm_settings
from cellcyrix.llm.agents_openrouter import openrouter_run_config
from cellcyrix.llm.env_names import OPENROUTER_API_KEY_ENV
from cellcyrix.llm.settings import primary_llm_api_key

logger = logging.getLogger(__name__)

# Wall-clock ceiling for one pipeline subprocess. A real cohort run is hours of
# compute, so this is a "the process is wedged" backstop, not a performance target.
PIPELINE_SUBPROCESS_TIMEOUT_S = 7200

# How long to wait for a streamed output line before checking whether the child is
# still alive; short enough that a dead child is noticed promptly, long enough not to
# spin the reader thread.
_STREAM_POLL_INTERVAL_S = 0.5

_ork = primary_llm_api_key()
if _ork:
    os.environ.setdefault(OPENROUTER_API_KEY_ENV, _ork)


#: Subprocess flags this layer forwards, keyed by the option name they come from.
#: Declared once because BOTH entry points below build the same command line — they
#: previously each carried an identical 11-branch `if`/`cmd.extend` block, so a new
#: flag had to be added in two places and adding it to only one silently changed
#: behaviour between the direct call and the agent tool.
#: `True` marks a store_true flag (emitted only when the value is truthy); `False`
#: marks a flag that takes a value.
_SUBPROCESS_FLAGS: tuple[tuple[str, str, bool], ...] = (
    ("sample_label", "--sample-label", False),
    ("group_label", "--group-label", False),
    ("do_pathway_clustering", "--do-pathway-clustering", True),
    ("do_groupwise_de", "--do-groupwise-de", True),
    ("do_dpt", "--do-dpt", True),
    ("batch_key", "--batch-key", False),
    ("integration_method", "--integration-method", False),
    ("geo_json_path", "--geo-json-path", False),
    ("logos_dir", "--logos-dir", False),
    ("generate_report", "--generate-report", True),
    ("prepare_for_bisque", "--prepare-for-bisque", True),
)

_SUBPROCESS_MODULE = (
    "cellcyrix.single_cell_pipeline_agent.singlecell_10x.pipeline_subprocess"
)


def _build_subprocess_cmd(
    *,
    single_10x_dir: str,
    output_dir: str,
    result_json_path: str,
    options: Dict[str, Any],
) -> List[str]:
    """Build the argv for one pipeline subprocess run.

    Args:
        single_10x_dir: 10x input directory.
        output_dir: Output directory for the run.
        result_json_path: File the subprocess writes its machine-readable result to.
        options: Option name -> value, filtered through `_SUBPROCESS_FLAGS`; anything
            not listed there is ignored because the subprocess CLI has no flag for it.

    Returns:
        The argument list for `subprocess.run` / `subprocess.Popen`. Never a shell
        string — the paths are user-supplied (standards §16.1).
    """
    cmd: List[str] = [
        sys.executable,
        "-u",  # Unbuffered stdout/stderr
        "-m",
        _SUBPROCESS_MODULE,
        "--single-10x-dir",
        str(single_10x_dir),
        "--output-dir",
        str(output_dir),
        "--result-json",
        result_json_path,
    ]
    for name, flag, is_store_true in _SUBPROCESS_FLAGS:
        value = options.get(name)
        if not value:
            continue
        if is_store_true:
            cmd.append(flag)
        else:
            cmd.extend([flag, str(value)])
    return cmd


def single_cell_pipeline_direct(
    single_10x_dir: str,
    output_dir: str,
    sample_label: Optional[str] = None,
    group_label: Optional[str] = None,
    do_pathway_clustering: bool = True,
    do_groupwise_de: bool = False,
    do_dpt: bool = False,
    batch_key: Optional[str] = None,
    integration_method: Optional[str] = None,
    geo_json_path: Optional[str] = None,
    logos_dir: Optional[str] = None,
    generate_report: bool = True,
    prepare_for_bisque: bool = True,
) -> Dict[str, Any]:
    """Direct wrapper for the single-cell pipeline (non-agent mode).

    For agent-based execution with self-correction, use
    ``run_single_cell_agent_with_args_sync()``.

    The parameter list mirrors ``run_single_cell_pipeline_tool`` on purpose: they are the
    same public surface, one for the model and one for a Python caller. See that
    function's docstring for why the 5-parameter soft limit (Rule 7.2) is waived here.

    NOTE: This function now uses subprocess execution for Celery compatibility.
    """
    # Use subprocess wrapper for Celery compatibility
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as tmp_result:
        result_json_path = tmp_result.name

    try:
        cmd = _build_subprocess_cmd(
            single_10x_dir=single_10x_dir,
            output_dir=output_dir,
            result_json_path=result_json_path,
            options={
                "sample_label": sample_label,
                "group_label": group_label,
                "do_pathway_clustering": do_pathway_clustering,
                "do_groupwise_de": do_groupwise_de,
                "do_dpt": do_dpt,
                "batch_key": batch_key,
                "integration_method": integration_method,
                "geo_json_path": geo_json_path,
                "logos_dir": logos_dir,
                "generate_report": generate_report,
                "prepare_for_bisque": prepare_for_bisque,
            },
        )

        subprocess.run(  # output is read from result_json_path, not stdout
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=PIPELINE_SUBPROCESS_TIMEOUT_S,
        )

        if Path(result_json_path).exists():
            with open(result_json_path, "r", encoding="utf-8") as f:
                subprocess_result = json.load(f)
            if subprocess_result.get("ok"):
                return subprocess_result.get(
                    "payload",
                    {
                        "output_dir": subprocess_result.get("output_dir"),
                        "status": "completed",
                    },
                )

        raise RuntimeError("Pipeline subprocess completed but result JSON invalid")

    except subprocess.CalledProcessError as e:
        error_msg = f"Pipeline failed (exit code {e.returncode})"
        if Path(result_json_path).exists():
            try:
                with open(result_json_path, "r", encoding="utf-8") as f:
                    error_result = json.load(f)
                    error_msg = error_result.get("error", error_msg)
            except (OSError, json.JSONDecodeError, AttributeError) as read_exc:
                # Keep the generic message; the subprocess failure is what matters.
                logger.debug(
                    "could not read error JSON %s: %s", result_json_path, read_exc
                )
        raise RuntimeError(error_msg) from e
    finally:
        if Path(result_json_path).exists():
            try:
                Path(result_json_path).unlink()
            except OSError as unlink_exc:  # cleanup is best-effort, never fatal
                logger.debug("could not remove %s: %s", result_json_path, unlink_exc)


class SingleCellPipelineArgs(BaseModel):
    """
    Arguments for single-cell 10x pipeline execution.
    """

    model_config = ConfigDict(extra="forbid")

    single_10x_dir: str = Field(
        ...,
        description="Path to 10x Genomics data folder (containing matrix.mtx, barcodes.tsv, features.tsv)",
    )
    output_dir: str = Field(
        ..., description="Output directory name (will be created as a folder)"
    )
    sample_label: Optional[str] = Field(
        default=None,
        description="Sample label to store in obs['sample']. If None, extracted from directory name",
    )
    group_label: Optional[str] = Field(
        default=None,
        description="Group label (e.g., 'CASE', 'CONTROL', 'TUMOR', 'NORMAL')",
    )
    do_pathway_clustering: bool = Field(
        default=True, description="Whether to run pathway enrichment analysis"
    )
    do_groupwise_de: bool = Field(
        default=False,
        description="Whether to run group-wise differential expression analysis",
    )
    do_dpt: bool = Field(
        default=False, description="Whether to compute diffusion pseudotime"
    )
    batch_key: Optional[str] = Field(
        default=None,
        description="Batch key for integration (if None, no batch correction)",
    )
    integration_method: Optional[str] = Field(
        default=None, description="Integration method ('bbknn' or None)"
    )
    geo_json_path: Optional[str] = Field(
        default=None,
        description="Path to GEO metadata JSON file. If None, auto-searches in single_10x_dir",
    )
    logos_dir: Optional[str] = Field(
        default=None,
        description="Directory containing logo files for report generation",
    )
    generate_report: bool = Field(
        default=True,
        description="Whether to generate HTML/PDF report after pipeline completion",
    )
    prepare_for_bisque: bool = Field(
        default=True,
        description="Whether to prepare output h5ad file for Bisque deconvolution",
    )


class SingleCellPipelineToolResult(BaseModel):
    """
    Structured result returned by the function tool.
    """

    model_config = ConfigDict(extra="forbid")

    ok: bool
    payload: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    args: Dict[str, Any]


@dataclass
class SingleCellRunnerContext:
    """
    Context for tracking retries and diagnostics.
    """

    max_attempts: int = 3
    attempts: int = 0
    last_args: Optional[Dict[str, Any]] = None
    errors: List[str] = field(default_factory=list)


@function_tool
def run_single_cell_pipeline_tool(
    single_10x_dir: str,
    output_dir: str,
    sample_label: Optional[str] = None,
    group_label: Optional[str] = None,
    do_pathway_clustering: bool = True,
    do_groupwise_de: bool = False,
    do_dpt: bool = False,
    batch_key: Optional[str] = None,
    integration_method: Optional[str] = None,
    geo_json_path: Optional[str] = None,
    logos_dir: Optional[str] = None,
    generate_report: bool = True,
    prepare_for_bisque: bool = True,
) -> SingleCellPipelineToolResult:
    """Safe wrapper around the single-cell pipeline; always returns a structured result.

    Validates inputs, handles errors gracefully, and provides detailed diagnostics.

    DELIBERATE EXCEPTION to the 5-parameter soft limit (Rule 7.2). ``@function_tool``
    turns this signature INTO the JSON schema the model sees, and each parameter's name
    and default is what the model reasons about when it decides how to call the tool.
    Collapsing them behind a single options object would present the model with one
    opaque parameter and no descriptions, which is a worse interface for the actual
    consumer. The flag list these feed — the subprocess command line — is shared via
    ``_SUBPROCESS_FLAGS``/``_build_subprocess_cmd``, so the duplication that
    mattered is gone.
    """
    args: Dict[str, Any] = {
        "single_10x_dir": single_10x_dir,
        "output_dir": output_dir,
        "sample_label": sample_label,
        "group_label": group_label,
        "do_pathway_clustering": do_pathway_clustering,
        "do_groupwise_de": do_groupwise_de,
        "do_dpt": do_dpt,
        "batch_key": batch_key,
        "integration_method": integration_method,
        "geo_json_path": geo_json_path,
        "logos_dir": logos_dir,
        "generate_report": generate_report,
        "prepare_for_bisque": prepare_for_bisque,
    }

    try:
        # Validate input directory exists
        single_10x_path = Path(single_10x_dir)
        if not single_10x_path.exists():
            raise FileNotFoundError(f"10x data directory not found: {single_10x_dir}")

        # Validate geo_json_path if provided
        if geo_json_path and not Path(geo_json_path).exists():
            raise FileNotFoundError(f"GEO JSON file not found: {geo_json_path}")

        # Validate logos_dir if provided
        if logos_dir and not Path(logos_dir).exists():
            raise FileNotFoundError(f"Logos directory not found: {logos_dir}")

        # Ensure output directory parent exists
        output_path = Path(output_dir)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # CRITICAL FOR CELERY: Run entire pipeline in isolated subprocess
        # This avoids deadlocks with matplotlib/R resources in Celery prefork mode
        logger.info(
            "Running single-cell pipeline in isolated subprocess for Celery compatibility..."
        )

        # Create temporary result JSON file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp_result:
            result_json_path = tmp_result.name

        try:
            # Same command builder as single_cell_pipeline_direct: this used to be a
            # second, hand-maintained copy of the identical flag list.
            cmd = _build_subprocess_cmd(
                single_10x_dir=single_10x_dir,
                output_dir=output_dir,
                result_json_path=result_json_path,
                options={
                    "sample_label": sample_label,
                    "group_label": group_label,
                    "do_pathway_clustering": do_pathway_clustering,
                    "do_groupwise_de": do_groupwise_de,
                    "do_dpt": do_dpt,
                    "batch_key": batch_key,
                    "integration_method": integration_method,
                    "geo_json_path": geo_json_path,
                    "logos_dir": logos_dir,
                    "generate_report": generate_report,
                    "prepare_for_bisque": prepare_for_bisque,
                },
            )

            logger.info("Executing subprocess: %s", " ".join(cmd))
            logger.info("Subprocess will write results to: %s", result_json_path)

            # Run subprocess with real-time output streaming
            # Use Popen instead of run to stream output in real-time
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Combine stderr into stdout
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True,
            )

            # Stream output in real-time
            logger.info("Subprocess started, streaming output...")
            output_lines = []

            # Use a queue to collect output lines
            output_queue = queue.Queue()

            def read_output() -> None:
                """Read output from subprocess in separate thread"""
                try:
                    for line in process.stdout:
                        line = line.rstrip()
                        if line:
                            output_queue.put(line)
                    output_queue.put(None)  # Signal end of output
                except Exception as e:
                    logger.debug(
                        "%s: falling back after %r", __name__, e, exc_info=True
                    )
                    output_queue.put(f"ERROR reading output: {e}")
                    output_queue.put(None)

            # Start output reading thread
            output_thread = threading.Thread(target=read_output, daemon=True)
            output_thread.start()

            # Stream output and wait for process
            start_time = time.time()
            timeout_seconds = PIPELINE_SUBPROCESS_TIMEOUT_S

            try:
                while True:
                    # Check timeout
                    if time.time() - start_time > timeout_seconds:
                        logger.error("❌ Subprocess timed out, killing process...")
                        process.kill()
                        process.wait()
                        raise RuntimeError(
                            f"Pipeline subprocess timed out after {timeout_seconds}s"
                        )

                    try:
                        line = output_queue.get(timeout=_STREAM_POLL_INTERVAL_S)
                        if line is None:
                            break  # End of output
                        output_lines.append(line)
                        logger.info("[PIPELINE] %s", line)
                    except queue.Empty:
                        # Check if process is still running
                        if process.poll() is not None:
                            # Process finished, read remaining output
                            while True:
                                try:
                                    line = output_queue.get_nowait()
                                    if line is None:
                                        break
                                    output_lines.append(line)
                                    logger.info("[PIPELINE] %s", line)
                                except queue.Empty:
                                    break
                            break
                        # Process still running, continue waiting
                        continue

                # Wait for process to complete
                return_code = process.wait()

                if return_code != 0:
                    error_output = "\n".join(output_lines[-50:])  # Last 50 lines
                    raise subprocess.CalledProcessError(
                        return_code,
                        cmd,
                        output=error_output,
                    )

                logger.info("✅ Subprocess completed successfully")

            except RuntimeError:
                # Timeout error, already handled
                raise
            except Exception as e:
                logger.exception("❌ Error during subprocess execution: %s", e)
                if process.poll() is None:
                    process.kill()
                    process.wait()
                raise

            # Load result JSON
            if Path(result_json_path).exists():
                with open(result_json_path, "r", encoding="utf-8") as f:
                    subprocess_result = json.load(f)

                if (
                    subprocess_result.get("ok")
                    and subprocess_result.get("status") == "completed"
                ):
                    result_path = Path(subprocess_result["output_dir"])
                    logger.info(
                        "✅ Pipeline subprocess completed successfully. Output: %s",
                        result_path,
                    )

                    return SingleCellPipelineToolResult(
                        ok=True,
                        payload={"output_dir": str(result_path), "status": "completed"},
                        error=None,
                        args=args,
                    )
                else:
                    error_msg = subprocess_result.get("error", "Unknown error")
                    raise RuntimeError(f"Pipeline subprocess failed: {error_msg}")
            else:
                raise FileNotFoundError(f"Result JSON not found: {result_json_path}")

        except subprocess.TimeoutExpired as e:
            logger.error(
                "❌ Pipeline subprocess timed out after %ss",
                PIPELINE_SUBPROCESS_TIMEOUT_S,
            )
            raise RuntimeError("Pipeline execution timed out") from e

        except subprocess.CalledProcessError as e:
            logger.exception(
                "❌ Pipeline subprocess failed with exit code %s", e.returncode
            )
            if e.stdout:
                logger.exception("Subprocess stdout:\n%s", e.stdout)
            if e.stderr:
                logger.exception("Subprocess stderr:\n%s", e.stderr)

            # Try to load error result JSON
            error_msg = f"Pipeline execution failed (exit code {e.returncode})"
            if Path(result_json_path).exists():
                try:
                    with open(result_json_path, "r", encoding="utf-8") as f:
                        error_result = json.load(f)
                        error_msg = error_result.get("error", error_msg)
                except (OSError, json.JSONDecodeError, AttributeError) as read_exc:
                    # Keep the generic message; the subprocess failure is what matters.
                    logger.debug(
                        "could not read error JSON %s: %s", result_json_path, read_exc
                    )

            raise RuntimeError(f"Pipeline execution failed: {error_msg}") from e

        finally:
            # Clean up temporary result JSON
            if Path(result_json_path).exists():
                try:
                    Path(result_json_path).unlink()
                except OSError as unlink_exc:  # cleanup is best-effort, never fatal
                    logger.debug(
                        "could not remove %s: %s", result_json_path, unlink_exc
                    )

    except Exception as exc:  # noqa: BLE001 - best-effort boundary; falls back below
        logger.debug("%s: falling back after %r", __name__, exc)
        tb = traceback.format_exc()
        return SingleCellPipelineToolResult(
            ok=False,
            payload=None,
            error=tb,
            args=args,
        )


async def single_cell_tool_use_behavior(
    context: RunContextWrapper[SingleCellRunnerContext],
    results: List[FunctionToolResult],
) -> ToolsToFinalOutputResult:
    """
    Custom tool use behavior with retry logic.
    """
    ctx = context.context
    latest_result: FunctionToolResult = results[-1]

    tool_output: SingleCellPipelineToolResult = latest_result.output

    if tool_output.ok:
        final_summary = {
            "status": "ok",
            "attempts": ctx.attempts + 1,
            "args_used": tool_output.args,
            "result": tool_output.payload,
        }
        return ToolsToFinalOutputResult(
            is_final_output=True,
            final_output=final_summary,
        )

    ctx.attempts += 1
    ctx.last_args = tool_output.args
    if tool_output.error:
        ctx.errors.append(tool_output.error)

    if ctx.attempts >= ctx.max_attempts:
        final_failure = {
            "status": "failed",
            "attempts": ctx.attempts,
            "last_args": ctx.last_args,
            "errors": ctx.errors,
        }
        return ToolsToFinalOutputResult(
            is_final_output=True,
            final_output=final_failure,
        )

    return ToolsToFinalOutputResult(is_final_output=False)


def build_single_cell_runner_agent() -> Agent[SingleCellRunnerContext]:
    """
    Create the specialized Single-Cell Runner Agent.
    """
    instructions = """
        You are a Single-Cell 10x Genomics Pipeline Runner Agent for bioinformatics analysis.

        Your job:
        1. Read the user-provided arguments for the single-cell pipeline.
        2. Validate and adjust arguments before calling the tool:
           - Ensure single_10x_dir exists and contains required files (matrix.mtx, barcodes.tsv, features.tsv)
           - Verify output_dir can be created or already exists
           - Handle optional parameters (sample_label, group_label, geo_json_path, logos_dir)
           - Fix relative paths using absolute paths or resolve relative to current working directory
           - Validate integration_method is either "bbknn" or None
        3. Call the `run_single_cell_pipeline_tool` to execute the pipeline.
        4. If the tool returns an error:
           - Carefully read the stacktrace
           - Infer which argument(s) caused the problem (file paths, directory paths, permissions, etc.)
           - Modify only those arguments, keeping others unchanged
           - Try again, up to the maximum number of attempts
        5. When the pipeline succeeds or attempts are exhausted, return a concise,
           structured JSON summary with:
           - status: "ok" or "failed"
           - attempts
           - args_used (final args)
           - result (the tool's payload if ok)
           - errors (if any)

        Always reason step-by-step internally, but output only the final structured result.
        """

    agent = Agent[SingleCellRunnerContext](
        name="Single-Cell Runner Agent",
        instructions=instructions.strip(),
        model=llm_settings.LLM_MODEL_AGENT,
        tools=[run_single_cell_pipeline_tool],
        tool_use_behavior=single_cell_tool_use_behavior,
        model_settings=ModelSettings(
            tool_choice="required",
            temperature=0.0,
        ),
    )
    return agent


async def run_single_cell_agent_with_args(
    args: SingleCellPipelineArgs,
    max_attempts: int = 3,
) -> Dict[str, Any]:
    """
    High-level async entrypoint: runs the Single-Cell Runner Agent with given args.

    Returns response format from single_cell_tool_use_behavior:
    - Success: {
        "status": "ok",
        "attempts": 1,
        "args_used": {...},
        "result": {
            "status": "completed",
            "output_dir": "..."
        }
      }
    - Failure: {
        "status": "failed",
        "attempts": N,
        "last_args": {...},
        "errors": [...]
      }
    """
    agent = build_single_cell_runner_agent()

    context = SingleCellRunnerContext(max_attempts=max_attempts)

    user_input = (
        f"Run single-cell 10x pipeline with robust self-correction.\n\n"
        f"Initial arguments:\n{json.dumps(args.model_dump(), indent=2)}"
    )

    result = await Runner.run(
        starting_agent=agent,
        input=user_input,
        context=context,
        run_config=openrouter_run_config(model=llm_settings.LLM_MODEL_AGENT),
    )

    # Ensure final_output is a dictionary (not a JSON string or Python dict string)
    final_output = result.final_output
    logger.info(
        "Raw final_output type: %s, value: %s", type(final_output), final_output
    )

    # Handle different types of final_output
    if isinstance(final_output, dict):
        # Already a dictionary - return as-is
        logger.info("final_output is already a dict, returning as-is")
        return final_output
    elif isinstance(final_output, str):
        # Try to parse as JSON string first (most common case)
        try:
            parsed = json.loads(final_output)
            logger.info("Successfully parsed JSON string to dict: %s", parsed)
            return parsed
        except json.JSONDecodeError:
            # If JSON parsing fails, try parsing as Python literal (e.g., "{'key': 'value'}")
            # This handles cases where the agent framework converts dict to string representation
            try:
                parsed = ast.literal_eval(final_output)
                if isinstance(parsed, dict):
                    logger.info(
                        "Successfully parsed Python dict string to dict: %s", parsed
                    )
                    return parsed
                else:
                    logger.warning(
                        "Parsed Python literal but got non-dict type: %s", type(parsed)
                    )
                    return {
                        "status": "failed",
                        "error": f"Parsed Python literal but got non-dict type: {type(parsed)}",
                        "raw_output": final_output[:500]
                        if len(final_output) > 500
                        else final_output,
                    }
            except (ValueError, SyntaxError) as e:
                logger.exception(
                    "Failed to parse final_output as JSON or Python literal. Type: %s, Value (first 200 chars): %s, JSON Error: %s",
                    type(final_output),
                    final_output[:200],
                    str(e),
                )
                # Fallback to a structured error if parsing fails
                return {
                    "status": "failed",
                    "error": f"Invalid format from agent final_output (not JSON or Python dict): {str(e)}",
                    "raw_output": final_output[:500]
                    if len(final_output) > 500
                    else final_output,
                }
    else:
        # Unexpected type - try to convert to dict or return error
        logger.warning(
            "final_output has unexpected type: %s, value: %s",
            type(final_output),
            final_output,
        )
        # Try to convert to dict if it's a Pydantic model or similar
        if hasattr(final_output, "model_dump"):
            return final_output.model_dump()
        elif hasattr(final_output, "dict"):
            return final_output.dict()
        else:
            # Return error format
            return {
                "status": "failed",
                "error": f"Unexpected final_output type: {type(final_output)}",
                "raw_output": str(final_output)[:500],
            }


def run_single_cell_agent_with_args_sync(
    args: SingleCellPipelineArgs,
    max_attempts: int = 3,
) -> Any:
    """
    Synchronous wrapper for run_single_cell_agent_with_args.

    Handles event loop creation for Django/Gunicorn thread pool environments.
    """
    logger.info(
        "Running single-cell agent synchronously: single_10x_dir=%s, output_dir=%s, max_attempts=%s",
        args.single_10x_dir,
        args.output_dir,
        max_attempts,
    )

    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("Event loop is closed")
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        logger.info("Created new event loop for single-cell agent")

    agent = build_single_cell_runner_agent()
    context = SingleCellRunnerContext(max_attempts=max_attempts)

    user_input = (
        f"Run single-cell 10x pipeline with robust self-correction.\n\n"
        f"Initial arguments:\n{json.dumps(args.model_dump(), indent=2)}"
    )

    try:
        result = Runner.run_sync(
            starting_agent=agent,
            input=user_input,
            context=context,
            run_config=openrouter_run_config(model=llm_settings.LLM_MODEL_AGENT),
        )

        status = (
            result.final_output.get("status")
            if isinstance(result.final_output, dict)
            else "unknown"
        )
        logger.info("Single-cell agent completed: status=%s", status)
        return result.final_output
    except Exception as e:
        logger.exception("Single-cell agent execution failed: %s", e)
        raise


if __name__ == "__main__":
    # This __main__ block is a manual smoke-run, so it configures logging itself —
    # importing this module as a library must not touch the host's handlers.
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = SingleCellPipelineArgs(
        single_10x_dir="cellcyrix/single_cell_pipeline_agent/data/GSM6360681_N_HPV_NEG_2",
        output_dir="output1",
    )
    result = asyncio.run(run_single_cell_agent_with_args(args))
    logger.info("%s", result)

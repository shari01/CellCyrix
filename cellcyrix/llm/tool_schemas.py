"""Convert Anthropic-style tool definitions (name + input_schema) to OpenAI tools."""

from __future__ import annotations

import json
from typing import Any, Dict, List


def anthropic_tools_to_openai(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert Anthropic-style tool definitions to OpenAI function-tool schemas.

    Args:
        tools: Tool definitions with `name` / `description` / `input_schema`.

    Returns:
        The same tools in OpenAI `{"type": "function", ...}` form.
    """
    out: List[Dict[str, Any]] = []
    for t in tools:
        name = t.get("name")
        if not name:
            continue
        params = t.get("input_schema") or {"type": "object", "properties": {}}
        out.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": (t.get("description") or "")[:8000],
                    "parameters": params,
                },
            }
        )
    return out


def parse_tool_calls_from_openai_message(
    msg: Any,
) -> tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    """Return (text, tool_calls_legacy, assistant_message_dict).

    tool_calls_legacy matches prior Bedrock shape: {"id","name","input"}.
    assistant_message_dict is suitable for the next chat.completions turn.
    """
    text = (getattr(msg, "content", None) or "") or ""
    raw_calls = getattr(msg, "tool_calls", None) or []
    legacy: List[Dict[str, Any]] = []
    serial_tc: List[Dict[str, Any]] = []

    for tc in raw_calls:
        tid = getattr(tc, "id", None) or ""
        fn = getattr(tc, "function", None)
        fname = getattr(fn, "name", "") if fn else ""
        args_raw = getattr(fn, "arguments", "") if fn else "{}"
        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
        except json.JSONDecodeError:
            args = {}
        if tid and fname:
            legacy.append({"id": tid, "name": fname, "input": args})
        if hasattr(tc, "model_dump"):
            serial_tc.append(tc.model_dump())
        else:
            serial_tc.append(
                {
                    "id": tid,
                    "type": "function",
                    "function": {"name": fname, "arguments": args_raw or "{}"},
                }
            )

    assistant: Dict[str, Any] = {
        "role": "assistant",
        "content": text if text else None,
    }
    if serial_tc:
        assistant["tool_calls"] = serial_tc

    return text, legacy, assistant

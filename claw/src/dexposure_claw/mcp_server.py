"""Dependency-free stdio MCP server for DeXposure Claw."""
from __future__ import annotations

import json
import sys
from typing import Any


TOOLS = [
    {
        "name": "dexposure_health",
        "description": "Check that the DeXposure Claw MCP server is reachable.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "name": "dexposure_install_snippet",
        "description": "Return install configuration for Claude Code, Hermes, Codex, or generic MCP clients.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "enum": ["hermes", "codex", "mcp"],
                    "description": "Runtime target to generate config for.",
                }
            },
            "required": ["target"],
            "additionalProperties": False,
        },
    },
    {
        "name": "dexposure_list_benchmarks",
        "description": "List the six DeXposure-Bench benchmark IDs and human-readable names.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
]

BENCHMARKS = [
    {"id": "b1_forecast", "label": "B1 Forecast / Temporal risk forecasting"},
    {"id": "b2_warning", "label": "B2 Warning / Streaming early warning"},
    {"id": "b3_calibration", "label": "B3 Calibration / Predictive uncertainty"},
    {"id": "b4_stress", "label": "B4 Stress / What-if scenario fidelity"},
    {"id": "b5_decision", "label": "B5 Decision / Supervisory ticket quality"},
    {"id": "b6_robustness", "label": "B6 Robustness / Data-quality sensitivity"},
]


def _response(message_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": message_id, "result": result}


def _error(message_id: Any, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": message_id,
        "error": {"code": code, "message": message},
    }


def _text_result(value: Any) -> dict[str, Any]:
    text = value if isinstance(value, str) else json.dumps(value, indent=2)
    return {"content": [{"type": "text", "text": text}]}


def _call_tool(name: str, arguments: dict[str, Any] | None) -> dict[str, Any]:
    args = arguments or {}
    if name == "dexposure_health":
        return _text_result(
            {
                "status": "ok",
                "package": "dexposure-claw",
                "benchmarks": len(BENCHMARKS),
            }
        )
    if name == "dexposure_install_snippet":
        from .installer import render_install_snippet

        return _text_result(render_install_snippet(str(args["target"])))
    if name == "dexposure_list_benchmarks":
        return _text_result({"benchmarks": BENCHMARKS})
    raise ValueError(f"Unknown tool: {name}")


def _handle(message: dict[str, Any]) -> dict[str, Any] | None:
    method = message.get("method")
    message_id = message.get("id")

    if method == "initialize":
        return _response(
            message_id,
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": "dexposure-claw",
                    "version": "0.1.0",
                },
            },
        )
    if method in {"notifications/initialized", "notifications/cancelled"}:
        return None
    if method == "ping":
        return _response(message_id, {})
    if method == "tools/list":
        return _response(message_id, {"tools": TOOLS})
    if method == "tools/call":
        params = message.get("params") or {}
        try:
            result = _call_tool(
                name=str(params.get("name", "")),
                arguments=params.get("arguments") or {},
            )
            return _response(message_id, result)
        except Exception as exc:
            return _error(message_id, -32603, str(exc))
    if method in {"resources/list", "prompts/list"}:
        key = "resources" if method == "resources/list" else "prompts"
        return _response(message_id, {key: []})
    return _error(message_id, -32601, f"Method not found: {method}")


def _write(message: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(message, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def main() -> int:
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            message = json.loads(line)
            response = _handle(message)
        except Exception as exc:
            response = _error(None, -32700, str(exc))
        if response is not None:
            _write(response)
    return 0


def describe() -> dict[str, Any]:
    return {
        "name": "dexposure-claw",
        "status": "ready",
        "tools": [tool["name"] for tool in TOOLS],
    }


if __name__ == "__main__":
    raise SystemExit(main())

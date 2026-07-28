"""Model Context Protocol wire format: JSON-RPC 2.0, newline-delimited.

Only the tool surface is implemented (initialize, ping, tools/list,
tools/call) — the subset agents need. No external dependencies.
"""

from __future__ import annotations

import json
from typing import Any

from aire.core.errors import AireError

PROTOCOL_VERSION = "2025-06-18"


class MCPError(AireError):
    """MCP transport or protocol failure."""

    code = "mcp.error"


def make_request(request_id: int, method: str, params: dict[str, Any] | None = None) -> str:
    message: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
    if params is not None:
        message["params"] = params
    return json.dumps(message)


def make_notification(method: str, params: dict[str, Any] | None = None) -> str:
    message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        message["params"] = params
    return json.dumps(message)


def make_response(request_id: Any, result: Any) -> str:
    return json.dumps({"jsonrpc": "2.0", "id": request_id, "result": result})


def make_error(request_id: Any, code: int, message: str) -> str:
    return json.dumps(
        {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}
    )


def parse_message(line: str) -> dict[str, Any]:
    """Parse one newline-delimited JSON-RPC message."""
    try:
        data = json.loads(line)
    except json.JSONDecodeError as exc:
        raise MCPError(f"invalid JSON-RPC frame: {exc}", cause=exc) from exc
    if not isinstance(data, dict) or data.get("jsonrpc") != "2.0":
        raise MCPError("not a JSON-RPC 2.0 message", context={"frame": line[:200]})
    return data

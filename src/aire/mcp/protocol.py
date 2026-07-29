"""Model Context Protocol wire format: JSON-RPC 2.0, newline-delimited.

aire implements a **subset of MCP**: tools, resources, prompts, plus client-side
roots/sampling handlers and progress notifications. No external dependencies.
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


def make_progress_notification(
    progress_token: str | int,
    progress: float,
    *,
    total: float | None = None,
    message: str | None = None,
) -> str:
    """Build a ``notifications/progress`` JSON-RPC notification."""
    params: dict[str, Any] = {"progressToken": progress_token, "progress": progress}
    if total is not None:
        params["total"] = total
    if message is not None:
        params["message"] = message
    return make_notification("notifications/progress", params)


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


def client_capabilities(
    *,
    roots: bool = True,
    sampling: bool = True,
) -> dict[str, Any]:
    """Default client capability advertisement for initialize."""
    caps: dict[str, Any] = {}
    if roots:
        caps["roots"] = {"listChanged": False}
    if sampling:
        caps["sampling"] = {}
    return caps

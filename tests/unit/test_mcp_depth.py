"""MCP protocol depth: progress, capabilities, HTTP describe surface."""

from __future__ import annotations

import json

from aire.mcp.http_client import MCPHttpClient
from aire.mcp.protocol import (
    client_capabilities,
    make_progress_notification,
    parse_message,
)
from aire.mcp.server import MCPServer
from aire.tools.builtins import builtin_tools


def test_make_progress_notification() -> None:
    raw = make_progress_notification("tok-1", 0.5, total=1.0, message="halfway")
    msg = parse_message(raw)
    assert msg["method"] == "notifications/progress"
    assert msg["params"]["progressToken"] == "tok-1"
    assert msg["params"]["progress"] == 0.5
    assert msg["params"]["total"] == 1.0


def test_client_capabilities() -> None:
    caps = client_capabilities()
    assert "roots" in caps
    assert "sampling" in caps


def test_mcp_server_progress_flag() -> None:
    frames: list[str] = []
    calc = next(t for t in builtin_tools() if t.name == "calculator")
    server = MCPServer([calc], knowledge=False, progress_writer=frames.append)
    assert server.describe()["progress"] is True


def test_mcp_http_describe_methods() -> None:
    client = MCPHttpClient("http://localhost:8000/mcp")
    desc = client.describe()
    assert "resources/list" in desc["methods"]
    assert "prompts/get" in desc["methods"]
    assert desc["roots"] == 1


def test_server_emits_progress_around_tool_call() -> None:
    from aire.models.base import run_sync

    frames: list[str] = []
    calc = next(t for t in builtin_tools() if t.name == "calculator")
    server = MCPServer([calc], knowledge=False, progress_writer=frames.append)
    result = run_sync(
        server.handle(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "calculator",
                    "arguments": {"expression": "1+1"},
                    "_meta": {"progressToken": "p1"},
                },
            }
        )
    )
    assert result is not None
    assert result["result"]["isError"] is False
    assert len(frames) == 2
    assert json.loads(frames[0])["params"]["progress"] == 0.0
    assert json.loads(frames[1])["params"]["progress"] == 1.0

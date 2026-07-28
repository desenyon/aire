"""MCP tests: server protocol handling + client/server round-trip over stdio."""

from __future__ import annotations

import json
import sys

import pytest

from aire.mcp.client import MCPClient
from aire.mcp.server import MCPServer
from aire.tools.builtins import builtin_tools


@pytest.mark.anyio
async def test_server_initialize_and_list() -> None:
    server = MCPServer(builtin_tools())
    response = await server.handle(
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
    )
    assert response is not None
    assert response["result"]["serverInfo"]["name"] == "aire"
    assert response["result"]["capabilities"]["tools"]

    listing = await server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    names = {t["name"] for t in listing["result"]["tools"]}
    assert "calculator" in names
    calc = next(t for t in listing["result"]["tools"] if t["name"] == "calculator")
    assert "expression" in calc["inputSchema"]["properties"]


@pytest.mark.anyio
async def test_server_call_tool_success_and_error() -> None:
    server = MCPServer(builtin_tools())
    ok = await server.handle(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "calculator", "arguments": {"expression": "6 * 7"}},
        }
    )
    assert ok["result"]["isError"] is False
    assert float(ok["result"]["content"][0]["text"]) == 42.0

    bad = await server.handle(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "calculator", "arguments": {"expression": "1 +"}},
        }
    )
    assert bad["result"]["isError"] is True

    unknown = await server.handle(
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {"name": "nope", "arguments": {}},
        }
    )
    assert "error" in unknown

    notification = await server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"})
    assert notification is None


@pytest.mark.anyio
async def test_client_server_roundtrip_stdio() -> None:
    """Full end-to-end: spawn `python -m aire.mcp`, list tools, call one."""
    async with MCPClient([sys.executable, "-m", "aire.mcp"]) as client:
        assert client.server_info.get("name") == "aire"
        infos = await client.list_tools()
        assert any(t["name"] == "calculator" for t in infos)

        text = await client.call_tool("calculator", {"expression": "2 ** 10"})
        assert float(text) == 1024.0

        tools = await client.tools()
        calculator = next(t for t in tools if t.name == "calculator")
        assert "expression" in calculator.spec.input_schema.get("properties", {})
        result = await calculator.execute({"expression": "1 + 2"})
        assert result.ok, result.error
        assert float(result.output) == 3.0
    assert client.describe()["connected"] is False


def test_protocol_frames() -> None:
    from aire.mcp.protocol import MCPError, make_request, parse_message

    frame = make_request(7, "tools/list")
    parsed = parse_message(frame)
    assert parsed["id"] == 7
    assert parsed["method"] == "tools/list"

    with pytest.raises(MCPError):
        parse_message(json.dumps({"not": "jsonrpc"}))

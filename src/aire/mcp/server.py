"""MCP server: expose aire tools to any MCP-speaking host (Claude Code, IDEs).

Zero dependencies — newline-delimited JSON-RPC 2.0 over stdio::

    aire mcp-serve                      # builtin + registered tools
    python -m aire.mcp                  # same, from anywhere

Or embed::

    server = MCPServer([my_tool])
    await server.serve_stdio()
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

from aire._version import __version__
from aire.core.errors import AireError
from aire.mcp.protocol import PROTOCOL_VERSION, MCPError, make_error, parse_message
from aire.tools.tool import Tool

_METHOD_NOT_FOUND = -32601
_INTERNAL_ERROR = -32603


class MCPServer:
    """Expose aire tools over the Model Context Protocol."""

    def __init__(
        self,
        tools: list[Tool] | None = None,
        *,
        name: str = "aire",
        version: str = __version__,
    ) -> None:
        self._tools: dict[str, Tool] = {t.name: t for t in tools or []}
        self.name = name
        self.version = version

    def add_tool(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    # -- protocol ------------------------------------------------------------------

    async def handle(self, message: dict[str, Any]) -> dict[str, Any] | None:
        """Handle one parsed JSON-RPC message; returns the response or None
        for notifications."""
        method = message.get("method", "")
        request_id = message.get("id")
        if request_id is None:  # notification: never respond
            return None
        try:
            result = await self._dispatch(method, message.get("params") or {})
        except MCPError as exc:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": _METHOD_NOT_FOUND, "message": exc.message},
            }
        except Exception as exc:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": _INTERNAL_ERROR, "message": str(exc)},
            }
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    async def _dispatch(self, method: str, params: dict[str, Any]) -> Any:
        if method == "initialize":
            return {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": self.name, "version": self.version},
            }
        if method == "ping":
            return {}
        if method == "tools/list":
            return {"tools": [self._tool_info(t) for t in self._tools.values()]}
        if method == "tools/call":
            return await self._call_tool(params)
        raise MCPError(f"unknown method: {method!r}", context={"method": method})

    @staticmethod
    def _tool_info(tool: Tool) -> dict[str, Any]:
        return {
            "name": tool.spec.name,
            "description": tool.spec.description,
            "inputSchema": tool.spec.input_schema,
        }

    async def _call_tool(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name", "")
        tool = self._tools.get(name)
        if tool is None:
            raise MCPError(f"unknown tool: {name!r}", context={"tool": name})
        result = await tool.execute(params.get("arguments") or {})
        if result.ok:
            text = result.output if isinstance(result.output, str) else _as_json(result.output)
            return {"content": [{"type": "text", "text": text}], "isError": False}
        error_text = result.error or "tool failed"
        return {"content": [{"type": "text", "text": error_text}], "isError": True}

    # -- stdio transport --------------------------------------------------------------

    async def serve_stdio(self, stdin: Any = None, stdout: Any = None) -> None:
        """Serve newline-delimited JSON-RPC on stdin/stdout (blocking loop)."""
        stdin = stdin or sys.stdin
        stdout = stdout or sys.stdout
        loop = asyncio.get_running_loop()
        while True:
            line = await loop.run_in_executor(None, stdin.readline)
            if not line:
                return
            line = line.strip()
            if not line:
                continue
            try:
                message = parse_message(line)
                response = await self.handle(message)
            except AireError as exc:
                stdout.write(make_error(None, _INTERNAL_ERROR, exc.message) + "\n")
                stdout.flush()
                continue
            if response is not None:
                stdout.write(json.dumps(response) + "\n")
                stdout.flush()

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "mcp_server",
            "name": self.name,
            "version": self.version,
            "protocol": PROTOCOL_VERSION,
            "tools": sorted(self._tools),
        }


def _as_json(value: Any) -> str:
    return json.dumps(value, default=str)


def default_server() -> MCPServer:
    """Server exposing builtin tools plus everything registered on the runtime."""
    from aire.ai import default_runtime
    from aire.tools.builtins import builtin_tools

    server = MCPServer(list(builtin_tools()))
    runtime = default_runtime()
    for name in runtime.tools.names():
        server.add_tool(runtime.tools.create(name))
    return server


async def amain() -> None:
    await default_server().serve_stdio()


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()

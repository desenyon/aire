"""MCP server: expose aire tools to any MCP-speaking host (Claude Code, IDEs).

Implements a **subset of MCP**: tools, resources, and prompts over stdio
(newline-delimited JSON-RPC 2.0). Zero dependencies::

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
_INVALID_PARAMS = -32602
_INTERNAL_ERROR = -32603
_APPLICATION_ERROR = -32000


class MCPServer:
    """Expose aire tools over a subset of MCP (tools, resources, prompts / stdio)."""

    def __init__(
        self,
        tools: list[Tool] | None = None,
        *,
        name: str = "aire",
        version: str = __version__,
        knowledge: bool = True,
        progress_writer: Any | None = None,
    ) -> None:
        self._tools: dict[str, Tool] = {t.name: t for t in tools or []}
        self.name = name
        self.version = version
        self.knowledge = knowledge
        self._progress_writer = progress_writer

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
            code = _METHOD_NOT_FOUND if "unknown method" in exc.message else _APPLICATION_ERROR
            if "unknown tool" in exc.message or "disabled" in exc.message:
                code = _INVALID_PARAMS
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": code, "message": exc.message},
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
            capabilities: dict[str, Any] = {"tools": {"listChanged": False}}
            if self.knowledge:
                capabilities["resources"] = {"subscribe": False, "listChanged": False}
                capabilities["prompts"] = {"listChanged": False}
            return {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": capabilities,
                "serverInfo": {"name": self.name, "version": self.version},
            }
        if method == "ping":
            return {}
        if method == "tools/list":
            return {"tools": [self._tool_info(t) for t in self._tools.values()]}
        if method == "tools/call":
            return await self._call_tool(params)
        if method == "resources/list":
            return {
                "resources": [
                    {
                        "uri": r.uri,
                        "name": r.name,
                        "description": r.description,
                        "mimeType": r.mime_type,
                    }
                    for r in self._resources()
                ]
            }
        if method == "resources/read":
            return self._read_resource(str(params.get("uri", "")))
        if method == "prompts/list":
            return {"prompts": [self._prompt_info(p) for p in self._prompts()]}
        if method == "prompts/get":
            return self._get_prompt(str(params.get("name", "")), params.get("arguments"))
        raise MCPError(f"unknown method: {method!r}", context={"method": method})

    # -- knowledge (resources + prompts) ----------------------------------------------

    def _resources(self) -> list[Any]:
        if not self.knowledge:
            return []
        from aire.mcp.knowledge import builtin_resources

        return builtin_resources()

    def _read_resource(self, uri: str) -> dict[str, Any]:
        from aire.mcp.knowledge import read_resource

        if not self.knowledge:
            raise MCPError("resources are disabled on this server", context={"uri": uri})
        text = read_resource(uri)
        mime = next((r.mime_type for r in self._resources() if r.uri == uri), "text/markdown")
        return {"contents": [{"uri": uri, "mimeType": mime, "text": text}]}

    def _prompts(self) -> list[Any]:
        if not self.knowledge:
            return []
        from aire.mcp.knowledge import builtin_prompts

        return builtin_prompts()

    @staticmethod
    def _prompt_info(prompt: Any) -> dict[str, Any]:
        return {
            "name": prompt.name,
            "description": prompt.description,
            "arguments": [a.model_dump() for a in prompt.arguments],
        }

    def _get_prompt(self, name: str, arguments: dict[str, Any] | None) -> dict[str, Any]:
        from aire.mcp.knowledge import get_prompt

        if not self.knowledge:
            raise MCPError("prompts are disabled on this server", context={"name": name})
        return get_prompt(name, arguments)

    @staticmethod
    def _tool_info(tool: Tool) -> dict[str, Any]:
        return {
            "name": tool.spec.name,
            "description": tool.spec.description,
            "inputSchema": tool.spec.input_schema,
        }

    async def _call_tool(self, params: dict[str, Any]) -> dict[str, Any]:
        from aire.mcp.protocol import make_progress_notification

        name = params.get("name", "")
        tool = self._tools.get(name)
        if tool is None:
            raise MCPError(f"unknown tool: {name!r}", context={"tool": name})
        meta = params.get("_meta") if isinstance(params.get("_meta"), dict) else {}
        assert isinstance(meta, dict)
        token = meta.get("progressToken")
        if token is not None and self._progress_writer is not None:
            self._progress_writer(
                make_progress_notification(token, 0.0, total=1.0, message=f"start {name}")
            )
        result = await tool.execute(params.get("arguments") or {})
        if token is not None and self._progress_writer is not None:
            self._progress_writer(
                make_progress_notification(token, 1.0, total=1.0, message=f"done {name}")
            )
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
            "subset": "tools, resources, prompts over stdio (+ progress notifications)",
            "name": self.name,
            "version": self.version,
            "protocol": PROTOCOL_VERSION,
            "tools": sorted(self._tools),
            "knowledge": self.knowledge,
            "progress": self._progress_writer is not None,
            "resources": [r.uri for r in self._resources()],
            "prompts": [p.name for p in self._prompts()],
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

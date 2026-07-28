"""MCP client: consume any MCP server as first-class aire tools.

Spawns the server as a subprocess and speaks newline-delimited JSON-RPC over
its stdio — zero dependencies::

    async with MCPClient(["python", "-m", "aire.mcp"]) as client:
        tools = await client.tools()          # list[aire Tool]
        result = await tools[0].execute({"text": "hi"})
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

from pydantic import ConfigDict, create_model

from aire.mcp.protocol import MCPError, make_notification, make_request, parse_message
from aire.tools.tool import Tool


class MCPClient:
    """Stdio MCP client for one server subprocess."""

    def __init__(self, command: list[str], *, startup_timeout: float = 15.0) -> None:
        if not command:
            raise MCPError("MCPClient requires a command to spawn")
        self.command = command
        self.startup_timeout = startup_timeout
        self._process: asyncio.subprocess.Process | None = None
        self._next_id = 0
        self._write_lock = asyncio.Lock()
        self.server_info: dict[str, Any] = {}

    # -- lifecycle -----------------------------------------------------------------

    async def connect(self) -> MCPClient:
        self._process = await asyncio.create_subprocess_exec(
            *self.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            response = await asyncio.wait_for(
                self._request(
                    "initialize",
                    {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {},
                        "clientInfo": {"name": "aire", "version": _aire_version()},
                    },
                ),
                timeout=self.startup_timeout,
            )
        except Exception:
            await self.close()
            raise
        self.server_info = dict(response.get("serverInfo", {}))
        await self._send(make_notification("notifications/initialized"))
        return self

    async def close(self) -> None:
        if self._process is None:
            return
        process, self._process = self._process, None
        if process.stdin is not None:
            with contextlib.suppress(Exception):
                process.stdin.close()
        with contextlib.suppress(ProcessLookupError):
            process.terminate()
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(process.wait(), timeout=5.0)

    async def __aenter__(self) -> MCPClient:
        return await self.connect()

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.close()

    # -- protocol --------------------------------------------------------------------

    async def _send(self, payload: str) -> None:
        if self._process is None or self._process.stdin is None:
            raise MCPError("MCP client is not connected")
        self._process.stdin.write((payload + "\n").encode())
        await self._process.stdin.drain()

    async def _request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        async with self._write_lock:
            self._next_id += 1
            request_id = self._next_id
            await self._send(make_request(request_id, method, params))
            if self._process is None or self._process.stdout is None:
                raise MCPError("MCP client is not connected")
            while True:
                line = await self._process.stdout.readline()
                if not line:
                    raise MCPError(
                        f"MCP server closed the connection during {method!r}",
                        context={"method": method},
                    )
                message = parse_message(line.decode().strip())
                if message.get("id") != request_id:
                    continue  # notification or unrelated frame
                if "error" in message:
                    error = message["error"]
                    raise MCPError(
                        f"MCP error {error.get('code')}: {error.get('message')}",
                        context={"method": method},
                    )
                return message.get("result", {})

    # -- tool surface ------------------------------------------------------------------

    async def list_tools(self) -> list[dict[str, Any]]:
        """Raw tool descriptors as returned by the server."""
        result = await self._request("tools/list")
        return list(result.get("tools", []))

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        """Call a remote tool; returns the joined text content."""
        result = await self._request("tools/call", {"name": name, "arguments": arguments or {}})
        text = "".join(
            part.get("text", "")
            for part in result.get("content", [])
            if isinstance(part, dict) and part.get("type") == "text"
        )
        if result.get("isError"):
            raise MCPError(f"remote tool {name!r} failed: {text}", context={"tool": name})
        return text

    async def tools(self) -> list[Tool]:
        """Adapt every remote tool into a first-class aire :class:`Tool`."""
        return [self._adapt(info) for info in await self.list_tools()]

    def _adapt(self, info: dict[str, Any]) -> Tool:
        client = self
        name = str(info.get("name", "remote"))

        async def _remote() -> str:  # replaced via _invoke override below
            return ""

        tool = Tool(_remote, name=name, description=str(info.get("description", "")))
        tool.spec.input_schema = info.get("inputSchema") or {"type": "object", "properties": {}}
        tool._args_model = create_model(  # accept the server's schema verbatim
            f"mcp_{name}_args",
            __config__=ConfigDict(extra="allow"),
        )

        async def _invoke(kwargs: dict[str, Any]) -> str:
            return await client.call_tool(name, kwargs)

        tool._invoke = _invoke  # type: ignore[method-assign]
        return tool

    # -- knowledge surface (resources + prompts) ----------------------------------------

    async def list_resources(self) -> list[dict[str, Any]]:
        """Raw resource descriptors as returned by the server."""
        result = await self._request("resources/list")
        return list(result.get("resources", []))

    async def read_resource(self, uri: str) -> str:
        """Read a resource's text content (e.g. ``aire://guide``)."""
        result = await self._request("resources/read", {"uri": uri})
        return "".join(str(part.get("text", "")) for part in result.get("contents", []))

    async def list_prompts(self) -> list[dict[str, Any]]:
        """Raw prompt descriptors as returned by the server."""
        result = await self._request("prompts/list")
        return list(result.get("prompts", []))

    async def get_prompt(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        """Render a remote prompt; returns the joined message text."""
        result = await self._request("prompts/get", {"name": name, "arguments": arguments or {}})
        parts: list[str] = []
        for message in result.get("messages", []):
            content = message.get("content", {})
            if isinstance(content, dict) and content.get("type") == "text":
                parts.append(str(content.get("text", "")))
        return "\n".join(parts)

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "mcp_client",
            "command": self.command,
            "server": self.server_info,
            "connected": self._process is not None,
        }


def _aire_version() -> str:
    from aire._version import __version__

    return __version__

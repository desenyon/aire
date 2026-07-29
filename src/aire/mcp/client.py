"""MCP client: consume any MCP server as first-class aire tools.

Spawns the server as a subprocess and speaks newline-delimited JSON-RPC over
its stdio — zero dependencies. Implements a **subset of MCP**: tools,
resources, and prompts over stdio::

    async with MCPClient(["python", "-m", "aire.mcp"]) as client:
        tools = await client.tools()          # list[aire Tool]
        result = await tools[0].execute({"text": "hi"})
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

from pydantic import ConfigDict, create_model

from aire.mcp.protocol import (
    MCPError,
    client_capabilities,
    make_error,
    make_notification,
    make_request,
    make_response,
    parse_message,
)
from aire.tools.tool import Tool

_log = logging.getLogger("aire.mcp.client")


class MCPClient:
    """Stdio MCP client for one server subprocess (tools/resources/prompts subset)."""

    def __init__(
        self,
        command: list[str],
        *,
        startup_timeout: float = 15.0,
        roots: list[dict[str, Any]] | None = None,
        sampling_handler: Any | None = None,
        on_progress: Any | None = None,
    ) -> None:
        if not command:
            raise MCPError("MCPClient requires a command to spawn")
        self.command = command
        self.startup_timeout = startup_timeout
        self.roots = roots or [{"uri": "file://.", "name": "cwd"}]
        self.sampling_handler = sampling_handler
        self.on_progress = on_progress
        self._process: asyncio.subprocess.Process | None = None
        self._next_id = 0
        self._write_lock = asyncio.Lock()
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._reader_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self.server_info: dict[str, Any] = {}

    # -- lifecycle -----------------------------------------------------------------

    async def connect(self) -> MCPClient:
        self._process = await asyncio.create_subprocess_exec(
            *self.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._reader_task = asyncio.create_task(self._read_loop())
        if self._process.stderr is not None:
            self._stderr_task = asyncio.create_task(self._stderr_loop())
        try:
            response = await asyncio.wait_for(
                self._request(
                    "initialize",
                    {
                        "protocolVersion": "2025-06-18",
                        "capabilities": client_capabilities(
                            roots=True, sampling=self.sampling_handler is not None
                        ),
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
        if self._reader_task is not None:
            self._reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reader_task
            self._reader_task = None
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(MCPError("MCP client closed while request pending"))
        self._pending.clear()
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

    async def _stderr_loop(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return
        while True:
            line = await process.stderr.readline()
            if not line:
                return
            _log.warning("mcp server stderr: %s", line.decode(errors="replace").rstrip())

    async def _read_loop(self) -> None:  # noqa: C901
        process = self._process
        if process is None or process.stdout is None:
            return
        try:
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                try:
                    message = parse_message(line.decode().strip())
                except MCPError as exc:
                    _log.warning("dropping invalid MCP frame: %s", exc)
                    continue
                request_id = message.get("id")
                method = message.get("method")
                if request_id is None:
                    if method == "notifications/progress" and self.on_progress is not None:
                        try:
                            self.on_progress(message.get("params") or {})
                        except Exception:
                            _log.exception("on_progress handler failed")
                    continue
                if method:
                    await self._handle_server_request(message)
                    continue
                try:
                    key = int(request_id)
                except (TypeError, ValueError):
                    _log.warning("MCP response with non-int id=%r dropped", request_id)
                    continue
                fut = self._pending.pop(key, None)
                if fut is None:
                    _log.warning("MCP response for unknown id=%s dropped", key)
                    continue
                if not fut.done():
                    fut.set_result(message)
        finally:
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(
                        MCPError("MCP server closed the connection while requests pending")
                    )
            self._pending.clear()

    async def _handle_server_request(self, message: dict[str, Any]) -> None:
        request_id = message.get("id")
        method = message.get("method")
        params = message.get("params") or {}
        try:
            if method == "roots/list":
                result: Any = {"roots": list(self.roots)}
            elif method == "sampling/createMessage":
                if self.sampling_handler is None:
                    await self._send(
                        make_error(request_id, -32601, "sampling handler not configured")
                    )
                    return
                maybe = self.sampling_handler(params)
                result = await maybe if asyncio.iscoroutine(maybe) else maybe
            else:
                await self._send(make_error(request_id, -32601, f"unsupported method: {method}"))
                return
            await self._send(make_response(request_id, result))
        except Exception as exc:
            await self._send(make_error(request_id, -32603, str(exc)))

    async def _request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        async with self._write_lock:
            self._next_id += 1
            request_id = self._next_id
            loop = asyncio.get_running_loop()
            fut: asyncio.Future[dict[str, Any]] = loop.create_future()
            self._pending[request_id] = fut
            try:
                await self._send(make_request(request_id, method, params))
            except Exception:
                self._pending.pop(request_id, None)
                raise
        message = await fut
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
            "subset": "tools, resources, prompts over stdio",
            "command": self.command,
            "server": self.server_info,
            "connected": self._process is not None,
            "pending": len(self._pending),
        }


def _aire_version() -> str:
    from aire._version import __version__

    return __version__

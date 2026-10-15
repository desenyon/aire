"""MCP Streamable HTTP client (subset).

Connects to an MCP server over HTTP JSON-RPC (e.g. ``http://localhost:8000/mcp``).
Supports initialize, tools/resources/prompts, plus client-side roots/sampling
capability advertisement and progress callbacks. Session continuity uses the
``mcp-session-id`` response header when the server provides it.

Requires ``httpx`` (core dependency).
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from typing import Any

from aire.mcp.protocol import (
    PROTOCOL_VERSION,
    MCPError,
    client_capabilities,
    make_notification,
    make_request,
)
from aire.tools.schema import schema_model
from aire.tools.tool import Tool
from aire.tools.types import SideEffect

_log = logging.getLogger("aire.mcp.http_client")

_SESSION_HEADER = "mcp-session-id"


class MCPHttpClient:
    """HTTP JSON-RPC MCP client (streamable HTTP transport subset)."""

    def __init__(
        self,
        url: str,
        *,
        timeout: float = 30.0,
        roots: list[dict[str, Any]] | None = None,
        sampling_handler: Any | None = None,
        on_progress: Any | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        if not url:
            raise MCPError("MCPHttpClient requires a base URL")
        if timeout <= 0:
            raise MCPError("MCP deadlines must be positive")
        self.url = url.rstrip("/")
        self.timeout = timeout
        self._extra_headers = dict(headers or {})
        self.roots = roots or [{"uri": "file://.", "name": "cwd"}]
        self.sampling_handler = sampling_handler
        self.on_progress = on_progress
        self._next_id = 0
        self._session_id: str | None = None
        self._http: Any = None
        self.server_info: dict[str, Any] = {}

    async def connect(self) -> MCPHttpClient:
        import httpx

        self._http = httpx.AsyncClient(timeout=self.timeout)
        try:
            result = await self._request(
                "initialize",
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": client_capabilities(
                        roots=True, sampling=self.sampling_handler is not None
                    ),
                    "clientInfo": {"name": "aire", "version": _aire_version()},
                },
            )
        except BaseException:
            await self.close()
            raise
        self.server_info = dict(result.get("serverInfo", {}))
        try:
            await self._notify("notifications/initialized")
        except BaseException:
            await self.close()
            raise
        return self

    async def close(self) -> None:
        http, self._http = self._http, None
        if http is not None:
            await http.aclose()

    async def __aenter__(self) -> MCPHttpClient:
        return await self.connect()

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.close()

    def _headers(self) -> dict[str, str]:
        headers = {
            **self._extra_headers,
            "MCP-Protocol-Version": PROTOCOL_VERSION,
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self._session_id:
            headers[_SESSION_HEADER] = self._session_id
        return headers

    async def _post(self, body: str) -> dict[str, Any]:
        if self._http is None:
            raise MCPError("MCP HTTP client is not connected")
        expected_id = json.loads(body).get("id")
        async with self._http.stream(
            "POST", self.url, content=body, headers=self._headers()
        ) as response:
            session = response.headers.get(_SESSION_HEADER)
            if session:
                self._session_id = session
            if response.status_code >= 400:
                raise MCPError(
                    f"MCP HTTP {response.status_code}", context={"status": response.status_code}
                )
            if "text/event-stream" not in response.headers.get("content-type", "").lower():
                try:
                    message = json.loads(await response.aread())
                except (ValueError, UnicodeDecodeError) as exc:
                    raise MCPError("invalid JSON-RPC HTTP body") from exc
                result = await self._route_message(message, expected_id)
                if result is None:
                    raise MCPError("HTTP response has no matching JSON-RPC result")
                return result
            return await self._read_sse_response(response, expected_id)

    async def _read_sse_response(self, response: Any, expected_id: Any) -> dict[str, Any]:
        data_lines: list[str] = []
        async for line in response.aiter_lines():
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
            elif not line and data_lines:
                try:
                    message = json.loads("\n".join(data_lines))
                except ValueError as exc:
                    raise MCPError("invalid JSON-RPC SSE frame") from exc
                data_lines = []
                result = await self._route_message(message, expected_id)
                if result is not None:
                    return result
        raise MCPError("no matching JSON-RPC response in SSE stream")

    async def _route_message(self, message: Any, expected_id: Any) -> dict[str, Any] | None:
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            raise MCPError("invalid JSON-RPC response")
        method = message.get("method")
        request_id = message.get("id")
        if method and request_id is not None:
            await self._server_request(message)
            return None
        if method == "notifications/progress" and self.on_progress is not None:
            value = self.on_progress(message.get("params") or {})
            if inspect.isawaitable(value):
                await value
            return None
        if request_id == expected_id and ("result" in message or "error" in message):
            return message
        if method:
            return None
        raise MCPError("JSON-RPC response ID does not match request")

    async def _server_request(self, message: dict[str, Any]) -> None:
        result: Any
        reply: dict[str, Any] = {"jsonrpc": "2.0", "id": message["id"]}
        try:
            if message["method"] == "roots/list":
                result = {"roots": list(self.roots)}
            elif (
                message["method"] == "sampling/createMessage" and self.sampling_handler is not None
            ):
                result = self.sampling_handler(message.get("params") or {})
                if inspect.isawaitable(result):
                    result = await result
            else:
                reply["error"] = {"code": -32601, "message": "unsupported client method"}
                result = None
            if "error" not in reply:
                reply["result"] = result
        except Exception:
            reply["error"] = {"code": -32603, "message": "client callback failed"}
        response = await self._http.post(self.url, json=reply, headers=self._headers())
        if response.status_code >= 400:
            raise MCPError("server rejected callback response")

    async def _request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        self._next_id += 1
        import httpx

        try:
            message = await asyncio.wait_for(
                self._post(make_request(self._next_id, method, params)), timeout=self.timeout
            )
        except TimeoutError as exc:
            raise MCPError(f"MCP request timed out: {method}") from exc
        except httpx.HTTPError as exc:
            raise MCPError(f"MCP HTTP transport failed: {method}") from exc
        if "error" in message:
            error = message["error"]
            raise MCPError(
                f"MCP error {error.get('code')}: {error.get('message')}",
                context={"method": method},
            )
        return message.get("result", {})

    async def _notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        if self._http is None:
            raise MCPError("MCP HTTP client is not connected")
        response = await self._http.post(
            self.url,
            content=make_notification(method, params),
            headers=self._headers(),
        )
        session = response.headers.get(_SESSION_HEADER) or response.headers.get("Mcp-Session-Id")
        if session:
            self._session_id = session
        if response.status_code >= 400:
            raise MCPError(f"MCP notification {method} failed: HTTP {response.status_code}")

    async def list_tools(self) -> list[dict[str, Any]]:
        result = await self._request("tools/list")
        return list(result.get("tools", []))

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        result = await self._request("tools/call", {"name": name, "arguments": arguments or {}})
        text = "".join(
            part.get("text", "")
            for part in result.get("content", [])
            if isinstance(part, dict) and part.get("type") == "text"
        )
        if result.get("isError"):
            raise MCPError(f"remote tool {name!r} failed: {text}", context={"tool": name})
        return text

    async def list_resources(self) -> list[dict[str, Any]]:
        result = await self._request("resources/list")
        return list(result.get("resources", []))

    async def read_resource(self, uri: str) -> str:
        result = await self._request("resources/read", {"uri": uri})
        contents = result.get("contents") or []
        return "".join(str(part.get("text") or "") for part in contents if isinstance(part, dict))

    async def list_prompts(self) -> list[dict[str, Any]]:
        result = await self._request("prompts/list")
        return list(result.get("prompts", []))

    async def get_prompt(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return dict(
            await self._request("prompts/get", {"name": name, "arguments": arguments or {}})
        )

    async def tools(self) -> list[Tool]:
        return [self._adapt(info) for info in await self.list_tools()]

    def _adapt(self, info: dict[str, Any]) -> Tool:
        client = self
        name = str(info.get("name", "remote"))

        async def _remote() -> str:
            return ""

        tool = Tool(
            _remote,
            name=name,
            description=str(info.get("description", "")),
            side_effect=SideEffect.EXTERNAL_SIDE_EFFECT,
        )
        tool.spec.input_schema = info.get("inputSchema") or {"type": "object", "properties": {}}
        tool._args_model = schema_model("mcp_http_client_" + name, tool.spec.input_schema)

        async def _invoke(kwargs: dict[str, Any]) -> str:
            return await client.call_tool(name, kwargs)

        tool._invoke = _invoke  # type: ignore[method-assign]
        return tool

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "mcp_http_client",
            "subset": "streamable HTTP transport subset",
            "url": self.url,
            "session_id": self._session_id,
            "server": self.server_info,
            "connected": self._http is not None,
            "roots": len(self.roots),
            "sampling": self.sampling_handler is not None,
            "methods": [
                "initialize",
                "tools/list",
                "tools/call",
                "resources/list",
                "resources/read",
                "prompts/list",
                "prompts/get",
            ],
        }


def _aire_version() -> str:
    from aire._version import __version__

    return __version__

"""MCP Streamable HTTP client (subset).

Connects to an MCP server over HTTP JSON-RPC (e.g. ``http://localhost:8000/mcp``).
Supports initialize, tools/resources/prompts, plus client-side roots/sampling
capability advertisement and progress callbacks. Session continuity uses the
``mcp-session-id`` response header when the server provides it.

Requires ``httpx`` (core dependency).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import ConfigDict, create_model

from aire.mcp.protocol import (
    PROTOCOL_VERSION,
    MCPError,
    client_capabilities,
    make_notification,
    make_request,
)
from aire.tools.tool import Tool

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
    ) -> None:
        if not url:
            raise MCPError("MCPHttpClient requires a base URL")
        self.url = url.rstrip("/")
        self.timeout = timeout
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
        except Exception:
            await self.close()
            raise
        self.server_info = dict(result.get("serverInfo", {}))
        await self._notify("notifications/initialized")
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
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self._session_id:
            headers[_SESSION_HEADER] = self._session_id
        return headers

    async def _post(self, body: str) -> dict[str, Any]:
        if self._http is None:
            raise MCPError("MCP HTTP client is not connected")
        response = await self._http.post(self.url, content=body, headers=self._headers())
        session = response.headers.get(_SESSION_HEADER) or response.headers.get("Mcp-Session-Id")
        if session:
            self._session_id = session
        if response.status_code >= 400:
            raise MCPError(
                f"MCP HTTP {response.status_code}: {response.text[:200]}",
                context={"url": self.url, "status": response.status_code},
            )
        content_type = (response.headers.get("content-type") or "").lower()
        text = response.text
        if "text/event-stream" in content_type:
            return _parse_sse_jsonrpc(text, on_progress=self.on_progress)
        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            raise MCPError(
                f"invalid JSON-RPC HTTP body: {exc}",
                cause=exc,
                context={"body": text[:200]},
            ) from exc
        if not isinstance(data, dict):
            raise MCPError("MCP HTTP response is not a JSON object")
        return data

    async def _request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        self._next_id += 1
        message = await self._post(make_request(self._next_id, method, params))
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
            _log.warning("MCP notification %s failed: %s", method, response.status_code)

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
        return "".join(
            str(part.get("text") or "")
            for part in contents
            if isinstance(part, dict)
        )

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

        tool = Tool(_remote, name=name, description=str(info.get("description", "")))
        tool.spec.input_schema = info.get("inputSchema") or {"type": "object", "properties": {}}
        tool._args_model = create_model(
            f"mcp_http_{name}_args",
            __config__=ConfigDict(extra="allow"),
        )

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


def _parse_sse_jsonrpc(  # noqa: C901
    text: str, *, on_progress: Any | None = None
) -> dict[str, Any]:
    """Extract the last JSON-RPC response from an SSE body; forward progress."""
    last: dict[str, Any] | None = None
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict) or data.get("jsonrpc") != "2.0":
            continue
        if data.get("id") is None and data.get("method") == "notifications/progress":
            if on_progress is not None:
                try:
                    on_progress(data.get("params") or {})
                except Exception:
                    _log.exception("on_progress handler failed")
            continue
        if data.get("id") is not None:
            last = data
    if last is None:
        raise MCPError("no JSON-RPC message found in SSE body", context={"body": text[:200]})
    return last


def _aire_version() -> str:
    from aire._version import __version__

    return __version__

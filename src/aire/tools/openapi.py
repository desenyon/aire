"""Import OpenAPI / Swagger specs as aire Tools."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

from aire.core.errors import ConfigurationError, ToolError
from aire.tools.tool import Tool
from aire.tools.types import SideEffect


def load_openapi(source: str | Path | dict[str, Any]) -> dict[str, Any]:
    """Load an OpenAPI 3.x document from path, http(s) URL, or dict."""
    if isinstance(source, dict):
        return source
    text: str
    source_str = str(source)
    if source_str.startswith(("http://", "https://")):
        import httpx

        try:
            response = httpx.get(source_str, follow_redirects=True, timeout=30.0)
            response.raise_for_status()
            text = response.text
        except Exception as exc:
            raise ConfigurationError(
                f"failed to fetch OpenAPI URL: {source_str}",
                code="tools.openapi_fetch",
                context={"source": source_str, "error": str(exc)},
            ) from exc
        # Prefer Content-Type, else URL suffix / body sniff
        ctype = (response.headers.get("content-type") or "").lower()
        if "yaml" in ctype or source_str.endswith((".yaml", ".yml")):
            import yaml

            data = yaml.safe_load(text)
        else:
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                import yaml

                data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ConfigurationError("OpenAPI document must be an object", code="tools.openapi")
        return data
    path = Path(source_str)
    if path.is_file():
        text = path.read_text()
        if path.suffix in {".yaml", ".yml"}:
            import yaml

            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        if not isinstance(data, dict):
            raise ConfigurationError("OpenAPI document must be an object", code="tools.openapi")
        return data
    raise ConfigurationError(
        f"OpenAPI source not found: {source}",
        code="tools.openapi_missing",
        context={"source": str(source)},
    )


def openapi_to_tools(
    source: str | Path | dict[str, Any],
    *,
    base_url: str | None = None,
    prefix: str = "",
    include: list[str] | None = None,
    timeout_seconds: float = 30.0,
) -> list[Tool]:
    """Convert each OpenAPI operation into an aire :class:`Tool`.

    Tools perform HTTP calls via httpx. Only ``operationId`` (or method+path)
    named operations are imported. Use ``include`` to whitelist operationIds.
    """
    spec = load_openapi(source)
    servers = spec.get("servers") or []
    resolved_base = base_url or (servers[0].get("url") if servers else None)
    if not resolved_base:
        raise ConfigurationError(
            "OpenAPI import requires base_url= or servers[0].url",
            code="tools.openapi_base",
        )
    paths = spec.get("paths") or {}
    tools: list[Tool] = []
    for path, methods in paths.items():
        if not isinstance(methods, dict):
            continue
        for method, op in methods.items():
            if method.startswith("x-") or not isinstance(op, dict):
                continue
            if method.lower() not in {"get", "post", "put", "patch", "delete", "head"}:
                continue
            op_id = str(op.get("operationId") or f"{method}_{path}").replace("/", "_").replace(
                "{", ""
            ).replace("}", "")
            if include is not None and op_id not in include:
                continue
            name = f"{prefix}{op_id}" if prefix else op_id
            description = str(
                op.get("summary") or op.get("description") or f"{method.upper()} {path}"
            )
            tools.append(
                _make_http_tool(
                    name=name,
                    description=description,
                    method=method.upper(),
                    url=urljoin(resolved_base.rstrip("/") + "/", path.lstrip("/")),
                    timeout_seconds=timeout_seconds,
                )
            )
    return tools


def _make_http_tool(
    *,
    name: str,
    description: str,
    method: str,
    url: str,
    timeout_seconds: float,
) -> Tool:
    async def _call(body: dict[str, Any] | None = None, **params: Any) -> str:
        import httpx

        # Path param substitution
        final_url = url
        for key, value in list(params.items()):
            token = "{" + key + "}"
            if token in final_url:
                final_url = final_url.replace(token, str(value))
                params.pop(key, None)
        try:
            async with httpx.AsyncClient(timeout=timeout_seconds, follow_redirects=True) as client:
                response = await client.request(
                    method,
                    final_url,
                    params=params if method == "GET" else None,
                    json=body if method != "GET" else None,
                )
                response.raise_for_status()
                return response.text[:65_536]
        except Exception as exc:
            raise ToolError(f"openapi tool {name} failed: {exc}", cause=exc) from exc

    _call.__doc__ = description
    _call.__name__ = name
    return Tool(
        _call,
        name=name,
        description=description,
        side_effect=SideEffect.EXTERNAL_SIDE_EFFECT,
        timeout_seconds=timeout_seconds,
    )


def describe() -> dict[str, Any]:
    return {
        "kind": "openapi_tools",
        "factory": "aire.tools.openapi.openapi_to_tools",
        "supports": ["openapi-3", "json", "yaml"],
    }

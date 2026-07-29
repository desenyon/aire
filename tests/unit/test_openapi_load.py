"""OpenAPI load from dict."""

from __future__ import annotations

from aire.tools.openapi import load_openapi, openapi_to_tools


def test_load_openapi_from_dict() -> None:
    spec = {
        "openapi": "3.0.0",
        "info": {"title": "Demo", "version": "1.0.0"},
        "servers": [{"url": "https://example.com"}],
        "paths": {
            "/ping": {
                "get": {
                    "operationId": "ping",
                    "summary": "Ping",
                    "responses": {"200": {"description": "ok"}},
                }
            }
        },
    }
    loaded = load_openapi(spec)
    assert loaded is spec
    assert loaded["info"]["title"] == "Demo"
    tools = openapi_to_tools(spec, base_url="https://example.com")
    assert any(t.name == "ping" for t in tools)

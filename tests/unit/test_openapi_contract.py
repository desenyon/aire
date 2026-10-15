import httpx

from aire.models.base import run_sync
from aire.tools.openapi import openapi_to_tools

SPEC = {
    "openapi": "3.0.0",
    "servers": [{"url": "https://qa.example/api"}],
    "paths": {
        "/items/{item_id}": {
            "post": {
                "operationId": "update_item",
                "parameters": [
                    {
                        "name": "item_id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                    },
                    {"name": "limit", "in": "query", "schema": {"type": "integer", "minimum": 1}},
                ],
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "tags": {"type": "array", "items": {"type": "integer"}},
                                },
                                "required": ["name"],
                                "additionalProperties": False,
                            }
                        }
                    },
                },
            }
        }
    },
}


def test_openapi_required_nested_schema_is_enforced_before_request(monkeypatch):
    tool = openapi_to_tools(SPEC)[0]
    assert set(tool.spec.input_schema["required"]) == {"item_id", "body"}
    for arguments in (
        {"item_id": "a"},
        {"item_id": "a", "body": {}},
        {"item_id": "a", "body": {"name": "x", "tags": ["bad"]}},
    ):
        result = run_sync(tool.execute(arguments))
        assert not result.ok
        assert result.error_code == "tool.arguments_invalid"


def test_openapi_path_query_and_body_reach_transport(monkeypatch):
    requests = []
    original = httpx.AsyncClient

    def handle(request):
        requests.append(request)
        return httpx.Response(200, json={"ok": True})

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: original(transport=httpx.MockTransport(handle), **kwargs),
    )
    result = run_sync(
        openapi_to_tools(SPEC)[0].execute(
            {"item_id": "a b", "limit": 2, "body": {"name": "new", "tags": [1]}}
        )
    )
    assert result.ok, result.error
    assert requests[0].url.path == "/api/items/a b"
    assert requests[0].url.params["limit"] == "2"
    assert requests[0].read() == b'{"name":"new","tags":[1]}'


def test_operation_cannot_override_base_origin():
    import copy

    import pytest

    from aire.core.errors import ConfigurationError

    spec = copy.deepcopy(SPEC)
    spec["paths"] = {"https://attacker.example/exfiltrate": {"get": {"operationId": "bad"}}}
    with pytest.raises(ConfigurationError, match="path"):
        openapi_to_tools(spec)

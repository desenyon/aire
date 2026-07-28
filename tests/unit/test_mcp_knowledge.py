"""MCP knowledge tests: resources + prompts over the protocol (0.2.2)."""

from __future__ import annotations

import sys

import pytest

from aire.mcp.client import MCPClient
from aire.mcp.knowledge import builtin_prompts, builtin_resources, get_prompt, read_resource
from aire.mcp.server import MCPServer

# -- direct knowledge access -------------------------------------------------------


def test_builtin_resources_catalog() -> None:
    uris = {r.uri for r in builtin_resources()}
    assert uris == {"aire://guide", "aire://manifest", "aire://errors", "aire://refs"}


def test_read_guide_and_refs() -> None:
    guide = read_resource("aire://guide")
    assert "provider:name" in guide
    assert "AI.ml" in guide
    refs = read_resource("aire://refs")
    assert "simple:centroid" in refs
    errors = read_resource("aire://errors")
    assert "RateLimitError" in errors and "retryable=True" in errors


def test_read_manifest_is_live_json() -> None:
    import json

    manifest = json.loads(read_resource("aire://manifest"))
    assert manifest["library"] == "aire"
    assert "ml" in manifest["namespaces"]


def test_read_unknown_resource_raises() -> None:
    from aire.core.errors import NotFoundError

    with pytest.raises(NotFoundError):
        read_resource("aire://nope")


def test_prompt_rendering_with_arguments() -> None:
    result = get_prompt("aire_rag", {"docs": "./manuals"})
    text = result["messages"][0]["content"]["text"]
    assert "./manuals" in text
    assert "local:default" in text  # default filled in
    assert result["description"]


def test_prompt_unknown_placeholder_survives() -> None:
    result = get_prompt("aire_agent", {"model": "mock:default"})
    text = result["messages"][0]["content"]["text"]
    assert "mock:default" in text
    assert "{task}" in text  # missing argument left intact, no KeyError


def test_prompt_catalog_complete() -> None:
    names = {p.name for p in builtin_prompts()}
    assert names == {"aire_quickstart", "aire_rag", "aire_agent", "aire_gateway", "aire_ml"}


# -- server protocol ----------------------------------------------------------------


@pytest.mark.anyio
async def test_server_advertises_knowledge_capabilities() -> None:
    server = MCPServer([])
    response = await server.handle(
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
    )
    capabilities = response["result"]["capabilities"]
    assert capabilities["resources"] and capabilities["prompts"]


@pytest.mark.anyio
async def test_server_resources_list_and_read() -> None:
    server = MCPServer([])
    listing = await server.handle({"jsonrpc": "2.0", "id": 2, "method": "resources/list"})
    uris = {r["uri"] for r in listing["result"]["resources"]}
    assert "aire://guide" in uris

    read = await server.handle(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "resources/read",
            "params": {"uri": "aire://refs"},
        }
    )
    content = read["result"]["contents"][0]
    assert content["mimeType"] == "text/markdown"
    assert "qdrant" in content["text"]


@pytest.mark.anyio
async def test_server_prompts_list_and_get() -> None:
    server = MCPServer([])
    listing = await server.handle({"jsonrpc": "2.0", "id": 4, "method": "prompts/list"})
    names = {p["name"] for p in listing["result"]["prompts"]}
    assert "aire_ml" in names
    rag = next(p for p in listing["result"]["prompts"] if p["name"] == "aire_rag")
    assert rag["arguments"][0]["name"] == "docs"
    assert rag["arguments"][0]["required"] is True

    rendered = await server.handle(
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "prompts/get",
            "params": {"name": "aire_rag", "arguments": {"docs": "./docs"}},
        }
    )
    assert "./docs" in rendered["result"]["messages"][0]["content"]["text"]


@pytest.mark.anyio
async def test_server_knowledge_disabled() -> None:
    server = MCPServer([], knowledge=False)
    response = await server.handle(
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
    )
    assert "resources" not in response["result"]["capabilities"]
    listing = await server.handle({"jsonrpc": "2.0", "id": 2, "method": "resources/list"})
    assert listing["result"]["resources"] == []


# -- client round-trip -----------------------------------------------------------------


@pytest.mark.anyio
async def test_client_reads_knowledge_over_stdio() -> None:
    async with MCPClient([sys.executable, "-m", "aire.mcp"]) as client:
        resources = await client.list_resources()
        assert any(r["uri"] == "aire://guide" for r in resources)

        guide = await client.read_resource("aire://guide")
        assert "aire" in guide and "Model creation" in guide

        prompts = await client.list_prompts()
        assert any(p["name"] == "aire_gateway" for p in prompts)

        text = await client.get_prompt("aire_ml", {"estimator": "simple:knn"})
        assert "simple:knn" in text

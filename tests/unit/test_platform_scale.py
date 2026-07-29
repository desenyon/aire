"""Wave 3 platform scale: Redis workers, MCP HTTP, unknown worker kinds."""

from __future__ import annotations

import importlib.util

import pytest

from aire.core.errors import ConfigurationError
from aire.mcp.http_client import MCPHttpClient
from aire.workers import create_worker


def test_create_worker_redis() -> None:
    if importlib.util.find_spec("redis") is None:
        with pytest.raises(ConfigurationError, match="aire\\[redis\\]"):
            create_worker("redis")
    else:
        worker = create_worker("redis", url="redis://localhost:6379/0")
        desc = worker.describe()
        assert desc["kind"] == "redis_queue_worker"
        assert "LPUSH" in desc["transport"] or "redis" in desc["transport"]


def test_create_worker_sqs_not_bundled() -> None:
    with pytest.raises(ConfigurationError, match=r"boto3|not bundled"):
        create_worker("sqs")


def test_create_worker_unknown_kind() -> None:
    with pytest.raises(ConfigurationError, match="unknown worker kind"):
        create_worker("celery")


def test_mcp_http_client_describe() -> None:
    client = MCPHttpClient("http://localhost:8000/mcp")
    desc = client.describe()
    assert desc["kind"] == "mcp_http_client"
    assert desc["subset"] == "streamable HTTP transport subset"
    assert desc["url"] == "http://localhost:8000/mcp"
    assert desc["connected"] is False
    assert "tools/list" in desc["methods"]
    assert "resources/list" in desc["methods"]

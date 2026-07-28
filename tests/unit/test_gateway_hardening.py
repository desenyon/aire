"""Gateway hardening tests: circuit breakers, budgets, /v1/messages, request log."""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from aire.core.runtime import Runtime
from aire.deployment.gateway import create_gateway
from aire.models.registry import register_callable


@pytest.fixture()
def client(runtime: Runtime) -> TestClient:
    app = create_gateway(
        runtime,
        aliases={
            "echo": "mock:echo",
            "flaky": ["callable:boom", "mock:echo"],
        },
        failure_threshold=2,
    )
    return TestClient(app)


@pytest.fixture(autouse=True)
def _boom_model() -> None:
    def boom(prompt: str) -> str:
        raise RuntimeError("upstream exploded")

    register_callable("boom", boom)


def _chat(client: TestClient, model: str, message: str = "hi") -> Any:
    return client.post(
        "/v1/chat/completions",
        json={"model": model, "messages": [{"role": "user", "content": message}]},
    )


def test_circuit_breaker_opens_and_falls_back(client: TestClient) -> None:
    # First two calls: callable:boom fails, circuit opens after threshold=2;
    # every call still succeeds via the mock:echo fallback.
    for _ in range(3):
        response = _chat(client, "flaky")
        assert response.status_code == 200
        assert response.headers["X-Aire-Resolved-Model"] == "mock:echo"

    manifest = client.get("/v1/gateway/manifest").json()
    circuits = manifest["circuit_breaker"]["circuits"]
    assert circuits["callable:boom"]["open"] is True
    assert circuits["callable:boom"]["failures"] >= 2


def test_circuit_breaker_429_when_all_candidates_open(runtime: Runtime) -> None:
    app = create_gateway(runtime, aliases={"only-broken": "callable:boom"}, failure_threshold=1)
    client = TestClient(app)
    assert _chat(client, "only-broken").status_code == 502  # real failure, first time
    response = _chat(client, "only-broken")
    assert response.status_code == 429  # circuit open: skipped before even trying
    assert "unavailable" in response.json()["error"]["message"]


def test_daily_budget_exhaustion(runtime: Runtime) -> None:
    app = create_gateway(runtime, aliases={"echo": "mock:echo"}, budgets={"echo": 0.0})
    client = TestClient(app)
    assert _chat(client, "echo").status_code == 200  # spend 0.0 recorded
    response = _chat(client, "echo")  # spent (0.0) >= budget (0.0) → over
    assert response.status_code == 429
    manifest = client.get("/v1/gateway/manifest").json()
    assert manifest["budgets"] == {"echo": 0.0}
    assert "echo" in manifest["spend_today"]


def test_anthropic_messages_endpoint(client: TestClient) -> None:
    response = client.post(
        "/v1/messages",
        json={
            "model": "echo",
            "system": "You are terse.",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
            "max_tokens": 32,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["type"] == "message"
    assert body["role"] == "assistant"
    assert body["content"][0]["type"] == "text"
    assert "hello" in body["content"][0]["text"]  # echo model reflects the last message
    assert body["stop_reason"] == "end_turn"
    assert body["usage"]["input_tokens"] > 0
    assert body["model"] == "echo"
    assert body["aire"]["resolved_model"] == "mock:echo"


def test_request_log_jsonl(runtime: Runtime, tmp_path) -> None:
    log_path = tmp_path / "requests.jsonl"
    app = create_gateway(runtime, aliases={"echo": "mock:echo"}, request_log=str(log_path))
    client = TestClient(app)
    assert _chat(client, "echo").status_code == 200
    entries = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["endpoint"] == "chat.completions"
    assert entry["resolved"] == "mock:echo"
    assert entry["latency_ms"] >= 0
    assert "sk-" not in json.dumps(entry)  # nothing sensitive logged


def test_gateway_config_new_fields(runtime: Runtime) -> None:
    from aire.core.config import GatewayConfig, Settings

    settings = Settings(
        gateway=GatewayConfig(
            models=["mock:echo"],
            budgets={"echo": 1.5},
            circuit_breaker=False,
            request_log="logs/gateway.jsonl",
        )
    )
    assert settings.gateway.budgets == {"echo": 1.5}
    assert settings.gateway.circuit_breaker is False
    assert settings.gateway.failure_threshold == 3
    assert settings.gateway.cooldown_seconds == 30.0

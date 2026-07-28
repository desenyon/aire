"""Model gateway: OpenAI-compatible serving over any provider refs."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from aire.ai import _GatewayNamespace
from aire.cli.main import parse_alias_options
from aire.core.config import Settings
from aire.core.runtime import Runtime
from aire.deployment.gateway import Gateway, create_gateway
from aire.models.registry import register_callable


@pytest.fixture()
def client(runtime: Runtime) -> TestClient:
    app = create_gateway(runtime, models=["mock:echo"], embeddings={"emb": "local:hashing"})
    return TestClient(app)


def _chat(client: TestClient, model: str, text: str = "hi", **extra: object) -> dict[str, object]:
    response = client.post(
        "/v1/chat/completions",
        json={"model": model, "messages": [{"role": "user", "content": text}], **extra},
    )
    assert response.status_code == 200, response.text
    return dict(response.json())


def test_models_listing(client: TestClient) -> None:
    body = client.get("/v1/models").json()
    ids = {entry["id"] for entry in body["data"]}
    assert {"mock:echo", "emb"} <= ids
    card = next(e for e in body["data"] if e["id"] == "emb")
    assert card["aire_kind"] == "embedding"


def test_chat_completion_openai_shape(client: TestClient) -> None:
    body = _chat(client, "mock:echo", "hello gateway")
    choice = body["choices"][0]
    assert body["object"] == "chat.completion"
    assert body["model"] == "mock:echo"
    assert choice["message"]["role"] == "assistant"
    assert choice["message"]["content"] == "hello gateway"
    assert choice["finish_reason"] == "stop"
    usage = body["usage"]
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    assert body["aire"]["resolved_model"] == "mock:echo"


def test_chat_completion_resolved_header(client: TestClient) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={"model": "mock:echo", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.headers["x-aire-resolved-model"] == "mock:echo"


def test_unknown_model_returns_openai_error(client: TestClient) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={"model": "does-not-exist", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 404
    error = response.json()["error"]
    assert error["type"] == "not_found_error"
    assert "code" in error and "message" in error


def test_alias_with_fallback_chain(runtime: Runtime) -> None:
    app = create_gateway(runtime, aliases={"robust": ["unknown_provider:x", "mock:echo"]})
    client = TestClient(app)
    body = _chat(client, "robust")
    assert body["aire"]["resolved_model"] == "mock:echo"
    assert body["choices"][0]["message"]["content"] == "hi"


def test_fallback_on_provider_failure(runtime: Runtime) -> None:
    def _failing_factory(name: str, *, runtime: Runtime, **options: object) -> object:
        from aire.models.base import Model

        class _Broken(Model):
            @property
            def info(self):  # type: ignore[no-untyped-def]
                from aire.core.types import Capability
                from aire.models.types import ModelInfo

                return ModelInfo(
                    ref="broken:x", provider="broken", capabilities=[Capability.TEXT_GENERATION]
                )

            async def generate(self, request):  # type: ignore[no-untyped-def]
                from aire.core.errors import ProviderError

                raise ProviderError("broken", "upstream exploded", status=503)

        return _Broken()

    runtime.model_providers.register("broken", _failing_factory)
    app = create_gateway(runtime, aliases={"safe": ["broken:x", "mock:echo"]})
    body = _chat(TestClient(app), "safe")
    assert body["aire"]["resolved_model"] == "mock:echo"


def test_round_robin_alternates(runtime: Runtime) -> None:
    register_callable("upper_gw", lambda prompt: prompt.upper())
    app = create_gateway(
        runtime,
        aliases={"rr": ["mock:echo", "callable:upper_gw"]},
        routing="round_robin",
    )
    client = TestClient(app)
    first = _chat(client, "rr")["aire"]["resolved_model"]
    second = _chat(client, "rr")["aire"]["resolved_model"]
    assert first != second


def test_objective_routing_picks_quality(runtime: Runtime) -> None:
    register_callable("upper_obj", lambda prompt: prompt.upper())
    app = create_gateway(
        runtime,
        aliases={"smart": ["mock:echo", "callable:upper_obj"]},
        objective="highest_quality",
    )
    body = _chat(TestClient(app), "smart", "hi")
    assert body["choices"][0]["message"]["content"] == "HI"
    assert body["aire"]["resolved_model"].startswith("router:")


def test_streaming_chat_completion(client: TestClient) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "mock:echo",
            "messages": [{"role": "user", "content": "stream me"}],
            "stream": True,
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = [line for line in response.text.splitlines() if line.startswith("data:")]
    assert events[-1] == "data: [DONE]"
    payloads = [json.loads(line[len("data:") :]) for line in events[:-1]]
    chunks = [p for p in payloads if p.get("object") == "chat.completion.chunk"]
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert text == "stream me"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


def test_streaming_unknown_model_emits_error_event(client: TestClient) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={"model": "nope", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    assert response.status_code == 200
    assert '"error"' in response.text
    assert response.text.rstrip().endswith("data: [DONE]")


def test_embeddings_endpoint(client: TestClient) -> None:
    response = client.post("/v1/embeddings", json={"model": "emb", "input": ["hello", "world"]})
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert len(body["data"]) == 2
    assert len(body["data"][0]["embedding"]) == 256
    assert body["aire"]["resolved_model"] == "local:hashing"


def test_embeddings_direct_ref(client: TestClient) -> None:
    response = client.post("/v1/embeddings", json={"model": "local:hashing", "input": "hi"})
    assert response.status_code == 200
    assert len(response.json()["data"][0]["embedding"]) == 256


def test_auth_required(runtime: Runtime) -> None:
    app = create_gateway(runtime, models=["mock:echo"], auth_token="sk-secret")
    client = TestClient(app)
    assert client.get("/v1/models").status_code == 401
    assert client.get("/health").status_code == 200  # health stays open
    ok = client.get("/v1/models", headers={"Authorization": "Bearer sk-secret"})
    assert ok.status_code == 200


def test_rate_limit(runtime: Runtime) -> None:
    app = create_gateway(runtime, models=["mock:echo"], rate_limit_per_minute=2)
    client = TestClient(app)
    codes = [client.get("/v1/models").status_code for _ in range(3)]
    assert codes[:2] == [200, 200]
    assert codes[2] == 429


def test_manifest(runtime: Runtime) -> None:
    app = create_gateway(
        runtime,
        aliases={"cheap": "mock:echo"},
        embeddings={"emb": "local:hashing"},
        routing="round_robin",
    )
    manifest = TestClient(app).get("/v1/gateway/manifest").json()
    assert manifest["kind"] == "gateway"
    assert manifest["routing"] == "round_robin"
    assert manifest["chat_models"] == {"cheap": ["mock:echo"]}
    assert manifest["embedding_models"] == {"emb": ["local:hashing"]}


def test_gateway_class_rejects_bad_routing(runtime: Runtime) -> None:
    from aire.core.errors import ConfigurationError

    with pytest.raises(ConfigurationError):
        Gateway(runtime, routing="bogus")  # type: ignore[arg-type]


def test_gateway_describe(runtime: Runtime) -> None:
    gateway = Gateway(runtime, chat_routes={"a": ["mock:echo"]})
    manifest = gateway.describe()
    assert manifest["chat_models"] == {"a": ["mock:echo"]}
    assert "aire_version" in manifest


def test_settings_driven_gateway() -> None:
    settings = Settings(
        project="test-project",
        gateway={"aliases": {"cfg": "mock:echo"}, "embeddings": {"emb": "local:hashing"}},
    )
    runtime = Runtime(settings)
    app = _GatewayNamespace(runtime).create()
    client = TestClient(app)
    body = _chat(client, "cfg")
    assert body["choices"][0]["message"]["content"] == "hi"
    assert client.post("/v1/embeddings", json={"model": "emb", "input": "x"}).status_code == 200


def test_parse_alias_options() -> None:
    parsed = parse_alias_options(["smart=a:b,c:d", "cheap=mock:echo"])
    assert parsed == {"smart": ["a:b", "c:d"], "cheap": "mock:echo"}


def test_parse_alias_options_rejects_garbage() -> None:
    import typer

    with pytest.raises(typer.Exit):
        parse_alias_options(["no-equals-sign"])

"""OpenAI-compatible provider contract against a mocked HTTP transport."""

from __future__ import annotations

import json

import httpx
import pytest

from aire.core.config import Settings
from aire.core.errors import AuthenticationError, RateLimitError
from aire.core.runtime import Runtime
from aire.integrations.http import ProviderHttpClient
from aire.integrations.openai import OpenAIEmbedder, OpenAIModel
from aire.models.types import GenerationRequest, ToolDefinition
from tests.conftest import arun


def _client(runtime: Runtime, handler: httpx.MockTransport) -> ProviderHttpClient:
    client = ProviderHttpClient(
        runtime,
        "openai",
        base_url="https://api.openai.com/v1",
        headers={"Authorization": "Bearer test-key"},
    )
    client._client = httpx.AsyncClient(
        base_url="https://api.openai.com/v1",
        headers={"Authorization": "Bearer test-key"},
        transport=handler,
    )
    return client


def _chat_payload(text: str, tool_calls: list[dict] | None = None) -> dict:
    message: dict = {"role": "assistant", "content": text}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-1",
        "choices": [{"message": message, "finish_reason": "tool_calls" if tool_calls else "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


@pytest.fixture()
def runtime() -> Runtime:
    return Runtime(Settings(project="test"))


def test_generate_maps_response(runtime: Runtime) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["model"] == "gpt-4o-mini"
        return httpx.Response(200, json=_chat_payload("hi there"))

    model = OpenAIModel("gpt-4o-mini", _client(runtime, httpx.MockTransport(handler)))
    result = arun(model.generate(GenerationRequest.of("hello")))
    assert result.text == "hi there"
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 5
    assert result.usage.cost_usd > 0  # priced model
    assert result.model == "openai:gpt-4o-mini"


def test_tool_calls_mapped(runtime: Runtime) -> None:
    calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "search", "arguments": '{"q": "aire"}'},
        }
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["tools"][0]["function"]["name"] == "search"
        return httpx.Response(200, json=_chat_payload("", calls))

    model = OpenAIModel("gpt-4o-mini", _client(runtime, httpx.MockTransport(handler)))
    request = GenerationRequest.of("find", tools=[ToolDefinition(name="search")])
    result = arun(model.generate(request))
    assert result.finish_reason == "tool_calls"
    assert result.tool_calls[0].name == "search"
    assert result.tool_calls[0].arguments == {"q": "aire"}


def test_streaming_chunks(runtime: Runtime) -> None:
    sse = (
        'data: {"choices": [{"delta": {"content": "Hel"}}]}\n\n'
        'data: {"choices": [{"delta": {"content": "lo"}}]}\n\n'
        'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}\n\n'
        "data: [DONE]\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})

    async def _collect() -> list[str]:
        model = OpenAIModel("gpt-4o-mini", _client(runtime, httpx.MockTransport(handler)))
        return [c.text async for c in model.stream(GenerationRequest.of("hi"))]

    assert arun(_collect()) == ["Hel", "lo", ""]


def test_auth_error_mapping(runtime: Runtime) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "invalid api key"}})

    model = OpenAIModel("gpt-4o-mini", _client(runtime, httpx.MockTransport(handler)))
    with pytest.raises(AuthenticationError) as excinfo:
        arun(model.generate(GenerationRequest.of("hi")))
    assert excinfo.value.status == 401


def test_rate_limit_retried(runtime: Runtime) -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, json={"error": {"message": "slow down"}})
        return httpx.Response(200, json=_chat_payload("recovered"))

    model = OpenAIModel("gpt-4o-mini", _client(runtime, httpx.MockTransport(handler)))
    import aire.models.retry as retry_module

    original = retry_module.asyncio.sleep

    async def _no_sleep(delay: float) -> None:
        return None

    retry_module.asyncio.sleep = _no_sleep  # type: ignore[attr-defined]
    try:
        result = arun(model.generate(GenerationRequest.of("hi")))
    finally:
        retry_module.asyncio.sleep = original  # type: ignore[attr-defined]
    assert result.text == "recovered"
    assert attempts == 2


def test_rate_limit_exhaustion(runtime: Runtime) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "always throttled"}})

    import aire.models.retry as retry_module

    async def _no_sleep(delay: float) -> None:
        return None

    original = retry_module.asyncio.sleep
    retry_module.asyncio.sleep = _no_sleep  # type: ignore[attr-defined]
    try:
        model = OpenAIModel("gpt-4o-mini", _client(runtime, httpx.MockTransport(handler)))
        with pytest.raises(RateLimitError):
            arun(model.generate(GenerationRequest.of("hi")))
    finally:
        retry_module.asyncio.sleep = original  # type: ignore[attr-defined]


def test_embeddings(runtime: Runtime) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 0, "embedding": [0.1, 0.2]},
                    {"index": 1, "embedding": [0.3, 0.4]},
                ],
                "usage": {"prompt_tokens": 4},
            },
        )

    embedder = OpenAIEmbedder(
        "text-embedding-3-small", _client(runtime, httpx.MockTransport(handler))
    )
    result = arun(embedder.embed_texts(["a", "b"]))
    assert result == [[0.1, 0.2], [0.3, 0.4]]


def test_structured_output_payload(runtime: Runtime) -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.update(body.get("response_format", {}))
        return httpx.Response(200, json=_chat_payload('{"answer": "yes"}'))

    from aire.models.types import StructuredOutputSpec

    model = OpenAIModel("gpt-4o-mini", _client(runtime, httpx.MockTransport(handler)))
    request = GenerationRequest.of(
        "q", response_format=StructuredOutputSpec(schema={"type": "object"})
    )
    result = arun(model.generate(request))
    assert seen["type"] == "json_schema"
    assert result.text == '{"answer": "yes"}'

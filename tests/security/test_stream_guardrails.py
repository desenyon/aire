from collections.abc import AsyncIterator

import pytest
from fastapi.testclient import TestClient

from aire.core.config import Settings
from aire.core.errors import ProviderError
from aire.core.runtime import Runtime
from aire.deployment.gateway import create_gateway
from aire.models.base import run_sync
from aire.models.builtin import EchoModel
from aire.models.types import GenerationChunk, GenerationRequest
from aire.safety.guardrails import GuardrailVerdict


class OutputRail:
    name = "output-test"

    def check(self, text):
        return GuardrailVerdict(passed="BLOCKME" not in text, guardrail=self.name)


class SplitModel(EchoModel):
    fail = False
    closed = False

    async def stream(self, request: GenerationRequest) -> AsyncIterator[GenerationChunk]:
        try:
            yield GenerationChunk(text="BLOCK")
            if self.fail:
                raise ProviderError("test", "stream interrupted")
            yield GenerationChunk(text="ME", finish_reason="stop")
        finally:
            self.closed = True


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/messages"])
@pytest.mark.parametrize("fail", [False, True])
def test_stream_releases_no_unchecked_output(path, fail):
    runtime = Runtime(Settings())
    model = SplitModel()
    model.fail = fail
    runtime.model_providers.register("test", lambda name, **kw: model)
    app = create_gateway(runtime, models=["test:split"], guardrails=[OutputRail()])
    with TestClient(app) as client:
        response = client.post(
            path,
            json={
                "model": "test:split",
                "stream": True,
                "max_tokens": 20,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert "BLOCK" not in response.text
        assert "error" in response.text
    assert model.closed
    run_sync(runtime.aclose())


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/messages"])
def test_stream_redacts_email_spanning_chunks(path):
    from aire.safety.guardrails import PIIGuardrail

    class EmailModel(EchoModel):
        async def stream(self, request):
            yield GenerationChunk(text="person@")
            yield GenerationChunk(text="example.com", finish_reason="stop")

    runtime = Runtime(Settings())
    runtime.model_providers.register("test", lambda name, **kw: EmailModel())
    with TestClient(
        create_gateway(runtime, models=["test:email"], guardrails=[PIIGuardrail(action="redact")])
    ) as client:
        response = client.post(
            path,
            json={
                "model": "test:email",
                "stream": True,
                "max_tokens": 20,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert "person@" not in response.text
        assert "[PII]" in response.text
    run_sync(runtime.aclose())


def test_input_redaction_preserves_tool_declarations_and_media():
    from aire.core.content import ImageContent, Message, TextContent
    from aire.deployment.gateway import _guard_generation_request
    from aire.models.types import ToolCall
    from aire.safety.guardrails import GuardrailChain, PIIGuardrail

    async def check():
        call = ToolCall(id="call", name="lookup", arguments={"q": "x"})
        image = ImageContent(uri="https://example.com/a.png")
        message = Message(
            role="assistant",
            content=[TextContent(text="person@example.com"), image],
            tool_calls=[call],
        )
        guarded = await _guard_generation_request(
            GenerationRequest(messages=[message]), GuardrailChain([PIIGuardrail(action="redact")])
        )
        assert guarded.messages[0].tool_calls == [call]
        assert image in guarded.messages[0].content
        assert "person@example.com" not in guarded.messages[0].text_content

    run_sync(check())


@pytest.mark.parametrize(
    "mode",
    ["gateway", "checked", "openai", "anthropic", "exact-cache", "semantic-cache", "redis-cache"],
)
def test_closing_consumer_immediately_closes_upstream(mode):
    from aire.deployment.gateway import Gateway, _anthropic_sse_stream, _checked_stream, _sse_stream
    from aire.models.builtin import HashingEmbedder
    from aire.optimization.cache import CachedModel, SemanticCachedModel

    async def check():
        runtime = Runtime(Settings())
        model = SplitModel()
        runtime.model_providers.register("test", lambda name, **kw: model)
        gateway = Gateway(runtime, chat_routes={"qa": ["test:split"]})
        request = GenerationRequest.of("hello")
        from aire.optimization.redis_cache import RedisCacheBackend, RedisCachedModel

        class Backend(RedisCacheBackend):
            def __init__(self):
                pass

            async def aget(self, key):
                return None

            async def aset(self, *args, **kwargs):
                raise AssertionError("partial output must not be cached")

        factories = {
            "gateway": lambda: gateway.stream("qa", request),
            "checked": lambda: _checked_stream(gateway, "qa", request, None),
            "openai": lambda: _sse_stream(gateway, "qa", request, None, None),
            "anthropic": lambda: _anthropic_sse_stream(gateway, "qa", request, None, None),
            "exact-cache": lambda: CachedModel(model).stream(request),
            "semantic-cache": lambda: SemanticCachedModel(model, HashingEmbedder()).stream(request),
            "redis-cache": lambda: RedisCachedModel(model, Backend()).stream(request),
        }
        stream = factories[mode]()
        try:
            async for value in stream:
                if "BLOCK" in str(value):
                    break
            await stream.aclose()
            assert model.closed
        finally:
            await stream.aclose()
            await runtime.aclose()

    run_sync(check())


def test_gateway_stream_bills_cumulative_usage_once():
    from aire.core.types import Usage
    from aire.deployment.gateway import Gateway

    class Billed(EchoModel):
        async def stream(self, request):
            yield GenerationChunk(text="a", usage=Usage(cost_usd=0.1))
            yield GenerationChunk(text="b", usage=Usage(cost_usd=0.2), finish_reason="stop")

    async def check():
        async with Runtime(Settings()) as runtime:
            runtime.model_providers.register("test", lambda name, **kw: Billed())
            gateway = Gateway(runtime, chat_routes={"qa": ["test:billed"]})
            _ = [part async for part in gateway.stream("qa", GenerationRequest.of("hello"))]
            assert gateway.spend_today()["qa"] == 0.2

    run_sync(check())

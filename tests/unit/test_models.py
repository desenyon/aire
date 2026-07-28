"""Model registry, builtin providers, request/response types, retries."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from aire.core.errors import NotFoundError, OutputValidationError
from aire.core.runtime import Runtime
from aire.core.types import Capability
from aire.models.builtin import CallableModel, EchoModel, HashingEmbedder
from aire.models.registry import ModelRegistry, register_callable
from aire.models.retry import with_retry
from aire.models.types import GenerationRequest, StructuredOutputSpec, ToolCall, ToolDefinition
from tests.conftest import arun


def test_echo_model_generate() -> None:
    model = EchoModel()
    result = arun(model.generate(GenerationRequest.of("hello world")))
    assert result.text == "hello world"
    assert result.model == "mock:echo"
    assert result.usage.input_tokens > 0


def test_echo_model_stream() -> None:
    async def _collect() -> list[str]:
        return [c.text async for c in EchoModel().stream(GenerationRequest.of("abc"))]

    assert arun(_collect()) == ["abc"]


def test_echo_structured_output() -> None:
    class Shape(BaseModel):
        answer: str
        score: int

    result = arun(EchoModel().ask_structured("give json", Shape))
    assert isinstance(result, Shape)
    assert result.answer == "mock-value"


def test_structured_output_retries_and_raises() -> None:
    class Shape(BaseModel):
        value: int

    class BadModel(EchoModel):
        async def generate(self, request: GenerationRequest):  # type: ignore[override]
            from aire.models.types import GenerationResult

            return GenerationResult.text_result("not json", model="mock:bad")

    with pytest.raises(OutputValidationError):
        arun(BadModel().ask_structured("x", Shape, retries=1))


def test_registry_resolves_mock(runtime: Runtime) -> None:
    registry = ModelRegistry(runtime)
    model = arun(registry.use("mock:echo"))
    assert model.info.provider == "mock"
    again = arun(registry.use("mock:echo"))
    assert again is model  # cached


def test_registry_unknown_provider(runtime: Runtime) -> None:
    registry = ModelRegistry(runtime)
    with pytest.raises(NotFoundError):
        arun(registry.use("nosuchprovider:x"))


def test_registry_invalid_spec(runtime: Runtime) -> None:
    registry = ModelRegistry(runtime)
    with pytest.raises(Exception) as excinfo:
        arun(registry.use("broken"))
    assert getattr(excinfo.value, "code", "") == "ref.invalid"


def test_callable_model(runtime: Runtime) -> None:
    register_callable("upper", lambda prompt: prompt.upper())
    model = arun(ModelRegistry(runtime).use("callable:upper"))
    assert isinstance(model, CallableModel)
    assert arun(model.ask("shout")) == "SHOUT"


def test_hashing_embedder_deterministic() -> None:
    embedder = HashingEmbedder(dimension=64)
    v1 = arun(embedder.embed_one("refund policy"))
    v2 = arun(embedder.embed_one("refund policy"))
    v3 = arun(embedder.embed_one("completely different topic"))
    assert v1 == v2
    assert v1 != v3
    assert len(v1) == 64
    norm = sum(x * x for x in v1) ** 0.5
    assert abs(norm - 1.0) < 1e-6


def test_tool_call_from_json() -> None:
    call = ToolCall.from_json("1", "search", '{"q": "x"}')
    assert call.arguments == {"q": "x"}
    broken = ToolCall.from_json("2", "search", "{invalid")
    assert "_raw" in broken.arguments


def test_model_info_capabilities() -> None:
    info = EchoModel().info
    assert info.supports(Capability.STREAMING)
    assert not info.supports(Capability.IMAGE_GENERATION)


def test_request_builder() -> None:
    request = GenerationRequest.of(
        "hi",
        tools=[ToolDefinition(name="t")],
        response_format=StructuredOutputSpec(schema={"type": "object"}),
    )
    assert request.messages[0].text_content == "hi"
    assert request.tools and request.tools[0].name == "t"
    assert request.response_format and request.response_format.json_schema["type"] == "object"


def test_with_retry_retries_transient() -> None:
    from aire.core.errors import RateLimitError

    attempts = 0

    async def flaky() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RateLimitError("p", "throttled")
        return "ok"

    assert arun(with_retry(flaky, base_delay=0.001)) == "ok"
    assert attempts == 3


def test_with_retry_does_not_retry_permanent() -> None:
    from aire.core.errors import AuthenticationError

    async def broken() -> str:
        raise AuthenticationError("p", "bad key")

    with pytest.raises(AuthenticationError):
        arun(with_retry(broken, base_delay=0.001))

"""Provider protocol regressions using recorded-shape synthetic events only."""

import json

import pytest

from aire.core.content import Message
from aire.core.errors import ProviderError
from aire.core.types import Usage
from aire.deployment.gateway import _build_anthropic_request, _build_generation_request
from aire.integrations.anthropic import AnthropicModel
from aire.integrations.ollama import OllamaModel
from aire.integrations.openai import OpenAIModel
from aire.models.base import run_sync
from aire.models.types import GenerationRequest, GenerationResult, ToolCall
from aire.optimization.cache import CachedModel


class EventClient:
    def __init__(self, events):
        self.events = events

    async def stream_sse(self, path, payload):
        for event in self.events:
            yield event


async def collect(model):
    return [c async for c in model.stream(GenerationRequest.of("test"))]


def openai_event(*calls, finish=None, text=""):
    return {
        "choices": [
            {"delta": {"tool_calls": list(calls), "content": text}, "finish_reason": finish}
        ]
    }


def fragment(index, *, id=None, name=None, arguments=""):
    return {"index": index, "id": id, "function": {"name": name, "arguments": arguments}}


def test_openai_interleaved_arguments_and_usage_only_event():
    events = [
        openai_event(
            fragment(0, id="a", name="lookup", arguments='{"query":'),
            fragment(1, id="b", name="add", arguments='{"x":'),
        ),
        openai_event(fragment(1, arguments="3}"), fragment(0, arguments='"hello"}')),
        openai_event(finish="tool_calls"),
        {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 4}},
    ]
    chunks = run_sync(collect(OpenAIModel("test", EventClient(events))))
    assert not chunks[0].tool_calls and not chunks[1].tool_calls
    assert chunks[2].tool_calls == [
        ToolCall(id="a", name="lookup", arguments={"query": "hello"}),
        ToolCall(id="b", name="add", arguments={"x": 3}),
    ]
    assert chunks[2].finish_reason == "tool_calls"
    assert chunks[3].usage == Usage(input_tokens=7, output_tokens=4)


@pytest.mark.parametrize("arguments", ['{"x":', "[1]", "null"])
def test_invalid_stream_arguments_are_not_executable(arguments):
    events = [
        openai_event(fragment(0, id="a", name="lookup", arguments=arguments)),
        openai_event(finish="tool_calls"),
    ]
    with pytest.raises(ProviderError, match="invalid streamed tool call"):
        run_sync(collect(OpenAIModel("test", EventClient(events))))


def test_interrupted_tool_stream_is_an_error():
    events = [openai_event(fragment(0, id="a", name="lookup", arguments='{"x":1}'))]
    with pytest.raises(ProviderError, match="stream ended"):
        run_sync(collect(OpenAIModel("test", EventClient(events))))


def test_anthropic_stream_assembles_tool_and_reports_finish_usage():
    events = [
        {"type": "message_start", "message": {"usage": {"input_tokens": 8, "output_tokens": 1}}},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "tool_use", "id": "a", "name": "lookup", "input": {}},
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": '{"q":'},
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": '"hi"}'},
        },
        {"type": "content_block_stop", "index": 1},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use"},
            "usage": {"output_tokens": 6},
        },
        {"type": "message_stop"},
    ]
    chunks = run_sync(collect(AnthropicModel("test", EventClient(events))))
    assert chunks[0].tool_calls == [ToolCall(id="a", name="lookup", arguments={"q": "hi"})]
    assert chunks[1].finish_reason == "tool_calls"
    assert chunks[1].usage == Usage(input_tokens=8, output_tokens=6)


def test_tool_history_roundtrips_through_gateway_and_providers():
    call = ToolCall(id="c", name="lookup", arguments={"x": 1})
    message = GenerationResult(content=[], tool_calls=[call]).message
    request = GenerationRequest(messages=[message, Message(role="tool", tool_call_id="c")])
    openai = OpenAIModel("test", EventClient([]))._request_payload(request, stream=False)
    assert json.loads(openai["messages"][0]["tool_calls"][0]["function"]["arguments"]) == {"x": 1}
    assert _build_generation_request(openai).messages[0].tool_calls == [call]
    anthropic = AnthropicModel("test", EventClient([]))._payload(request, stream=False)
    assert anthropic["messages"][0]["content"][0]["type"] == "tool_use"
    normalized = _build_anthropic_request(anthropic)
    assert normalized.messages[0].tool_calls == [call]
    # Legacy structured blocks and normalized calls must not be sent twice.
    replay = AnthropicModel("test", EventClient([]))._payload(normalized, stream=False)
    assert len(replay["messages"][0]["content"]) == 1
    ollama = OllamaModel("test", EventClient([]))._payload(request)
    assert ollama["messages"][0]["tool_calls"][0]["function"] == {
        "name": "lookup",
        "arguments": {"x": 1},
    }


def test_exact_stream_cache_retains_tool_calls_and_usage():
    events = [openai_event(fragment(0, id="a", name="lookup", arguments="{}"), finish="tool_calls")]
    cached = CachedModel(OpenAIModel("test", EventClient(events)))
    first = run_sync(collect(cached))
    second = run_sync(collect(cached))
    assert second[0].tool_calls == first[0].tool_calls
    assert second[0].finish_reason == "tool_calls"


def test_tool_context_bypasses_semantic_cache():
    from aire.models.builtin import EchoModel, HashingEmbedder
    from aire.models.types import ToolDefinition
    from aire.optimization.cache import SemanticCachedModel

    class CountingEcho(EchoModel):
        count = 0

        async def generate(self, request):
            self.count += 1
            return await super().generate(request)

    model = CountingEcho()
    cache = SemanticCachedModel(model, HashingEmbedder())
    request = GenerationRequest.of("same", tools=[ToolDefinition(name="lookup")])
    run_sync(cache.generate(request))
    run_sync(cache.generate(request))
    assert model.count == 2
    assert cache.stats()["entries"] == 0


def test_default_model_stream_preserves_tool_metadata():
    from aire.models.builtin import EchoModel

    model = EchoModel()
    call = ToolCall(id="a", name="lookup")
    model.scripted_tool_calls = [call]
    chunks = run_sync(collect(model))
    assert chunks[0].tool_calls == [call]
    assert chunks[0].usage.output_tokens == 8
    assert chunks[0].finish_reason == "tool_calls"

import base64

import pytest
from fastapi.testclient import TestClient

from aire.deployment.gateway import _build_generation_request, create_gateway


def test_gateway_preserves_images_audio_and_cache_namespace():
    request = _build_generation_request(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe"},
                        {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": base64.b64encode(b"audio").decode(),
                                "format": "wav",
                            },
                        },
                    ],
                }
            ],
            "metadata": {"tenant": "a"},
        }
    )
    assert [c.kind for c in request.messages[0].content] == ["text", "image", "audio"]
    assert request.messages[0].content[2].data == b"audio"
    assert request.metadata == {"tenant": "a"}


@pytest.mark.parametrize(
    "messages",
    [
        [],
        ["bad"],
        [{"role": "invalid"}],
        [{"role": "user", "content": [{"type": "unknown"}]}],
        [{"role": "assistant", "tool_calls": [{}]}],
    ],
)
def test_malformed_gateway_payload_is_client_error(messages):
    with TestClient(create_gateway(guardrails=False)) as client:
        response = client.post(
            "/v1/chat/completions", json={"model": "mock:echo", "messages": messages}
        )
        assert response.status_code == 400
        assert "error" in response.json()


@pytest.mark.parametrize(
    "extra", [{"tools": "bad"}, {"tools": [{}]}, {"response_format": {"type": "unknown"}}]
)
def test_invalid_gateway_options_are_not_silently_ignored(extra):
    with TestClient(create_gateway(guardrails=False)) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "mock:echo",
                "messages": [{"role": "user", "content": "hello"}],
                **extra,
            },
        )
        assert response.status_code == 400


def test_gateway_preserves_named_tool_choice():
    request = _build_generation_request(
        {
            "messages": [{"role": "user", "content": "hello"}],
            "tool_choice": {"type": "function", "function": {"name": "lookup"}},
        }
    )
    assert request.tool_choice == "lookup"


def test_openai_provider_encodes_named_tool_choice():
    from aire.integrations.openai import OpenAIModel
    from aire.models.types import GenerationRequest, ToolDefinition

    model = object.__new__(OpenAIModel)
    model._name = "qa"
    request = GenerationRequest.of(
        "hello", tools=[ToolDefinition(name="lookup")], tool_choice="lookup"
    )
    assert model._request_payload(request, stream=False)["tool_choice"] == {
        "type": "function",
        "function": {"name": "lookup"},
    }

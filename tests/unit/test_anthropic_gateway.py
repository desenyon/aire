"""Anthropic gateway request/response parity (offline, no FastAPI needed)."""

from __future__ import annotations

from aire.core.content import ImageContent, StructuredContent, TextContent
from aire.core.types import Usage
from aire.deployment.gateway import (
    _anthropic_body,
    _build_anthropic_request,
    _map_anthropic_tool_choice,
)
from aire.models.types import GenerationResult, ToolCall


def test_map_anthropic_tool_choice() -> None:
    assert _map_anthropic_tool_choice({"type": "auto"}) == "auto"
    assert _map_anthropic_tool_choice({"type": "any"}) == "required"
    assert _map_anthropic_tool_choice({"type": "none"}) == "none"
    assert _map_anthropic_tool_choice({"type": "tool", "name": "calc"}) == "calc"


def test_build_anthropic_tool_result_messages() -> None:
    req = _build_anthropic_request(
        {
            "model": "claude",
            "max_tokens": 64,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": "4",
                        }
                    ],
                }
            ],
        }
    )
    assert len(req.messages) == 1
    assert req.messages[0].role == "tool"
    assert req.messages[0].tool_call_id == "toolu_1"
    assert req.messages[0].text_content == "4"


def test_build_anthropic_tool_use_structured() -> None:
    req = _build_anthropic_request(
        {
            "model": "claude",
            "max_tokens": 64,
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "calling"},
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "calculator",
                            "input": {"expression": "2+2"},
                        },
                    ],
                }
            ],
        }
    )
    assert req.messages[0].role == "assistant"
    kinds = [c.kind for c in req.messages[0].content]
    assert "text" in kinds
    assert "structured" in kinds
    structured = next(c for c in req.messages[0].content if isinstance(c, StructuredContent))
    assert structured.data["name"] == "calculator"


def test_build_anthropic_image_block() -> None:
    import base64

    raw = base64.b64encode(b"PNG").decode()
    req = _build_anthropic_request(
        {
            "model": "claude",
            "max_tokens": 16,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": raw,
                            },
                        }
                    ],
                }
            ],
        }
    )
    assert any(isinstance(c, ImageContent) for c in req.messages[0].content)


def test_anthropic_body_emits_tool_use() -> None:
    result = GenerationResult(
        content=[TextContent(text="")],
        tool_calls=[ToolCall(id="toolu_9", name="calculator", arguments={"expression": "1+1"})],
        finish_reason="tool_calls",
        model="claude",
        usage=Usage(input_tokens=3, output_tokens=2),
    )
    body = _anthropic_body("claude", "anthropic:claude", result)
    assert body["stop_reason"] == "tool_use"
    assert body["content"][0]["type"] == "tool_use"
    assert body["content"][0]["name"] == "calculator"

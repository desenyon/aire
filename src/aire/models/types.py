"""Library-owned request/response objects for model interactions.

Providers translate these to/from vendor payloads at their boundary; the rest
of the library only ever sees these normalized types.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from aire.core.content import Content, Message, TextContent, coerce_messages
from aire.core.types import Capability, Usage


class ToolDefinition(BaseModel):
    """A tool offered to a model, described by JSON Schema."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})


class ToolCall(BaseModel):
    """A model's request to invoke a tool."""

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_json(cls, id: str, name: str, arguments: str | dict[str, Any]) -> ToolCall:
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError:
                parsed = {"_raw": arguments}
        else:
            parsed = arguments
        return cls(id=id, name=name, arguments=parsed)


class StructuredOutputSpec(BaseModel):
    """Ask the model for output conforming to a JSON Schema."""

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    name: str = "output"
    json_schema: dict[str, Any] = Field(alias="schema")
    strict: bool = True


class GenerationRequest(BaseModel):
    """Everything needed to produce one model completion."""

    model_config = ConfigDict(frozen=True)

    messages: list[Message]
    temperature: float | None = None
    max_tokens: int | None = None
    stop: list[str] | None = None
    tools: list[ToolDefinition] | None = None
    tool_choice: Literal["auto", "none", "required"] | str | None = None
    response_format: StructuredOutputSpec | None = None
    seed: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def of(cls, prompt: str | Message | list[Message], **kwargs: Any) -> GenerationRequest:
        return cls(messages=coerce_messages(prompt), **kwargs)

    def with_messages(self, messages: list[Message]) -> GenerationRequest:
        return self.model_copy(update={"messages": messages})


class GenerationChunk(BaseModel):
    """One incremental piece of a streamed completion."""

    model_config = ConfigDict(frozen=True)

    text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: str | None = None


FinishReason = Literal["stop", "length", "tool_calls", "content_filter", "error"]


class GenerationResult(BaseModel):
    """A completed generation."""

    model_config = ConfigDict(frozen=True)

    content: list[Content]
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: FinishReason = "stop"
    model: str = "unknown"
    usage: Usage = Field(default_factory=Usage)
    raw: dict[str, Any] | None = Field(default=None, exclude=True)

    @property
    def text(self) -> str:
        return "".join(c.text for c in self.content if isinstance(c, TextContent))

    @property
    def message(self) -> Message:
        return Message(role="assistant", content=self.content)

    @classmethod
    def text_result(
        cls, text: str, *, model: str, usage: Usage | None = None, **kwargs: Any
    ) -> GenerationResult:
        return cls(
            content=[TextContent(text=text)],
            model=model,
            usage=usage or Usage(),
            **kwargs,
        )

    def parsed(self, schema_model: type[BaseModel]) -> BaseModel:
        """Validate ``text`` as JSON against a pydantic model."""
        return schema_model.model_validate_json(self.text)


class EmbeddingRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    inputs: list[str]
    metadata: dict[str, Any] = Field(default_factory=dict)


class EmbeddingResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    vectors: list[list[float]]
    model: str = "unknown"
    usage: Usage = Field(default_factory=Usage)

    @property
    def dimension(self) -> int:
        return len(self.vectors[0]) if self.vectors else 0


class CostInfo(BaseModel):
    """Per-token pricing in USD, when known."""

    input_per_million: float | None = None
    output_per_million: float | None = None

    def estimate(self, usage: Usage) -> float:
        cost = 0.0
        if self.input_per_million is not None:
            cost += usage.input_tokens * self.input_per_million / 1_000_000
        if self.output_per_million is not None:
            cost += usage.output_tokens * self.output_per_million / 1_000_000
        return cost


class ModelInfo(BaseModel):
    """Normalized metadata every model must expose."""

    model_config = ConfigDict(frozen=True)

    ref: str
    provider: str
    capabilities: list[Capability] = Field(default_factory=list)
    context_window: int | None = None
    max_output_tokens: int | None = None
    input_kinds: list[str] = Field(default_factory=lambda: ["text"])
    output_kinds: list[str] = Field(default_factory=lambda: ["text"])
    cost: CostInfo = Field(default_factory=CostInfo)
    latency_ms_p50: float | None = None
    hardware: str | None = None

    def supports(self, capability: Capability) -> bool:
        return capability in self.capabilities

    def describe(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

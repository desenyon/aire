"""Shared scalar types: identifiers, usage accounting, health and manifests."""

from __future__ import annotations

import re
import time
import uuid
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from aire.core.errors import ConfigurationError

_REF_PATTERN = re.compile(r"^(?P<provider>[a-z][a-z0-9_-]*):(?P<name>[^\s:]+)$")


class Ref(BaseModel):
    """A ``provider:name`` identifier (models, stores, embedders, tools...)."""

    model_config = ConfigDict(frozen=True)

    provider: str
    name: str

    @classmethod
    def parse(cls, spec: str | Ref) -> Ref:
        if isinstance(spec, Ref):
            return spec
        m = _REF_PATTERN.match(spec.strip())
        if not m:
            raise ConfigurationError(
                f"invalid reference {spec!r}; expected 'provider:name'",
                code="ref.invalid",
                context={"spec": spec},
            )
        return cls(provider=m.group("provider"), name=m.group("name"))

    def __str__(self) -> str:
        return f"{self.provider}:{self.name}"


class HealthStatus(BaseModel):
    """Health probe result for any component."""

    ok: bool
    detail: str = ""
    latency_ms: float | None = None

    @classmethod
    def healthy(cls, detail: str = "ok", latency_ms: float | None = None) -> HealthStatus:
        return cls(ok=True, detail=detail, latency_ms=latency_ms)

    @classmethod
    def unhealthy(cls, detail: str) -> HealthStatus:
        return cls(ok=False, detail=detail)


class Usage(BaseModel):
    """Token and cost accounting for one operation."""

    model_config = ConfigDict(frozen=True)

    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
            latency_ms=self.latency_ms + other.latency_ms,
        )


class Capability(StrEnum):
    """Normalized model/component capabilities."""

    TEXT_GENERATION = "text-generation"
    STREAMING = "streaming"
    TOOL_CALLING = "tool-calling"
    STRUCTURED_OUTPUT = "structured-output"
    EMBEDDINGS = "embeddings"
    VISION_INPUT = "vision-input"
    AUDIO_INPUT = "audio-input"
    IMAGE_GENERATION = "image-generation"
    SPEECH_RECOGNITION = "speech-recognition"
    TEXT_TO_SPEECH = "text-to-speech"
    RERANKING = "reranking"
    CLASSIFICATION = "classification"


class Manifest(BaseModel):
    """Machine-readable self-description emitted by every component.

    Agents use manifests to discover what exists, what it accepts, and what it
    can do without reading source code.
    """

    kind: str
    name: str
    version: str = "0.1.0"
    provider: str | None = None
    capabilities: list[str] = Field(default_factory=list)
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None
    config_schema: dict[str, Any] | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


def new_id(prefix: str) -> str:
    """Generate a short unique identifier with a readable prefix."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def monotonic_ms() -> float:
    return time.perf_counter() * 1000.0

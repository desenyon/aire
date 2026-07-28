"""Tool contracts: specifications, side-effect levels, results."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class SideEffect(StrEnum):
    """Risk classification for a tool's effects on the world.

    Ordered by severity; policies gate execution on this level.
    """

    READ_ONLY = "read_only"
    REVERSIBLE_WRITE = "reversible_write"
    EXTERNAL_SIDE_EFFECT = "external_side_effect"
    HIGH_IMPACT = "high_impact"
    PROHIBITED = "prohibited"


_SEVERITY = {
    SideEffect.READ_ONLY: 0,
    SideEffect.REVERSIBLE_WRITE: 1,
    SideEffect.EXTERNAL_SIDE_EFFECT: 2,
    SideEffect.HIGH_IMPACT: 3,
    SideEffect.PROHIBITED: 4,
}


def at_least(level: SideEffect, threshold: SideEffect) -> bool:
    return _SEVERITY[level] >= _SEVERITY[threshold]


class RetryPolicy(BaseModel):
    attempts: int = 1
    backoff_seconds: float = 0.0


class ToolSpec(BaseModel):
    """Full machine-readable declaration of a tool.

    This is what agents read to decide whether and how to call a tool — the
    universal contract across functions, REST APIs, MCP servers and CLIs.
    """

    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] | None = None
    permissions: list[str] = Field(default_factory=list)
    timeout_seconds: float = 30.0
    retry: RetryPolicy = Field(default_factory=RetryPolicy)
    side_effect: SideEffect = SideEffect.READ_ONLY
    cost_estimate_usd: float | None = None
    audit: bool = True

    def describe(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class ToolResult(BaseModel):
    """Normalized outcome of one tool execution."""

    ok: bool
    output: Any = None
    error: str | None = None
    error_code: str | None = None
    duration_ms: float = 0.0
    tool: str = ""

    @classmethod
    def success(cls, output: Any, *, tool: str, duration_ms: float) -> ToolResult:
        return cls(ok=True, output=output, tool=tool, duration_ms=duration_ms)

    @classmethod
    def failure(
        cls, error: str, *, tool: str, code: str | None = None, duration_ms: float = 0.0
    ) -> ToolResult:
        return cls(ok=False, error=error, error_code=code, tool=tool, duration_ms=duration_ms)

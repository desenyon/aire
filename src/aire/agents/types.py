"""Agent runtime types: configuration, steps, state, results."""

from __future__ import annotations

import time
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from aire.core.content import Message
from aire.core.types import Usage, new_id
from aire.models.types import ToolCall


class AgentStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BUDGET_EXCEEDED = "budget_exceeded"
    MAX_STEPS = "max_steps"
    AWAITING_APPROVAL = "awaiting_approval"


class StepKind(StrEnum):
    MODEL_CALL = "model_call"
    TOOL_CALL = "tool_call"
    OBSERVATION = "observation"
    PERMISSION_DENIED = "permission_denied"
    FINISH = "finish"
    ERROR = "error"


class AgentStep(BaseModel):
    """One deterministic transition in the agent state machine."""

    index: int
    kind: StepKind
    detail: dict[str, Any] = Field(default_factory=dict)
    usage: Usage = Field(default_factory=Usage)
    timestamp: float = Field(default_factory=time.time)

    def describe(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class AgentConfig(BaseModel):
    """Execution limits and behavior for an agent run."""

    max_steps: int = 12
    token_budget: int | None = None
    cost_budget_usd: float | None = None
    planning: bool = False
    system_prompt: str | None = None
    # Side-effect levels that require approval before tool execution.
    approval_levels: list[str] = Field(
        default_factory=lambda: ["external_side_effect", "high_impact", "prohibited"]
    )
    # Permissions granted to this agent for tool execution.
    permissions: list[str] = Field(default_factory=list)
    temperature: float | None = None
    # When True, independent tool calls in one model turn run concurrently.
    parallel_tools: bool = False


class AgentState(BaseModel):
    """Full serializable state of an agent execution (checkpointable)."""

    id: str = Field(default_factory=lambda: new_id("agent_run"))
    input: str = ""
    messages: list[Message] = Field(default_factory=list)
    steps: list[AgentStep] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    status: AgentStatus = AgentStatus.RUNNING
    output: str | None = None
    error: str | None = None


class AgentResult(BaseModel):
    """Final outcome of an agent run."""

    output: str
    status: AgentStatus
    steps: list[AgentStep] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    run_id: str = ""
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == AgentStatus.COMPLETED

    @property
    def tool_calls(self) -> list[ToolCall]:
        calls: list[ToolCall] = []
        for step in self.steps:
            if step.kind == StepKind.TOOL_CALL and "call" in step.detail:
                calls.append(ToolCall.model_validate(step.detail["call"]))
        return calls

    def describe(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"steps": {"__all__": {"usage"}}})

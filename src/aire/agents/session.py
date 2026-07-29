"""Durable agent sessions — resume mid-run across process restarts."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from aire.agents.types import AgentResult, AgentState, AgentStatus, AgentStep
from aire.core.content import Message
from aire.core.errors import ConfigurationError
from aire.core.types import Usage, new_id


class SessionState(BaseModel):
    id: str = Field(default_factory=lambda: new_id("sess"))
    goal: str = ""
    status: str = "running"  # running | paused | completed | failed
    steps: list[dict[str, Any]] = Field(default_factory=list)
    messages: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    updated_at: float = Field(default_factory=time.time)
    result: dict[str, Any] | None = None

    def to_agent_state(self, input: str | None = None) -> AgentState:
        """Rebuild an :class:`AgentState` from persisted messages/steps/status."""
        messages: list[Message] = []
        for raw in self.messages:
            try:
                messages.append(Message.model_validate(raw))
            except Exception:  # noqa: S112
                continue
        steps: list[AgentStep] = []
        for raw in self.steps:
            try:
                steps.append(AgentStep.model_validate(raw))
            except Exception:  # noqa: S112
                continue
        status_map = {
            "running": AgentStatus.RUNNING,
            "paused": AgentStatus.RUNNING,  # resume continues as running
            "completed": AgentStatus.COMPLETED,
            "failed": AgentStatus.FAILED,
        }
        return AgentState(
            id=self.id,
            input=input if input is not None else self.goal,
            messages=messages,
            steps=steps,
            usage=Usage(),
            status=status_map.get(self.status, AgentStatus.RUNNING),
            output=(self.result or {}).get("output") if self.result else None,
            error=self.metadata.get("error"),
        )


class DurableSession:
    """JSONL/JSON backed session store for agent runs."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.state = SessionState()
        if self.path.is_file():
            self.load()

    def load(self) -> SessionState:
        try:
            self.state = SessionState.model_validate(json.loads(self.path.read_text()))
        except (json.JSONDecodeError, ValueError) as exc:
            raise ConfigurationError(
                f"corrupt session at {self.path}",
                code="session.corrupt",
                cause=exc,
            ) from exc
        return self.state

    def save(self) -> Path:
        self.state.updated_at = time.time()
        self.path.write_text(json.dumps(self.state.model_dump(mode="json"), indent=2))
        return self.path

    def append_step(self, step: AgentStep | dict[str, Any]) -> None:
        payload = step.model_dump(mode="json") if isinstance(step, AgentStep) else dict(step)
        self.state.steps.append(payload)
        self.save()

    def persist_messages(self, messages: list[Message] | list[dict[str, Any]]) -> None:
        """Persist conversation messages (as model_dump dicts)."""
        dumped: list[dict[str, Any]] = []
        for m in messages:
            if isinstance(m, Message):
                dumped.append(m.model_dump(mode="json"))
            else:
                dumped.append(dict(m))
        self.state.messages = dumped
        self.save()

    def hydrate_agent_state(self, input: str | None = None) -> AgentState:
        """Return an AgentState rebuilt from this session (for resume)."""
        return self.state.to_agent_state(input)

    def to_agent_state(self, input: str | None = None) -> AgentState:
        return self.hydrate_agent_state(input)

    def complete(self, result: AgentResult | dict[str, Any]) -> None:
        self.state.status = "completed"
        self.state.result = (
            result.model_dump(mode="json") if isinstance(result, AgentResult) else dict(result)
        )
        self.save()

    def fail(self, error: str) -> None:
        self.state.status = "failed"
        self.state.metadata["error"] = error
        self.save()

    def pause(self) -> None:
        self.state.status = "paused"
        self.save()

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "session",
            "id": self.state.id,
            "status": self.state.status,
            "steps": len(self.state.steps),
            "messages": len(self.state.messages),
            "path": str(self.path),
        }

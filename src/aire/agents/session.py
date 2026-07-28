"""Durable agent sessions — resume mid-run across process restarts."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from aire.agents.types import AgentResult, AgentStep
from aire.core.errors import ConfigurationError
from aire.core.types import new_id


class SessionState(BaseModel):
    id: str = Field(default_factory=lambda: new_id("sess"))
    goal: str = ""
    status: str = "running"  # running | paused | completed | failed
    steps: list[dict[str, Any]] = Field(default_factory=list)
    messages: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    updated_at: float = Field(default_factory=time.time)
    result: dict[str, Any] | None = None


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
            "path": str(self.path),
        }

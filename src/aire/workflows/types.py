"""Workflow primitives: nodes, edges, events, state."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from aire.core.types import new_id

# A node receives its input and the shared context dict; returns its output.
NodeFn = Callable[[Any, dict[str, Any]], Any | Awaitable[Any]]
# A condition inspects a node's output and decides whether to follow the edge.
Condition = Callable[[Any], bool]


class NodeStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"
    AWAITING_APPROVAL = "awaiting_approval"


class NodeSpec(BaseModel):
    """Static definition of a node."""

    name: str
    retries: int = 0
    retry_backoff: float = 0.25
    timeout_seconds: float | None = None
    requires_approval: bool = False


class Edge(BaseModel):
    """A directed connection with an optional condition on the source output."""

    source: str
    target: str
    condition: str | None = None  # description only; callable held separately


class WorkflowEvent(BaseModel):
    """Emitted as a workflow runs (streamable)."""

    kind: str  # node_started | node_completed | node_failed | node_skipped | workflow_completed
    node: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)


class NodeRecord(BaseModel):
    """Execution record for one node visit."""

    name: str
    status: NodeStatus
    visits: int = 0
    duration_ms: float = 0.0
    error: str | None = None


class WorkflowState(BaseModel):
    """Serializable execution state — the checkpoint unit."""

    id: str = Field(default_factory=lambda: new_id("wf"))
    input: Any = None
    outputs: dict[str, Any] = Field(default_factory=dict)
    records: list[NodeRecord] = Field(default_factory=list)
    completed: bool = False
    error: str | None = None

    def record(self, name: str, status: NodeStatus, **kw: Any) -> NodeRecord:
        for existing in self.records:
            if existing.name == name:
                existing.status = status
                existing.visits += 1
                for key, value in kw.items():
                    setattr(existing, key, value)
                return existing
        rec = NodeRecord(name=name, status=status, visits=1, **kw)
        self.records.append(rec)
        return rec


class WorkflowResult(BaseModel):
    """Final outcome of a workflow run."""

    output: Any = None
    outputs: dict[str, Any] = Field(default_factory=dict)
    records: list[NodeRecord] = Field(default_factory=list)
    ok: bool = True
    error: str | None = None
    run_id: str = ""

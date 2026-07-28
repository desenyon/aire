"""Execution context: identity, budgets, cancellation and trace correlation."""

from __future__ import annotations

import asyncio
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from aire.core.errors import BudgetExceededError, TimeoutError
from aire.core.types import Usage, new_id

_current: ContextVar[ExecutionContext | None] = ContextVar("aire_context", default=None)


@dataclass
class Budget:
    """Limits for one execution. ``None`` means unlimited."""

    max_tokens: int | None = None
    max_cost_usd: float | None = None
    max_steps: int | None = None
    deadline_seconds: float | None = None

    def check(self, usage: Usage, steps: int, started: float) -> None:
        if self.max_tokens is not None and usage.total_tokens > self.max_tokens:
            raise BudgetExceededError(
                f"token budget exceeded ({usage.total_tokens} > {self.max_tokens})",
                context={"used": usage.total_tokens, "limit": self.max_tokens},
            )
        if self.max_cost_usd is not None and usage.cost_usd > self.max_cost_usd:
            raise BudgetExceededError(
                f"cost budget exceeded (${usage.cost_usd:.4f} > ${self.max_cost_usd:.4f})",
                context={"used": usage.cost_usd, "limit": self.max_cost_usd},
            )
        if self.max_steps is not None and steps > self.max_steps:
            raise BudgetExceededError(
                f"step budget exceeded ({steps} > {self.max_steps})",
                context={"used": steps, "limit": self.max_steps},
            )
        if self.deadline_seconds is not None and time.monotonic() - started > self.deadline_seconds:
            raise TimeoutError(
                f"deadline exceeded ({self.deadline_seconds}s)",
                context={"limit": self.deadline_seconds},
            )


@dataclass
class ExecutionContext:
    """Carries per-run identity and control state through the call stack."""

    run_id: str = field(default_factory=lambda: new_id("run"))
    trace_id: str | None = None
    parent_id: str | None = None
    user_id: str | None = None
    permissions: set[str] = field(default_factory=set)
    budget: Budget = field(default_factory=Budget)
    metadata: dict[str, Any] = field(default_factory=dict)
    usage: Usage = field(default_factory=Usage)
    steps: int = 0
    started: float = field(default_factory=time.monotonic)
    _cancel: asyncio.Event = field(default_factory=asyncio.Event, repr=False)

    # -- context variable plumbing ------------------------------------------------

    def __enter__(self) -> ExecutionContext:
        self._token = _current.set(self)
        return self

    def __exit__(self, *exc_info: Any) -> None:
        _current.reset(self._token)

    @classmethod
    def current(cls) -> ExecutionContext:
        """Return the ambient context, creating a detached one if absent."""
        ctx = _current.get()
        if ctx is None:
            ctx = cls()
            _current.set(ctx)
        return ctx

    def child(self, **metadata: Any) -> ExecutionContext:
        """Derive a child context that inherits identity, permissions and budget."""
        return ExecutionContext(
            trace_id=self.trace_id,
            parent_id=self.run_id,
            user_id=self.user_id,
            permissions=set(self.permissions),
            budget=self.budget,
            metadata={**self.metadata, **metadata},
        )

    # -- control --------------------------------------------------------------------

    def record_usage(self, usage: Usage) -> None:
        self.usage = self.usage + usage
        self.budget.check(self.usage, self.steps, self.started)

    def tick(self, n: int = 1) -> None:
        self.steps += n
        self.budget.check(self.usage, self.steps, self.started)

    def cancel(self) -> None:
        self._cancel.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def check_cancelled(self) -> None:
        if self._cancel.is_set():
            raise TimeoutError("execution cancelled", code="aire.cancelled")

    @property
    def elapsed_ms(self) -> float:
        return (time.monotonic() - self.started) * 1000.0

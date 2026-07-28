"""Cron-like scheduler for workflows (interval runner; APScheduler optional)."""

from __future__ import annotations

import asyncio
import importlib.util
import time
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel

from aire.core.errors import ConfigurationError
from aire.workflows.graph import Workflow


class ScheduleEntry(BaseModel):
    name: str
    interval_seconds: float
    workflow: str
    input: Any = None
    enabled: bool = True
    last_run: float | None = None
    runs: int = 0
    last_error: str | None = None


class Scheduler:
    """Pure interval scheduler — no external deps.

    Call :meth:`tick` periodically (or :meth:`run_forever`).
    """

    def __init__(self) -> None:
        self.entries: dict[str, ScheduleEntry] = {}
        self.workflows: dict[str, Workflow] = {}
        self._handlers: dict[str, Callable[[], Awaitable[Any]]] = {}
        self.history: list[dict[str, Any]] = []

    def register_workflow(self, name: str, workflow: Workflow) -> None:
        self.workflows[name] = workflow

    def every(
        self,
        interval_seconds: float,
        workflow: str | Workflow | Callable[[], Awaitable[Any]],
        *,
        name: str | None = None,
        input: Any = None,
    ) -> ScheduleEntry:
        if isinstance(workflow, Workflow):
            wf_name = workflow.name
            self.workflows[wf_name] = workflow
            entry_name = name or wf_name
            self.entries[entry_name] = ScheduleEntry(
                name=entry_name,
                interval_seconds=interval_seconds,
                workflow=wf_name,
                input=input,
            )
        elif callable(workflow) and not isinstance(workflow, str):
            entry_name = name or str(getattr(workflow, "__name__", "job"))
            handler = workflow
            self._handlers[entry_name] = handler
            self.entries[entry_name] = ScheduleEntry(
                name=entry_name,
                interval_seconds=interval_seconds,
                workflow=entry_name,
                input=input,
            )
        else:
            entry_name = name or str(workflow)
            self.entries[entry_name] = ScheduleEntry(
                name=entry_name,
                interval_seconds=interval_seconds,
                workflow=str(workflow),
                input=input,
            )
        return self.entries[entry_name]

    async def tick(self, *, now: float | None = None) -> list[dict[str, Any]]:
        """Run any due entries once. Returns list of run records."""
        ts = now if now is not None else time.time()
        ran: list[dict[str, Any]] = []
        for entry in list(self.entries.values()):
            if not entry.enabled:
                continue
            due = entry.last_run is None or (ts - entry.last_run) >= entry.interval_seconds
            if not due:
                continue
            record = await self._run_entry(entry)
            entry.last_run = ts
            entry.runs += 1
            ran.append(record)
            self.history.append(record)
        return ran

    async def _run_entry(self, entry: ScheduleEntry) -> dict[str, Any]:
        started = time.time()
        try:
            if entry.name in self._handlers:
                await self._handlers[entry.name]()
                output = None
            else:
                wf = self.workflows.get(entry.workflow)
                if wf is None:
                    raise ConfigurationError(
                        f"scheduled workflow {entry.workflow!r} not registered",
                        code="schedule.workflow_missing",
                    )
                result = await wf.run(entry.input)
                output = result.output
                if not result.ok:
                    raise RuntimeError(result.error or "workflow failed")
            entry.last_error = None
            return {
                "name": entry.name,
                "ok": True,
                "output": output,
                "duration_s": time.time() - started,
            }
        except Exception as exc:
            entry.last_error = f"{type(exc).__name__}: {exc}"
            return {
                "name": entry.name,
                "ok": False,
                "error": entry.last_error,
                "duration_s": time.time() - started,
            }

    async def run_forever(self, *, poll_seconds: float = 1.0, max_ticks: int | None = None) -> None:
        ticks = 0
        while max_ticks is None or ticks < max_ticks:
            await self.tick()
            ticks += 1
            await asyncio.sleep(poll_seconds)

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "scheduler",
            "entries": [e.model_dump() for e in self.entries.values()],
            "workflows": sorted(self.workflows),
            "history": len(self.history),
            "apscheduler": importlib.util.find_spec("apscheduler") is not None,
        }


def create_apscheduler(scheduler: Scheduler | None = None) -> Any:
    """Bridge to APScheduler when installed."""
    if importlib.util.find_spec("apscheduler") is None:
        raise ConfigurationError(
            "APScheduler is optional: pip install apscheduler "
            "(or use aire.schedule.Scheduler interval runner)",
            code="schedule.apscheduler_missing",
        )
    from apscheduler.schedulers.asyncio import AsyncIOScheduler  # type: ignore[import-not-found]

    sched = scheduler or Scheduler()
    apo = AsyncIOScheduler()
    for entry in sched.entries.values():

        async def _job(name: str = entry.name) -> None:
            e = sched.entries[name]
            await sched._run_entry(e)
            e.last_run = time.time()
            e.runs += 1

        apo.add_job(_job, "interval", seconds=entry.interval_seconds, id=entry.name)
    return apo


def describe() -> dict[str, Any]:
    return {
        "kind": "schedule",
        "backends": ["interval", "apscheduler (optional)"],
        "apscheduler": importlib.util.find_spec("apscheduler") is not None,
    }

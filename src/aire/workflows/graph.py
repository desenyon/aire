"""Graph-based workflow engine.

Semantics:
- The entry node receives the workflow input.
- A node with one completed predecessor receives that predecessor's output;
  with several, a ``{name: output}`` dict (fan-in join).
- Conditional edges fire only when their condition accepts the source output.
- A node becomes ready when at least one incoming edge fired and all of its
  predecessors reached a terminal state; nodes whose edges never fire are
  skipped (which satisfies downstream joins).
- Ready nodes execute concurrently; cycles are bounded by ``max_visits``.
- When ``checkpoint_path`` is set, state is persisted after every node.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any

from aire.core.errors import WorkflowError
from aire.core.serialization import write_json_file
from aire.workflows.types import (
    Condition,
    NodeFn,
    NodeSpec,
    NodeStatus,
    WorkflowEvent,
    WorkflowResult,
    WorkflowState,
)

ApproverFn = Callable[[str], bool | Awaitable[bool]]

_TERMINAL = {NodeStatus.COMPLETED, NodeStatus.SKIPPED, NodeStatus.FAILED}


class Workflow:
    """A directed graph of executable nodes with deterministic scheduling."""

    def __init__(
        self,
        name: str = "workflow",
        *,
        max_visits: int = 3,
        checkpoint_path: str | Path | None = None,
        approver: ApproverFn | None = None,
    ) -> None:
        self.name = name
        self.max_visits = max_visits
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.approver = approver
        self._nodes: dict[str, tuple[NodeSpec, NodeFn]] = {}
        self._edges: list[tuple[str, str, Condition | None]] = []
        self._entry: str | None = None
        self._last_state: WorkflowState | None = None

    # -- definition ----------------------------------------------------------------

    def add(
        self,
        name: str,
        fn: NodeFn,
        *,
        retries: int = 0,
        timeout_seconds: float | None = None,
        requires_approval: bool = False,
    ) -> Workflow:
        """Add a node. The first node added becomes the default entry point."""
        if name in self._nodes:
            raise WorkflowError(f"duplicate node {name!r}", context={"node": name})
        self._nodes[name] = (
            NodeSpec(
                name=name,
                retries=retries,
                timeout_seconds=timeout_seconds,
                requires_approval=requires_approval,
            ),
            fn,
        )
        if self._entry is None:
            self._entry = name
        return self

    def connect(self, source: str, target: str, *, when: Condition | None = None) -> Workflow:
        """Connect two nodes, optionally gated on a condition over the output."""
        for node in (source, target):
            if node not in self._nodes:
                raise WorkflowError(
                    f"unknown node {node!r} in edge {source}->{target}",
                    context={"source": source, "target": target},
                )
        self._edges.append((source, target, when))
        return self

    def entry(self, name: str) -> Workflow:
        if name not in self._nodes:
            raise WorkflowError(f"unknown entry node {name!r}")
        self._entry = name
        return self

    # -- graph queries --------------------------------------------------------------

    def _predecessors(self, node: str) -> list[str]:
        return [s for s, t, _ in self._edges if t == node]

    def _successors(self, node: str) -> list[tuple[str, Condition | None]]:
        return [(t, c) for s, t, c in self._edges if s == node]

    # -- execution -----------------------------------------------------------------

    async def run(self, input: Any = None, *, state: WorkflowState | None = None) -> WorkflowResult:
        """Execute the workflow to completion (optionally resuming from ``state``)."""
        async for _ in self.run_stream(input, state=state):
            pass
        assert self._last_state is not None
        wf_state = self._last_state
        return WorkflowResult(
            output=_terminal_output(self, wf_state) if not wf_state.error else None,
            outputs=wf_state.outputs,
            records=wf_state.records,
            ok=wf_state.error is None,
            error=wf_state.error,
            run_id=wf_state.id,
        )

    async def run_stream(
        self, input: Any = None, *, state: WorkflowState | None = None
    ) -> AsyncIterator[WorkflowEvent]:
        """Execute, yielding an event after every node transition."""
        wf_state = state or WorkflowState(input=input)
        context: dict[str, Any] = {"workflow": self.name, "run_id": wf_state.id}
        status, visits, fired, consumed, ready = self._init_scheduling(wf_state, state)

        def _finish_node(
            name: str,
            node_status: NodeStatus,
            output: Any = None,
            error: str | None = None,
            duration_ms: float = 0.0,
        ) -> WorkflowEvent:
            return self._finish_node(
                wf_state, status, fired, name, node_status, output, error, duration_ms
            )

        try:
            while ready:
                # Launch all ready nodes concurrently.
                current, ready = sorted(ready), set()
                tasks = {
                    name: asyncio.create_task(self._run_node(name, wf_state, context, status))
                    for name in current
                }
                for name in tasks:
                    yield WorkflowEvent(kind="node_started", node=name)
                failures: list[str] = []
                async for event in self._collect_wave(tasks, visits, _finish_node, failures):
                    yield event
                if failures and wf_state.error is None:
                    wf_state.error = "; ".join(failures)
                    break
                # Schedule next wave, then mark unreachable nodes skipped.
                ready |= self._schedule_next(status, visits, fired, consumed, ready)
                skipped = self._skip_unreachable(status, visits, fired)
                for name in skipped:
                    yield _finish_node(name, NodeStatus.SKIPPED)
                if not ready and not skipped:
                    self._check_stalled(status)
            wf_state.completed = wf_state.error is None
        except WorkflowError as exc:
            wf_state.error = exc.message
            wf_state.completed = False
            yield WorkflowEvent(kind="workflow_failed", data={"error": exc.message})
        self._checkpoint(wf_state)
        self._last_state = wf_state
        yield WorkflowEvent(
            kind="workflow_completed" if wf_state.completed else "workflow_failed",
            data={"outputs": list(wf_state.outputs), "error": wf_state.error},
        )

    async def _run_node(
        self,
        name: str,
        wf_state: WorkflowState,
        context: dict[str, Any],
        status: dict[str, NodeStatus],
    ) -> tuple[NodeStatus, Any, float]:
        """Execute one node with approval, retries and timeout."""
        spec, fn = self._nodes[name]
        await self._check_approval(name, spec)
        node_input = self._node_input(name, wf_state)
        started = time.perf_counter()
        last_error: BaseException | None = None
        for attempt in range(spec.retries + 1):
            try:
                result = fn(node_input, context)
                if inspect.isawaitable(result):
                    if spec.timeout_seconds:
                        result = await asyncio.wait_for(result, timeout=spec.timeout_seconds)
                    else:
                        result = await result
                return NodeStatus.COMPLETED, result, (time.perf_counter() - started) * 1000.0
            except TimeoutError:
                last_error = WorkflowError(
                    f"node {name!r} timed out after {spec.timeout_seconds}s",
                    context={"node": name},
                )
                break
            except Exception as exc:
                last_error = exc
                if attempt < spec.retries:
                    await asyncio.sleep(spec.retry_backoff * (attempt + 1))
        assert last_error is not None
        raise WorkflowError(
            f"node {name!r} failed: {last_error}",
            context={"node": name},
            cause=last_error,
        ) from last_error

    def _check_stalled(self, status: dict[str, NodeStatus]) -> None:
        unresolved = [n for n in self._nodes if status.get(n) not in _TERMINAL]
        if unresolved:
            raise WorkflowError(
                f"workflow stalled; unresolved nodes: {unresolved}",
                context={"unresolved": unresolved},
            )

    def _finish_node(
        self,
        wf_state: WorkflowState,
        status: dict[str, NodeStatus],
        fired: dict[tuple[str, str], int],
        name: str,
        node_status: NodeStatus,
        output: Any = None,
        error: str | None = None,
        duration_ms: float = 0.0,
    ) -> WorkflowEvent:
        """Record a terminal node transition and fire its matching out-edges."""
        status[name] = node_status
        wf_state.record(name, node_status, duration_ms=duration_ms, error=error)
        if node_status == NodeStatus.COMPLETED:
            wf_state.outputs[name] = output
            for target, condition in self._successors(name):
                if condition is None or condition(output):
                    edge = (name, target)
                    fired[edge] = fired.get(edge, 0) + 1
        self._checkpoint(wf_state)
        kind = {
            NodeStatus.COMPLETED: "node_completed",
            NodeStatus.FAILED: "node_failed",
            NodeStatus.SKIPPED: "node_skipped",
        }[node_status]
        return WorkflowEvent(kind=kind, node=name, data={"error": error} if error else {})

    async def _collect_wave(
        self,
        tasks: dict[str, asyncio.Task[tuple[NodeStatus, Any, float]]],
        visits: dict[str, int],
        finish_node: Any,
        failures: list[str],
    ) -> AsyncIterator[WorkflowEvent]:
        """Gather one wave of node tasks, finishing each node and yielding events."""
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        for name, result in zip(tasks, results, strict=True):
            visits[name] = visits.get(name, 0) + 1
            if isinstance(result, BaseException):
                failures.append(f"{name}: {result}")
                yield finish_node(name, NodeStatus.FAILED, error=str(result))
            else:
                node_status, output, duration_ms = result
                yield finish_node(name, node_status, output, duration_ms=duration_ms)

    def _init_scheduling(
        self, wf_state: WorkflowState, state: WorkflowState | None
    ) -> tuple[
        dict[str, NodeStatus],
        dict[str, int],
        dict[tuple[str, str], int],
        dict[tuple[str, str], int],
        set[str],
    ]:
        """Build the initial scheduling maps, including resume seeding."""
        status: dict[str, NodeStatus] = {rec.name: rec.status for rec in wf_state.records}
        visits: dict[str, int] = {rec.name: rec.visits for rec in wf_state.records}
        fired: dict[tuple[str, str], int] = {}  # edge -> times fired
        consumed: dict[tuple[str, str], int] = {}  # edge -> firings already scheduled
        ready: set[str] = set()
        resuming = state is not None and bool(state.records)
        if resuming:
            status = self._prepare_resume(wf_state, status, fired, consumed)
        if self._entry is not None and status.get(self._entry) not in _TERMINAL:
            ready.add(self._entry)
        elif resuming:
            # The entry is terminal: seed the first wave from reconstructed
            # edge firings — otherwise nothing would ever run.
            ready |= self._schedule_next(status, visits, fired, consumed, ready)
        return status, visits, fired, consumed, ready

    def _prepare_resume(
        self,
        wf_state: WorkflowState,
        status: dict[str, NodeStatus],
        fired: dict[tuple[str, str], int],
        consumed: dict[tuple[str, str], int],
    ) -> dict[str, NodeStatus]:
        """Resume semantics: clear the persisted failure, treat FAILED nodes as
        pending, and rebuild runtime-local edge firing counts from persisted
        statuses + outputs (edges into finished targets are consumed so those
        never re-run). Returns the filtered status map."""
        wf_state.error = None
        wf_state.completed = False
        status = {n: s for n, s in status.items() if s != NodeStatus.FAILED}
        for source, target, condition in self._edges:
            if status.get(source) != NodeStatus.COMPLETED or source not in wf_state.outputs:
                continue
            output = wf_state.outputs[source]
            if condition is None or condition(output):
                edge = (source, target)
                fired[edge] = 1
                consumed[edge] = 1 if status.get(target) in _TERMINAL else 0
        return status

    def _schedule_next(
        self,
        status: dict[str, NodeStatus],
        visits: dict[str, int],
        fired: dict[tuple[str, str], int],
        consumed: dict[tuple[str, str], int],
        ready: set[str],
    ) -> set[str]:
        """Nodes whose in-edges fired since their last scheduling and whose
        predecessors are all terminal; loops revisit until ``max_visits``."""
        scheduled: set[str] = set()
        for name in self._nodes:
            if name in ready or status.get(name) == NodeStatus.FAILED:
                continue
            if visits.get(name, 0) >= self.max_visits:
                continue
            preds = self._predecessors(name)
            if not preds:
                if name == self._entry and visits.get(name, 0) == 0:
                    scheduled.add(name)
                continue
            has_new_firing = any(
                fired.get((pred, name), 0) > consumed.get((pred, name), 0) for pred in preds
            )
            if has_new_firing and all(status.get(pred) in _TERMINAL for pred in preds):
                for pred in preds:
                    edge = (pred, name)
                    consumed[edge] = fired.get(edge, 0)
                scheduled.add(name)
        return scheduled

    def _skip_unreachable(
        self,
        status: dict[str, NodeStatus],
        visits: dict[str, int],
        fired: dict[tuple[str, str], int],
    ) -> list[str]:
        """Never-visited nodes whose predecessors all terminated without firing
        an in-edge (e.g. a conditional branch that was not taken)."""
        skipped: list[str] = []
        for name in self._nodes:
            if visits.get(name, 0) > 0:
                continue
            preds = self._predecessors(name)
            if (
                preds
                and all(status.get(p) in _TERMINAL for p in preds)
                and all(fired.get((p, name), 0) == 0 for p in preds)
            ):
                skipped.append(name)
        return skipped

    async def _check_approval(self, name: str, spec: NodeSpec) -> None:
        if not spec.requires_approval:
            return
        if self.approver is None:
            raise WorkflowError(
                f"node {name!r} requires approval but no approver is configured",
                context={"node": name},
            )
        decision = self.approver(name)
        if inspect.isawaitable(decision):
            decision = await decision
        if not decision:
            raise WorkflowError(f"approval denied for node {name!r}", context={"node": name})

    def _node_input(self, name: str, wf_state: WorkflowState) -> Any:
        if name == self._entry and not self._predecessors(name):
            return wf_state.input
        preds = self._predecessors(name)
        completed = [p for p in preds if p in wf_state.outputs]
        if not completed:
            return wf_state.input if name == self._entry else None
        if len(completed) == 1:
            return wf_state.outputs[completed[0]]
        return {p: wf_state.outputs[p] for p in completed}

    def _checkpoint(self, wf_state: WorkflowState) -> None:
        if self.checkpoint_path is not None:
            write_json_file(self.checkpoint_path, wf_state)

    @staticmethod
    def load_checkpoint(path: str | Path) -> WorkflowState:
        """Load a persisted :class:`WorkflowState` from a checkpoint file."""
        from aire.core.errors import ConfigurationError
        from aire.core.serialization import read_json_file

        target = Path(path)
        if not target.exists():
            raise ConfigurationError(
                f"workflow checkpoint not found: {target}",
                code="workflow.checkpoint_missing",
                context={"path": str(target)},
            )
        return WorkflowState.model_validate(read_json_file(target))

    async def resume(self, checkpoint_path: str | Path | None = None) -> WorkflowResult:
        """Resume from a checkpoint file (defaults to ``checkpoint_path``).

        Completed nodes are not re-executed; pending nodes continue with the
        persisted inputs/outputs.
        """
        path = Path(checkpoint_path) if checkpoint_path else self.checkpoint_path
        if path is None:
            raise WorkflowError(
                "resume() requires a checkpoint path",
                context={"workflow": self.name},
            )
        return await self.run(state=self.load_checkpoint(path))

    # -- introspection -----------------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "workflow",
            "name": self.name,
            "nodes": [spec.model_dump(mode="json") for spec, _ in self._nodes.values()],
            "edges": [
                {"source": s, "target": t, "conditional": c is not None} for s, t, c in self._edges
            ],
            "entry": self._entry,
            "max_visits": self.max_visits,
        }


def _terminal_output(workflow: Workflow, state: WorkflowState) -> Any:
    """Output of the workflow: the completed sink node's output (or a join)."""
    sinks = [name for name in workflow._nodes if not workflow._successors(name)]
    completed_sinks = [s for s in sinks if s in state.outputs]
    if len(completed_sinks) == 1:
        return state.outputs[completed_sinks[0]]
    if completed_sinks:
        return {name: state.outputs[name] for name in completed_sinks}
    if workflow._entry and workflow._entry in state.outputs:
        return state.outputs[workflow._entry]
    return None

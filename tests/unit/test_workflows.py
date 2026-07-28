"""Workflow graph execution: linear, branching, parallel, retries, checkpoints."""

from __future__ import annotations

from pathlib import Path

import pytest

from aire.core.errors import WorkflowError
from aire.workflows import NodeStatus, Workflow
from tests.conftest import arun


def test_linear_pipeline() -> None:
    wf = Workflow("linear")
    wf.add("a", lambda x, ctx: x + 1)
    wf.add("b", lambda x, ctx: x * 10)
    wf.connect("a", "b")
    result = arun(wf.run(4))
    assert result.ok
    assert result.output == 50


def test_conditional_branching() -> None:
    wf = Workflow("branch")
    wf.add("start", lambda x, ctx: x)
    wf.add("big", lambda x, ctx: f"big:{x}")
    wf.add("small", lambda x, ctx: f"small:{x}")
    wf.connect("start", "big", when=lambda out: out > 10)
    wf.connect("start", "small", when=lambda out: out <= 10)
    big = arun(wf.run(42))
    assert big.output == "big:42"
    statuses = {r.name: r.status for r in big.records}
    assert statuses["small"] == NodeStatus.SKIPPED
    small = arun(wf.run(3))
    assert small.output == "small:3"


def test_parallel_fanout_and_join() -> None:
    wf = Workflow("parallel")
    wf.add("start", lambda x, ctx: x)
    wf.add("left", lambda x, ctx: x + 1)
    wf.add("right", lambda x, ctx: x + 2)
    wf.add("join", lambda x, ctx: x["left"] + x["right"])
    wf.connect("start", "left")
    wf.connect("start", "right")
    wf.connect("left", "join")
    wf.connect("right", "join")
    result = arun(wf.run(10))
    assert result.ok
    assert result.output == 23


def test_retries_eventually_succeed() -> None:
    attempts = 0

    def flaky(x: int, ctx: dict) -> int:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ValueError("flaky")
        return x * 2

    wf = Workflow("retry")
    wf.add("flaky", flaky, retries=3)
    result = arun(wf.run(5))
    assert result.ok and result.output == 10
    assert attempts == 3


def test_timeout_fails_node() -> None:
    import asyncio

    async def slow(x: int, ctx: dict) -> int:
        await asyncio.sleep(5)
        return x

    wf = Workflow("timeout")
    wf.add("slow", slow, timeout_seconds=0.05)
    result = arun(wf.run(1))
    assert not result.ok
    assert "timed out" in (result.error or "")


def test_duplicate_node_rejected() -> None:
    wf = Workflow("dup")
    wf.add("a", lambda x, ctx: x)
    with pytest.raises(WorkflowError):
        wf.add("a", lambda x, ctx: x)


def test_unknown_edge_rejected() -> None:
    wf = Workflow("edge")
    wf.add("a", lambda x, ctx: x)
    with pytest.raises(WorkflowError):
        wf.connect("a", "ghost")


def test_checkpoint_written(tmp_path: Path) -> None:
    checkpoint = tmp_path / "state.json"
    wf = Workflow("checkpoint", checkpoint_path=checkpoint)
    wf.add("a", lambda x, ctx: x + 1)
    result = arun(wf.run(1))
    assert result.ok
    import json

    state = json.loads(checkpoint.read_text())
    assert state["outputs"]["a"] == 2
    assert state["completed"] is True


def test_streaming_events() -> None:
    async def _collect() -> list[str]:
        wf = Workflow("stream")
        wf.add("a", lambda x, ctx: x)
        wf.add("b", lambda x, ctx: x)
        wf.connect("a", "b")
        return [e.kind async for e in wf.run_stream(1)]

    kinds = arun(_collect())
    assert kinds[0] == "node_started"
    assert "node_completed" in kinds
    assert kinds[-1] == "workflow_completed"


def test_loop_bounded_by_max_visits() -> None:
    counter = {"n": 0}

    def step(x: int, ctx: dict) -> int:
        counter["n"] += 1
        return x + 1

    wf = Workflow("loop", max_visits=3)
    wf.add("step", step)
    wf.connect("step", "step")
    result = arun(wf.run(0))
    assert result.ok
    assert counter["n"] == 3


def test_describe_manifest() -> None:
    wf = Workflow("manifest")
    wf.add("a", lambda x, ctx: x)
    wf.add("b", lambda x, ctx: x)
    wf.connect("a", "b", when=lambda out: True)
    manifest = wf.describe()
    assert manifest["kind"] == "workflow"
    assert manifest["edges"][0]["conditional"] is True
    assert manifest["entry"] == "a"

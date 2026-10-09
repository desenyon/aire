"""Workflow scheduler checkpoints preserve edge firings and visit budgets."""

import asyncio

from aire.models.base import run_sync
from aire.workflows.graph import Workflow
from aire.workflows.types import NodeStatus, WorkflowState


def test_skipped_only_branch_completes_every_node():
    flow = Workflow().add("entry", lambda x, ctx: x)
    for name in ["third", "second", "first"]:
        flow.add(name, lambda x, ctx: x)
    flow.connect("entry", "first", when=lambda x: False)
    flow.connect("first", "second").connect("second", "third")
    result = run_sync(flow.run("input"))
    assert result.ok
    assert len(result.records) == 4
    assert all(r.visits == 1 for r in result.records)


def test_failed_node_retries_on_resume_without_repeating_entry(tmp_path):
    calls = []

    def entry(value, ctx):
        calls.append("entry")
        return value

    def unstable(value, ctx):
        calls.append("unstable")
        if calls.count("unstable") == 1:
            raise ValueError("transient")
        return value + 1

    path = tmp_path / "state.json"
    flow = Workflow(max_visits=2, checkpoint_path=path)
    flow.add("entry", entry).add("unstable", unstable).connect("entry", "unstable")
    assert not run_sync(flow.run(4)).ok
    result = run_sync(flow.resume())
    assert result.ok and result.output == 5
    assert calls == ["entry", "unstable", "unstable"]
    assert next(r for r in result.records if r.name == "unstable").visits == 2
    assert run_sync(flow.resume()).output == 5
    assert calls == ["entry", "unstable", "unstable"]


def test_cycle_interruption_preserves_pending_edge_and_budget(tmp_path):
    path = tmp_path / "state.json"
    flow = Workflow(max_visits=3, checkpoint_path=path)
    flow.add("loop", lambda x, ctx: (x or 0) + 1).connect("loop", "loop")

    async def interrupt():
        events = flow.run_stream(0)
        async for event in events:
            if event.kind == "node_completed":
                await events.aclose()
                break

    run_sync(interrupt())
    state = flow.load_checkpoint(path)
    assert state.edge_firings == {"loop": {"loop": 1}}
    result = run_sync(flow.resume())
    assert result.ok and result.output == 3
    assert result.records[0].visits == 3
    assert run_sync(flow.resume()).output == 3


def test_closing_started_stream_cancels_nodes_and_can_resume(tmp_path):
    path = tmp_path / "state.json"
    flow = Workflow(checkpoint_path=path).add("node", lambda x, ctx: x)

    async def interrupt():
        stream = flow.run_stream("input")
        assert (await anext(stream)).kind == "node_started"
        await stream.aclose()
        assert not [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]

    run_sync(interrupt())
    assert run_sync(flow.resume()).output == "input"


def test_legacy_dag_checkpoint_remains_readable():
    flow = Workflow().add("entry", lambda x, ctx: x)
    flow.add("next", lambda x, ctx: x + 1).connect("entry", "next")
    state = WorkflowState(input=4, outputs={"entry": 4})
    state.record("entry", NodeStatus.COMPLETED)
    assert run_sync(flow.run(state=state)).output == 5


def test_disconnected_nodes_do_not_report_success():
    flow = Workflow().add("entry", lambda x, ctx: x).add("unreachable", lambda x, ctx: x)
    result = run_sync(flow.run())
    assert not result.ok
    assert "stalled" in result.error


def test_concurrent_runs_return_their_own_state():
    async def step(value, ctx):
        await asyncio.sleep(0)
        return value

    flow = Workflow().add("step", step)

    async def execute():
        return await asyncio.gather(flow.run("a"), flow.run("b"))

    assert [r.output for r in run_sync(execute())] == ["a", "b"]


def test_condition_failure_is_checkpointed_and_emits_one_terminal_event(tmp_path):
    def invalid_condition(value):
        raise ValueError("invalid predicate")

    path = tmp_path / "condition.json"
    flow = Workflow(checkpoint_path=path).add("entry", lambda x, ctx: x)
    flow.add("next", lambda x, ctx: x).connect("entry", "next", when=invalid_condition)

    async def collect():
        return [event async for event in flow.run_stream("value")]

    events = run_sync(collect())
    assert sum(e.kind == "workflow_failed" for e in events) == 1
    state = flow.load_checkpoint(path)
    assert "invalid predicate" in state.error
    assert state.records[0].status == NodeStatus.FAILED
    assert state.edge_firings == {}

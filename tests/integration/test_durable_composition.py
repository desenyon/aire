import pytest

from aire.core.content import Message
from aire.core.errors import WorkflowError
from aire.models.base import run_sync
from aire.workflows.graph import Workflow


def test_resume_rejects_changed_graph(tmp_path):
    path = tmp_path / "workflow.json"
    original = Workflow("qa", checkpoint_path=path).add("start", lambda x, c: x)
    assert run_sync(original.run(1)).ok
    changed = (
        Workflow("qa", checkpoint_path=path)
        .add("start", lambda x, c: x)
        .add("extra", lambda x, c: x)
    )
    changed.connect("start", "extra")
    with pytest.raises(WorkflowError, match="graph"):
        run_sync(changed.resume())


def test_long_term_memory_keeps_window_after_restart(tmp_path):
    from aire.memory.store import LongTermMemory

    async def check():
        memory = LongTermMemory(path=tmp_path, window=2)
        for value in ("one", "two", "three"):
            await memory.add(Message.text("user", value))
        restored = LongTermMemory(path=tmp_path, window=2)
        assert [m.text_content for m in await restored.recall()] == ["two", "three"]
        assert await restored.recall(limit=0) == []

    run_sync(check())


def test_concurrent_checkpoint_owners_are_rejected(tmp_path):
    import asyncio

    async def check():
        entered, release = asyncio.Event(), asyncio.Event()

        async def node(x, c):
            entered.set()
            await release.wait()
            return x

        path = tmp_path / "shared.json"
        first = Workflow("qa", checkpoint_path=path).add("node", node)
        second = Workflow("qa", checkpoint_path=path).add("node", node)
        running = asyncio.create_task(first.run(1))
        await entered.wait()
        try:
            with pytest.raises(WorkflowError, match="checkpoint"):
                await asyncio.wait_for(second.run(2), timeout=0.1)
        finally:
            release.set()
            await running

    run_sync(check())


def test_agent_memory_restores_window_and_zero_limit(tmp_path):
    from aire.agents.memory import BufferMemory, JsonlMemory

    with pytest.raises(ValueError):
        BufferMemory(window=0)

    async def check():
        path = tmp_path / "memory.jsonl"
        memory = JsonlMemory(path, window=2)
        for text in ["one", "two", "three"]:
            await memory.add(Message.text("user", text))
        restored = JsonlMemory(path, window=2)
        assert [m.text_content for m in await restored.recall()] == ["two", "three"]
        assert await restored.recall(limit=0) == []

    run_sync(check())

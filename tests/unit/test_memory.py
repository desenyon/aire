"""Long-term memory tests: remember/recall, persistence, consolidation, agent use."""

from __future__ import annotations

import pytest

from aire.core.content import Message
from aire.memory import LongTermMemory, MemoryKind
from aire.models.builtin import HashingEmbedder


@pytest.mark.anyio
async def test_remember_and_semantic_recall() -> None:
    memory = LongTermMemory(embedder=HashingEmbedder())
    await memory.remember("The user prefers dark mode", kind="semantic", salience=2.0)
    await memory.remember("Quantum computing uses qubits", kind="semantic")

    hits = await memory.recall_semantic("what theme does the user like", k=2)
    assert hits
    assert hits[0].kind == MemoryKind.SEMANTIC
    assert hits[0].salience >= 1.0

    kind_filtered = await memory.recall_semantic("qubits", k=3, kind="procedural")
    assert kind_filtered == []


@pytest.mark.anyio
async def test_agent_memory_interface_and_persistence(tmp_path) -> None:
    memory = LongTermMemory(embedder=HashingEmbedder(), path=tmp_path)
    await memory.add(Message.text("user", "hello"))
    await memory.add(Message.text("assistant", "hi there"))
    recalled = await memory.recall()
    assert [m.text_content for m in recalled] == ["hello", "hi there"]

    await memory.remember("Persistent fact", kind="semantic")
    reloaded = LongTermMemory(embedder=HashingEmbedder(), path=tmp_path)
    episodes = await reloaded.recall()
    assert len(episodes) == 2
    hits = await reloaded.recall_semantic("persistent fact", k=1)
    assert hits and hits[0].text == "Persistent fact"

    counts = await reloaded.count()
    assert counts["episodes"] == 2
    assert counts["semantic"] == 1


@pytest.mark.anyio
async def test_consolidate_folds_episodes_into_facts(runtime) -> None:
    from aire.models.builtin import CallableModel

    async def summarize(prompt: str) -> str:
        assert "dark mode" in prompt  # episodes actually reach the model
        return '{"facts": ["User prefers dark mode", "User writes Python"]}'

    memory = LongTermMemory(embedder=HashingEmbedder())
    for i in range(30):
        topic = "dark mode" if i == 0 else f"chatter {i}"
        await memory.add(Message.text("user", topic))

    entries = await memory.consolidate(CallableModel("summarizer", summarize), keep=10)
    assert [e.text for e in entries] == ["User prefers dark mode", "User writes Python"]
    assert all(e.kind == MemoryKind.SEMANTIC for e in entries)
    episodes = await memory.recall()
    assert len(episodes) == 10  # old episodes pruned after consolidation
    hits = await memory.recall_semantic("theme preference", k=1)
    assert hits and "dark mode" in hits[0].text


def test_memory_facade(runtime) -> None:
    from aire.ai import _MemoryNamespace

    ns = _MemoryNamespace(runtime)
    memory = ns.create(embedder=HashingEmbedder())
    assert isinstance(memory, LongTermMemory)
    described = ns.describe()
    assert "episodic" in described["types"]

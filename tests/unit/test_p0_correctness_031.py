"""P0 correctness fixes from the 0.2 audit (shipped as 0.3.1)."""

from __future__ import annotations

from pathlib import Path

from aire.agents.agent import Agent
from aire.agents.memory import BufferMemory
from aire.core.runtime import Runtime
from aire.deployment.gateway import _log_request
from aire.models.builtin import EchoModel
from aire.models.types import GenerationResult, Usage
from aire.observability.otlp import OTLPExporter
from aire.observability.tracing import Tracer
from aire.rag.retriever import Retriever
from aire.rag.store import LocalVectorStore
from tests.conftest import arun


def test_agent_does_not_duplicate_user_in_memory() -> None:
    memory = BufferMemory()
    agent = Agent(EchoModel(), memory=memory, name="t")
    arun(agent.run("hello once"))
    recalled = arun(memory.recall())
    user_turns = [m for m in recalled if m.role == "user"]
    assert len(user_turns) == 1
    assert agent.state.input == "hello once"
    assert agent.state.steps  # state is populated from the run


def test_agent_reset_clears_memory() -> None:
    memory = BufferMemory()
    agent = Agent(EchoModel(), memory=memory)
    arun(agent.run("keep"))
    agent.reset()
    assert arun(memory.recall()) == []
    assert agent.state.input == ""


def test_gateway_audit_ts_is_iso_datetime(tmp_path: Path) -> None:
    entries: list[dict[str, object]] = []

    def log(entry: dict[str, object]) -> None:
        entries.append(entry)

    from aire.core.content import TextContent

    result = GenerationResult(
        content=[TextContent(text="ok")],
        usage=Usage(),
        model="mock:echo",
    )
    _log_request(log, "chat.completions", "mock:echo", "mock:echo", result, 0.0)
    assert "T" in str(entries[0]["ts"])
    assert str(entries[0]["ts"]).endswith("Z")


def test_runtime_aclose_flushes_otlp(monkeypatch) -> None:
    flushed = {"n": 0}

    class FakeExporter(OTLPExporter):
        def __init__(self) -> None:
            super().__init__("http://localhost:9")

        def flush(self) -> None:  # type: ignore[override]
            flushed["n"] += 1

    runtime = Runtime()
    runtime.tracer = Tracer(exporter=FakeExporter())
    arun(runtime.aclose())
    assert flushed["n"] >= 1


def test_hybrid_skips_without_keyword_capability() -> None:
    store = LocalVectorStore()
    # Local has keyword-search — hybrid stays on
    from aire.models.builtin import HashingEmbedder

    retriever = Retriever(store, HashingEmbedder(), hybrid=True)
    assert retriever._store_supports_keyword() is True

    class VectorOnly(LocalVectorStore):
        def describe(self):  # type: ignore[override]
            manifest = super().describe()
            manifest.capabilities = ["vector-search"]
            return manifest

    only = Retriever(VectorOnly(), HashingEmbedder(), hybrid=True)
    assert only._store_supports_keyword() is False

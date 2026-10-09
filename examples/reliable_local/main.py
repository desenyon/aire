"""Offline integration smoke: SQLite updates, agent tools, workflow and file queue."""

from __future__ import annotations

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from aire.agents.agent import Agent
from aire.core.config import Settings
from aire.core.runtime import Runtime
from aire.models.builtin import EchoModel, HashingEmbedder
from aire.models.types import ToolCall
from aire.rag.pipeline import Knowledge
from aire.rag.sqlite import SQLiteVectorStore
from aire.rag.types import Document
from aire.tools.tool import tool
from aire.workers.queue import FileQueueWorker
from aire.workflows.graph import Workflow


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


async def check(directory: Path) -> None:
    async with Runtime(Settings()) as runtime:
        store = SQLiteVectorStore(directory / "knowledge.db")
        try:
            knowledge = Knowledge(runtime, store=store, embedder=HashingEmbedder())
            await knowledge.ingest([Document(id="guide", text="Access uses short lived tokens.")])
            await knowledge.reindex_document(
                Document(id="guide", text="Access uses short lived tokens and role checks.")
            )
            answer = await knowledge.ask("How does access work?", model=EchoModel())
            assert answer.citations
            assert "role checks" in answer.text
            assert await store.count() == 1
        finally:
            await store.aclose()
        reopened = SQLiteVectorStore(directory / "knowledge.db")
        assert await reopened.count() == 1
        await reopened.aclose()

        model = EchoModel()
        model.scripted_tool_calls = [ToolCall(id="add-1", name="add", arguments={"a": 2, "b": 3})]
        agent = Agent(model, tools=[add], session=directory / "session.json")
        result = await agent.run("Use the addition tool.")
        assert result.ok
        assert any(m.role == "tool" and m.text_content == "5" for m in agent.state.messages)
        assert any(m.tool_calls for m in agent.state.messages)

        workflow = Workflow("double", checkpoint_path=directory / "workflow.json")
        workflow.add("double", lambda value, ctx: value * 2)
        worker = FileQueueWorker(directory / "queue", {"double": workflow})
        job = worker.enqueue("double", 21)
        completed = (await worker.drain())[0]
        assert completed.job.id == job.id
        assert completed.job.result == 42
        assert (await workflow.resume()).output == 42
        assert worker.recover() == 0
    print("Offline smoke passed: SQLite persistence, RAG citations, tools, session, queue, resume.")


def main() -> None:
    with TemporaryDirectory(prefix="aire-smoke-") as directory:
        asyncio.run(check(Path(directory)))


if __name__ == "__main__":
    main()

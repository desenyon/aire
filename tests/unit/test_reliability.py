"""Regression cases for failure-safe local operations; no network or credentials."""

import json
from unittest.mock import patch

import pytest

from aire.agents.agent import Agent
from aire.core.content import Message
from aire.core.runtime import Runtime
from aire.core.serialization import write_json_file
from aire.models.base import Model, run_sync
from aire.models.builtin import HashingEmbedder
from aire.models.types import GenerationResult, ModelInfo, ToolCall
from aire.rag.pipeline import Knowledge
from aire.rag.store import LocalVectorStore
from aire.rag.types import Chunk, Document
from aire.workers.queue import FileQueueWorker
from aire.workflows.graph import Workflow
from aire.workflows.types import NodeStatus


def test_agent_history_retains_tool_declarations():
    class CallingModel(Model):
        def __init__(self):
            self.requests = []

        @property
        def info(self):
            return ModelInfo(ref="test:tools", provider="test")

        async def generate(self, request):
            self.requests.append(request.model_copy(deep=True))
            if request.messages[-1].role != "tool":
                return GenerationResult(
                    content=[], tool_calls=[ToolCall(id="call-1", name="missing")]
                )
            return GenerationResult.text_result("done", model="test:tools")

    model = CallingModel()
    agent = Agent(model)
    agent.run_sync("first")
    history = model.requests[1].messages
    assert history[-2].tool_calls == [ToolCall(id="call-1", name="missing")]
    assert history[-1].tool_call_id == "call-1"
    agent.run_sync("second")
    history = model.requests[2].messages
    for i, message in enumerate(history):
        if message.role == "tool":
            assert any(c.id == message.tool_call_id for c in history[i - 1].tool_calls)
    restored = Message.model_validate_json(history[1].model_dump_json())
    assert restored == history[1]


@pytest.mark.parametrize("query", ["", "!!!", "你好"])
def test_tokenless_search_honors_filter(query):
    store = LocalVectorStore()
    run_sync(
        store.upsert(
            [
                Chunk(id="private", text="private", metadata={"tenant": "a"}),
                Chunk(id="public", text="public", metadata={"tenant": "b"}),
            ]
        )
    )
    assert [
        h.chunk.id for h in run_sync(store.search_text(query, k=1, filter={"tenant": "b"}))
    ] == ["public"]


def test_failed_reindex_preserves_existing_document():
    class BrokenEmbedder(HashingEmbedder):
        async def embed(self, request):
            raise RuntimeError("embedding unavailable")

    store = LocalVectorStore()
    old = Chunk(id="old", document_id="doc", text="original", embedding=[1.0])
    run_sync(store.upsert([old]))
    knowledge = Knowledge(Runtime(), store=store, embedder=BrokenEmbedder())
    with pytest.raises(RuntimeError, match="embedding unavailable"):
        run_sync(knowledge.reindex_document(Document(id="doc", text="replacement")))
    assert [h.chunk for h in run_sync(store.search_text(""))] == [old]


def test_failed_json_replace_preserves_previous_snapshot(tmp_path):
    path = tmp_path / "state.json"
    write_json_file(path, {"version": 1})
    with patch("os.replace", side_effect=OSError("disk error")), pytest.raises(OSError):
        write_json_file(path, {"version": 2})
    assert json.loads(path.read_text()) == {"version": 1}
    assert list(tmp_path.iterdir()) == [path]


def test_queue_unknown_workflow_is_a_retained_failure(tmp_path):
    worker = FileQueueWorker(tmp_path)
    job = worker.enqueue("missing")
    result = run_sync(worker.drain())[0]
    assert result.job.id == job.id
    assert result.job.created_at == job.created_at
    assert result.job.status == "failed"
    assert (tmp_path / "failed" / f"{job.id}.json").is_file()


def test_queue_disk_failure_preserves_claim(tmp_path):
    worker = FileQueueWorker(tmp_path, {"ok": Workflow().add("echo", lambda x, ctx: x)})
    job = worker.enqueue("ok", "payload")
    with (
        patch("aire.workers.queue.write_json_file", side_effect=OSError("disk full")),
        pytest.raises(OSError),
    ):
        run_sync(worker.drain())
    assert (tmp_path / "queued" / f"{job.id}.json.claimed").is_file()


def test_workflow_propagates_skipped_branches_to_join():
    workflow = Workflow()
    workflow.add("entry", lambda x, ctx: x)
    # Reverse dependency registration order exposes incomplete skip propagation.
    workflow.add("join", lambda x, ctx: x)
    workflow.add("deep", lambda x, ctx: pytest.fail("unreachable"))
    workflow.add("skipped", lambda x, ctx: pytest.fail("unreachable"))
    workflow.add("taken", lambda x, ctx: "answer")
    workflow.connect("entry", "skipped", when=lambda x: False)
    workflow.connect("skipped", "deep").connect("deep", "join")
    workflow.connect("entry", "taken").connect("taken", "join")
    result = run_sync(workflow.run())
    assert result.ok
    assert result.output == "answer"
    assert {r.name: r.status for r in result.records} == {
        "entry": NodeStatus.COMPLETED,
        "taken": NodeStatus.COMPLETED,
        "skipped": NodeStatus.SKIPPED,
        "deep": NodeStatus.SKIPPED,
        "join": NodeStatus.COMPLETED,
    }


def test_resume_cannot_turn_exhausted_failure_into_success(tmp_path):
    workflow = Workflow(max_visits=1, checkpoint_path=tmp_path / "workflow.json")
    workflow.add("entry", lambda x, ctx: x)

    def fail(x, ctx):
        raise ValueError("failure")

    workflow.add("failure", fail).connect("entry", "failure")
    assert not run_sync(workflow.run()).ok
    assert not run_sync(workflow.resume()).ok

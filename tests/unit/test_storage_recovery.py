"""Storage transactions and queue recovery under injected failures."""

import asyncio
import json
import sqlite3
from unittest.mock import patch

import pytest

from aire.core.errors import RetrievalError
from aire.core.runtime import Runtime
from aire.core.serialization import write_jsonl
from aire.models.base import run_sync
from aire.models.builtin import HashingEmbedder
from aire.models.types import EmbeddingResult
from aire.rag.incremental import IncrementalIndex
from aire.rag.pipeline import Knowledge
from aire.rag.sqlite import SQLiteVectorStore
from aire.rag.store import LocalVectorStore
from aire.rag.types import Chunk, Document
from aire.workers.queue import FileQueueWorker, RedisQueueWorker
from aire.workflows.graph import Workflow


@pytest.fixture(params=["local", "sqlite"])
def store(request, tmp_path):
    value = (
        LocalVectorStore() if request.param == "local" else SQLiteVectorStore(tmp_path / "store.db")
    )
    yield value
    if isinstance(value, SQLiteVectorStore):
        run_sync(value.aclose())


def test_atomic_document_and_full_index_replacement(store):
    knowledge = Knowledge(Runtime(), store=store, embedder=HashingEmbedder())
    original = Document(id="doc", text="original content")
    other = Document(id="other", text="keep other")
    run_sync(knowledge.ingest_documents([original, other]))
    run_sync(knowledge.reindex_document(Document(id="doc", text="new content")))
    assert sorted(h.chunk.text for h in run_sync(store.search_text(""))) == [
        "keep other",
        "new content",
    ]
    run_sync(IncrementalIndex(knowledge).reindex([Document(id="only", text="only replacement")]))
    assert [h.chunk.text for h in run_sync(store.search_text(""))] == ["only replacement"]


@pytest.mark.parametrize("vectors", [[], [[]], [[float("nan")]], [[1.0], [2.0]]])
def test_invalid_embeddings_do_not_destroy_index(store, vectors):
    class BrokenEmbedder(HashingEmbedder):
        async def embed(self, request):
            return EmbeddingResult(vectors=vectors)

    old = Chunk(id="old", document_id="doc", text="old", embedding=[1.0])
    run_sync(store.upsert([old]))
    knowledge = Knowledge(Runtime(), store=store, embedder=BrokenEmbedder())
    with pytest.raises(RetrievalError):
        run_sync(IncrementalIndex(knowledge).reindex([Document(id="doc", text="new")]))
    assert [h.chunk for h in run_sync(store.search_text(""))] == [old]


def test_sqlite_failed_transaction_preserves_cache_and_disk(tmp_path):
    path = tmp_path / "db"
    store = SQLiteVectorStore(path)
    old = Chunk(id="old", document_id="doc", text="old")
    run_sync(store.upsert([old]))
    store._db.executescript(
        "CREATE TRIGGER reject_new BEFORE INSERT ON chunks "
        "BEGIN SELECT RAISE(ABORT, 'reject'); END;"
    )
    with pytest.raises(sqlite3.IntegrityError):
        run_sync(store.replace_document("doc", [Chunk(id="new", document_id="doc", text="new")]))
    assert [h.chunk for h in run_sync(store.search_text(""))] == [old]
    run_sync(store.aclose())
    reopened = SQLiteVectorStore(path)
    assert [h.chunk for h in run_sync(reopened.search_text(""))] == [old]
    run_sync(reopened.aclose())


def test_atomic_jsonl_generator_failure(tmp_path):
    path = tmp_path / "memory.jsonl"
    write_jsonl(path, [{"old": True}])

    def failing():
        yield {"new": True}
        raise RuntimeError("interrupted serialization")

    with pytest.raises(RuntimeError):
        write_jsonl(path, failing())
    assert path.read_text() == '{"old": true}\n'
    assert list(tmp_path.iterdir()) == [path]


def test_file_claim_recovery_and_acknowledged_claim_cleanup(tmp_path):
    calls = []

    def echo(value, ctx):
        calls.append(value)
        return value

    worker = FileQueueWorker(tmp_path, {"echo": Workflow().add("echo", echo)})
    job = worker.enqueue("echo", "payload")
    with (
        patch("aire.workers.queue.write_json_file", side_effect=OSError("disk full")),
        pytest.raises(OSError),
    ):
        run_sync(worker.drain())
    assert worker.recover() == 1
    assert run_sync(worker.drain())[0].job.id == job.id
    assert calls == ["payload", "payload"]  # explicitly at-least-once
    claim = tmp_path / "queued" / f"{job.id}.json.claimed"
    claim.write_text(job.model_dump_json())
    assert worker.recover() == 0
    assert not claim.exists()
    assert run_sync(worker.drain()) == []


def test_cancelled_file_job_is_recoverable(tmp_path):
    async def cancel(value, ctx):
        raise asyncio.CancelledError()

    # Cancellation of the drain itself must retain the claim regardless of node semantics.
    worker = FileQueueWorker(tmp_path, {"cancel": Workflow().add("cancel", cancel)})
    job = worker.enqueue("cancel")
    with (
        patch.object(worker.inner, "submit", side_effect=asyncio.CancelledError()),
        pytest.raises(asyncio.CancelledError),
    ):
        run_sync(worker.drain())
    assert worker.recover() == 1
    assert (tmp_path / "queued" / f"{job.id}.json").exists()


class FakeRedis:
    """Minimal list/Lua contract; does not contact a Redis server."""

    def __init__(self):
        self.lists = {}
        self.results = {}
        self.fail_ack = False

    def lpush(self, key, value):
        self.lists.setdefault(key, []).insert(0, value)

    def rpoplpush(self, source, target):
        if not self.lists.get(source):
            return None
        value = self.lists[source].pop()
        self.lpush(target, value)
        return value

    def brpoplpush(self, source, target, timeout):
        return self.rpoplpush(source, target)

    def eval(self, script, numkeys, results, processing, id, payload, raw):
        if self.fail_ack:
            raise OSError("redis unavailable")
        self.results[id] = json.loads(payload)
        self.lists[processing].remove(raw)

    def llen(self, key):
        return len(self.lists.get(key, []))


def test_redis_acknowledges_only_after_saving_result():
    client = FakeRedis()
    worker = RedisQueueWorker(
        client=client, workflows={"echo": Workflow().add("echo", lambda x, ctx: x)}
    )
    job = worker.enqueue("echo", "payload")
    client.fail_ack = True
    with pytest.raises(OSError):
        run_sync(worker.process_one())
    assert client.llen("aire:jobs:processing") == 1
    assert worker.recover() == 1
    client.fail_ack = False
    result = run_sync(worker.process_one())
    assert result.job.id == job.id and result.job.created_at == job.created_at
    assert client.results[job.id]["result"] == "payload"
    assert client.llen("aire:jobs:processing") == 0


def test_tokenless_acl_filter_and_nonpositive_limits(store):
    run_sync(
        store.upsert(
            [
                Chunk(id="a", text="hidden", metadata={"tenant": "a"}),
                Chunk(id="b", text="visible", metadata={"tenant": "b"}),
            ]
        )
    )
    hits = run_sync(store.search_text("!!!", k=1, filter={"__acl__": {"tenant": "b"}}))
    assert [h.chunk.id for h in hits] == ["b"]
    assert run_sync(store.search_text("", k=-1)) == []
    assert run_sync(store.search([1.0], k=-1)) == []

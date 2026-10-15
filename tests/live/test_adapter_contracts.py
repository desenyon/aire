"""Run actual Aire adapters against disposable local services; opt in via env."""

import os
import uuid

import pytest

from aire.core.config import Settings
from aire.core.runtime import Runtime
from aire.integrations.http import ProviderHttpClient
from aire.models.base import run_sync
from aire.rag.types import Chunk

pytestmark = pytest.mark.live


@pytest.mark.parametrize(
    "backend,variable",
    [
        ("qdrant", "AIRE_LIVE_QDRANT"),
        ("chroma", "AIRE_LIVE_CHROMA"),
        ("weaviate", "AIRE_LIVE_WEAVIATE"),
        ("milvus", "AIRE_LIVE_MILVUS"),
        ("pgvector", "AIRE_LIVE_PGVECTOR"),
    ],
)
def test_real_vector_crud_and_replacement(backend, variable):
    endpoint = os.environ.get(variable)
    if not endpoint:
        pytest.skip(f"set {variable} to enable")

    async def check():
        name = "Qa" + uuid.uuid4().hex[:10]
        async with Runtime(Settings()) as runtime:
            if backend == "pgvector":
                from aire.rag.pgvector import PgVectorStore

                store = PgVectorStore(endpoint, table=name.lower(), dim=2)
            else:
                import importlib

                cls = getattr(
                    importlib.import_module(f"aire.integrations.{backend}"),
                    {
                        "qdrant": "QdrantVectorStore",
                        "chroma": "ChromaVectorStore",
                        "weaviate": "WeaviateVectorStore",
                        "milvus": "MilvusVectorStore",
                    }[backend],
                )
                client = ProviderHttpClient(runtime, backend, base_url=endpoint)
                store = cls(name, client)
            acl_chunks = [
                Chunk(
                    id=f"deny{i}",
                    document_id="acl",
                    text="private",
                    metadata={"tenant": "other"},
                    embedding=[1.0, 0.0],
                )
                for i in range(7)
            ]
            acl_chunks.append(
                Chunk(
                    id="allow",
                    document_id="acl",
                    text="permitted",
                    metadata={"tenant": "qa"},
                    embedding=[0.5, 0.5],
                )
            )
            await store.upsert(acl_chunks)
            allowed = await store.search([1.0, 0.0], k=1, filter={"__acl__": {"tenant": "qa"}})
            assert [h.chunk.id for h in allowed] == ["allow"]
            await store.delete([c.id for c in acl_chunks])
            chunk = Chunk(
                id="one",
                document_id="doc",
                index=3,
                text="token access",
                metadata={"tenant": "qa"},
                embedding=[1.0, 0.0],
            )
            assert await store.upsert([chunk]) == 1
            hits = await store.search([1.0, 0.0], k=5, filter={"tenant": "qa"})
            assert len(hits) == 1
            assert (hits[0].chunk.id, hits[0].chunk.document_id, hits[0].chunk.index) == (
                "one",
                "doc",
                3,
            )
            assert not await store.search([1.0, 0.0], filter={"tenant": "other"})
            await store.replace_document(
                "doc",
                [Chunk(id="two", document_id="doc", text="role access", embedding=[1.0, 0.0])],
            )
            assert await store.count() == 1
            assert [h.chunk.id for h in await store.search([1.0, 0.0])] == ["two"]
            await store.delete(["two"])
            assert await store.count() == 0
            if backend == "pgvector":
                with store._connect() as conn, conn.cursor() as cur:
                    cur.execute(f'DROP TABLE "{name.lower()}"')
            elif backend == "qdrant":
                await client.request("DELETE", f"/collections/{name}")
            elif backend == "chroma":
                await client.request(
                    "DELETE",
                    f"/api/v2/tenants/default_tenant/databases/default_database/collections/{name}",
                )
            elif backend == "weaviate":
                await client.request("DELETE", f"/v1/schema/{store._class}")
            elif backend == "milvus":
                await client.post_json("/v2/vectordb/collections/drop", {"collectionName": name})

    run_sync(check())


def test_real_redis_worker_recovery_and_cache():
    endpoint = os.environ.get("AIRE_LIVE_REDIS")
    if not endpoint:
        pytest.skip("set AIRE_LIVE_REDIS")

    async def check():
        from aire.workers.queue import RedisQueueWorker
        from aire.workflows.graph import Workflow

        prefix = "aire:qa:" + uuid.uuid4().hex
        workflow = Workflow("double").add("double", lambda x, c: x * 2)
        worker = RedisQueueWorker(url=endpoint, key=prefix, workflows={"double": workflow})
        job = worker.enqueue("double", 21)
        result = await worker.process_one(block_seconds=0.1)
        assert result.job.id == job.id
        assert result.job.result == 42
        assert worker.recover() == 0
        import asyncio

        entered, release = asyncio.Event(), asyncio.Event()

        async def interrupted(value, context):
            entered.set()
            await release.wait()
            return value * 2

        recoverable = Workflow("recoverable").add("node", interrupted)
        worker.register("recoverable", recoverable)
        pending = worker.enqueue("recoverable", 22)
        processing = asyncio.create_task(worker.process_one(block_seconds=0.1))
        await asyncio.wait_for(entered.wait(), timeout=5)
        processing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await processing
        assert worker.recover() == 1
        assert worker.recover() == 0
        release.set()
        recovered = await worker.process_one(block_seconds=0.1)
        assert recovered.job.id == pending.id
        assert recovered.job.result == 44
        assert worker.recover() == 0
        from aire.models.builtin import EchoModel
        from aire.models.types import GenerationRequest, ToolCall
        from aire.optimization.redis_cache import RedisCacheBackend, RedisCachedModel

        inner = EchoModel()
        inner.scripted_tool_calls = [ToolCall(id="t", name="read", arguments={"x": 1})]
        backend = RedisCacheBackend(endpoint, prefix=prefix + ":cache:")
        cache = RedisCachedModel(inner, backend)
        request = GenerationRequest.of("hello")
        first = [c async for c in cache.stream(request)]
        second = [c async for c in cache.stream(request)]
        assert [c for part in first for c in part.tool_calls] == [
            c for part in second for c in part.tool_calls
        ]

    run_sync(check())


def test_neo4j_real_triple_roundtrip():
    endpoint = os.environ.get("AIRE_LIVE_NEO4J")
    if not endpoint:
        pytest.skip("set AIRE_LIVE_NEO4J")

    async def check():
        from aire.graph.neo4j_store import Neo4jGraphStore
        from aire.graph.types import ExtractedEntity, ExtractedRelation, Extraction

        store = Neo4jGraphStore(endpoint, user="", password="")
        prefix = "qa_" + uuid.uuid4().hex
        a, b = prefix + "_alice", prefix + "_company"
        try:
            await store.upsert(
                Extraction(
                    entities=[
                        ExtractedEntity(name=a, type="person"),
                        ExtractedEntity(name=b, type="organization"),
                    ],
                    relations=[ExtractedRelation(subject=a, predicate="works_at", object=b)],
                )
            )
            hits = await store.match_entities(a)
            assert any(e.name == a for e in hits)
            graph = await store.neighborhood([a], depth=1)
            assert any(r.subject == a and r.object == b for r in graph.relations)
        finally:
            with store._session() as session:
                session.run(
                    "MATCH (n:Entity) WHERE n.name STARTS WITH $prefix DETACH DELETE n",
                    prefix=prefix,
                ).consume()
            await store.aclose()

    run_sync(check())

import json
import os
import subprocess
import sys

import httpx
import pytest

from aire.core.config import Settings
from aire.core.errors import ProviderError
from aire.core.runtime import Runtime
from aire.integrations.http import ProviderHttpClient
from aire.integrations.qdrant import QdrantVectorStore
from aire.models.base import run_sync
from aire.rag.types import Chunk


def test_qdrant_ids_are_stable_and_delete_matches_upsert():
    code = "from aire.integrations.qdrant import point_id; print(point_id('chunk-one'))"
    values = [
        subprocess.check_output(
            [sys.executable, "-c", code], text=True, env={**os.environ, "PYTHONHASHSEED": seed}
        ).strip()
        for seed in ("1", "2")
    ]
    assert values[0] == values[1]


def test_qdrant_roundtrip_identity_delete_and_pagination():
    async def check():
        stored = {}
        deleted = []

        def handle(request):
            body = json.loads(request.content) if request.content else {}
            if request.method == "GET":
                return httpx.Response(200, json={"result": {"points_count": 2}})
            if request.method == "PUT":
                stored.update({p["id"]: p for p in body["points"]})
                return httpx.Response(200, json={"result": {"status": "completed"}})
            if request.url.path.endswith("scroll"):
                points = list(stored.values())
                idx = 1 if body.get("offset") else 0
                return httpx.Response(
                    200,
                    json={
                        "result": {
                            "points": [points[idx]],
                            "next_page_offset": points[0]["id"] if idx == 0 else None,
                        }
                    },
                )
            deleted.extend(body["points"])
            return httpx.Response(200, json={"result": {"status": "completed"}})

        async with Runtime(Settings()) as runtime:
            client = ProviderHttpClient(runtime, "qdrant", base_url="http://qa")
            await client.raw.aclose()
            client._client = httpx.AsyncClient(
                transport=httpx.MockTransport(handle), base_url="http://qa"
            )
            runtime.resources.track_resource("test-http", client.raw)
            store = QdrantVectorStore("qa", client)
            chunks = [
                Chunk(id=f"one-{i}", document_id="doc", index=i, text="facts", embedding=[1.0, 0.0])
                for i in range(2)
            ]
            await store.upsert(chunks)
            hits = await store.search_text("", k=10)
            assert [(h.chunk.document_id, h.chunk.index) for h in hits] == [("doc", 0), ("doc", 1)]
            await store.delete([chunks[0].id])
            assert deleted == [next(iter(stored))]

    run_sync(check())


@pytest.mark.parametrize("status", [401, 404, 429, 500])
def test_qdrant_delete_does_not_report_http_failure_as_success(status):
    async def check():
        async with Runtime(Settings()) as runtime:
            client = ProviderHttpClient(runtime, "qdrant", base_url="http://qa")
            await client.raw.aclose()
            client._client = httpx.AsyncClient(
                transport=httpx.MockTransport(
                    lambda r: httpx.Response(status, json={"error": "failed"})
                ),
                base_url="http://qa",
            )
            runtime.resources.track_resource("test-http", client.raw)
            with pytest.raises(ProviderError):
                await QdrantVectorStore("qa", client).delete(["one"])

    run_sync(check())


@pytest.mark.parametrize("backend", ["chroma", "milvus", "pinecone", "weaviate"])
def test_remote_adapter_roundtrips_document_identity(backend):
    import importlib

    captured = {}

    async def check():
        def handle(request):
            body = json.loads(request.content) if request.content else {}
            path = request.url.path
            if path.endswith("/upsert") or (path.endswith("/objects") and request.method == "POST"):
                captured.update(body)
                return httpx.Response(200, json=[] if backend == "weaviate" else {"code": 0})
            if backend == "chroma":
                return httpx.Response(
                    200,
                    json={
                        "ids": [captured["ids"]],
                        "documents": [captured["documents"]],
                        "metadatas": [captured["metadatas"]],
                        "distances": [[0.0]],
                    },
                )
            if backend == "milvus":
                return httpx.Response(200, json={"code": 0, "data": captured["data"]})
            if backend == "pinecone":
                return httpx.Response(
                    200,
                    json={
                        "matches": [
                            {
                                "id": "chunk",
                                "score": 1.0,
                                "metadata": captured["vectors"][0]["metadata"],
                            }
                        ]
                    },
                )
            if backend == "weaviate":
                return httpx.Response(
                    200,
                    json={
                        "data": {
                            "Get": {
                                "Qa": [
                                    {
                                        **captured["objects"][0]["properties"],
                                        "_additional": {"distance": 0.0},
                                    }
                                ]
                            }
                        }
                    },
                )
            raise AssertionError(path)

        async with Runtime(Settings()) as runtime:
            client = ProviderHttpClient(runtime, backend, base_url="http://qa")
            await client.raw.aclose()
            client._client = httpx.AsyncClient(
                transport=httpx.MockTransport(handle), base_url="http://qa"
            )
            runtime.resources.track_resource("test-http", client.raw)
            cls = getattr(
                importlib.import_module(f"aire.integrations.{backend}"),
                {
                    "chroma": "ChromaVectorStore",
                    "milvus": "MilvusVectorStore",
                    "pinecone": "PineconeVectorStore",
                    "weaviate": "WeaviateVectorStore",
                }[backend],
            )
            store = cls("qa", client)
            if backend == "chroma":
                store._collection_id = "qa"
            if backend in {"milvus", "weaviate"}:
                store._ensured = True
            chunk = Chunk(
                id="chunk",
                document_id="document",
                index=7,
                text="facts",
                metadata={"tenant": "a"},
                embedding=[1.0, 0.0],
            )
            await store.upsert([chunk])
            hit = (await store.search([1.0, 0.0]))[0].chunk
            assert (hit.id, hit.document_id, hit.index, hit.metadata) == (
                "chunk",
                "document",
                7,
                {"tenant": "a"},
            )

    run_sync(check())


@pytest.mark.parametrize("backend", ["chroma", "milvus", "pinecone", "weaviate"])
@pytest.mark.parametrize("status", [401, 404, 429, 500])
def test_all_remote_deletes_raise_on_http_errors(backend, status):
    import importlib

    async def check():
        async with Runtime(Settings()) as runtime:
            client = ProviderHttpClient(runtime, backend, base_url="http://qa")
            await client.raw.aclose()
            client._client = httpx.AsyncClient(
                transport=httpx.MockTransport(
                    lambda r: httpx.Response(status, json={"error": "failure"})
                ),
                base_url="http://qa",
            )
            runtime.resources.track_resource("test-http", client.raw)
            cls = getattr(
                importlib.import_module(f"aire.integrations.{backend}"),
                {
                    "chroma": "ChromaVectorStore",
                    "milvus": "MilvusVectorStore",
                    "pinecone": "PineconeVectorStore",
                    "weaviate": "WeaviateVectorStore",
                }[backend],
            )
            store = cls("qa", client)
            if backend == "chroma":
                store._collection_id = "qa"
            if backend == "weaviate":
                store._ensured = True
            with pytest.raises(ProviderError):
                await store.delete(["one"])

    run_sync(check())


def test_incomplete_document_listing_never_publishes_replacement():
    from aire.rag.store import VectorStore

    class Incomplete(VectorStore):
        written = False

        async def upsert(self, chunks):
            self.written = True
            return len(chunks)

        async def search(self, *a, **kw):
            return []

        async def search_text(self, *a, **kw):
            return []

        async def delete(self, ids):
            raise AssertionError("must not delete")

        async def count(self):
            return 2

    async def check():
        from aire.core.errors import RetrievalError

        store = Incomplete()
        with pytest.raises(RetrievalError):
            await store.replace_document("doc", [Chunk(id="new", document_id="doc", text="new")])
        assert not store.written

    run_sync(check())

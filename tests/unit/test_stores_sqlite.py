"""Embedded SQLite vector store + hosted REST adapter wiring tests."""

from __future__ import annotations

import pytest

from aire.core.errors import ConfigurationError
from aire.rag.sqlite import SQLiteVectorStore
from aire.rag.types import Chunk


def _chunks() -> list[Chunk]:
    return [
        Chunk(
            id=f"c{i}",
            text=text,
            metadata={"source": "test"},
            embedding=[float(i), 1.0, 0.0],
        )
        for i, text in enumerate(
            ["vector databases store embeddings", "graph databases store relations"]
        )
    ]


@pytest.mark.anyio
async def test_sqlite_vector_store_roundtrip(tmp_path) -> None:
    store = SQLiteVectorStore(tmp_path / "v.db")
    assert await store.upsert(_chunks()) == 2
    assert await store.count() == 2

    hits = await store.search([0.0, 1.0, 0.0], k=1)
    assert hits and hits[0].chunk.text == "vector databases store embeddings"

    keyword = await store.search_text("relations", k=1)
    assert keyword and keyword[0].chunk.id == "c1"

    assert await store.delete(["c1"]) == 1
    assert await store.count() == 1


@pytest.mark.anyio
async def test_sqlite_vector_store_persists(tmp_path) -> None:
    path = tmp_path / "v.db"
    store = SQLiteVectorStore(path)
    await store.upsert(_chunks())

    reloaded = SQLiteVectorStore(path)
    assert await reloaded.count() == 2
    assert (await reloaded.search_text("embeddings", k=1))[0].chunk.id == "c0"
    described = reloaded.describe()
    assert described.provider == "sqlite"
    assert "embedded-persistence" in described.capabilities


@pytest.mark.anyio
async def test_sqlite_registered_with_local(runtime) -> None:
    """The sqlite store ships with the builtin `local` registration."""
    assert runtime.vector_stores.has("sqlite")
    store = runtime.vector_stores.create("sqlite", name=":memory:", runtime=runtime)
    assert isinstance(store, SQLiteVectorStore)


def test_hosted_adapters_register_via_hint(runtime) -> None:
    from aire.ai import _RagNamespace

    ns = _RagNamespace(runtime)
    # Pinecone requires an index host.
    with pytest.raises(ConfigurationError):
        ns.vector_store("pinecone:my-index")

    pinecone = ns.vector_store("pinecone:my-index", base_url="https://idx.svc.pinecone.io")
    assert pinecone.describe().provider == "pinecone"

    weaviate = ns.vector_store("weaviate:Docs")
    assert weaviate.describe().provider == "weaviate"
    assert "keyword-search" in weaviate.describe().capabilities

    milvus = ns.vector_store("milvus:docs")
    assert milvus.describe().provider == "milvus"

    stores = ns.describe()["stores"]
    for provider in ("pinecone", "weaviate", "milvus", "sqlite"):
        assert provider in stores

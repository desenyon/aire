"""Vector stores, retrieval, reranking and the Knowledge pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from aire.core.errors import RetrievalError
from aire.core.runtime import Runtime
from aire.models.builtin import HashingEmbedder
from aire.rag import (
    Chunk,
    Knowledge,
    LexicalOverlapReranker,
    LocalVectorStore,
    Retriever,
    cosine_similarity,
)
from tests.conftest import arun


def _embedded_chunks(embedder: HashingEmbedder, texts: list[str]) -> list[Chunk]:
    vectors = arun(embedder.embed_texts(texts))
    return [
        Chunk(text=t, embedding=v, metadata={"source": f"doc-{i}"})
        for i, (t, v) in enumerate(zip(texts, vectors, strict=True))
    ]


def test_cosine_similarity() -> None:
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine_similarity([], []) == 0.0


def test_local_store_search(embedder: HashingEmbedder) -> None:
    store = LocalVectorStore()
    chunks = _embedded_chunks(
        embedder, ["refund within 30 days", "oauth2 tokens", "encryption at rest"]
    )
    assert arun(store.upsert(chunks)) == 3
    assert arun(store.count()) == 3
    query = arun(embedder.embed_one("what is the refund window"))
    hits = arun(store.search(query, k=2))
    assert len(hits) == 2
    assert hits[0].score >= hits[1].score


def test_local_store_keyword_search(embedder: HashingEmbedder) -> None:
    store = LocalVectorStore()
    arun(store.upsert(_embedded_chunks(embedder, ["refund policy details", "unrelated content"])))
    hits = arun(store.search_text("refund", k=5))
    assert hits and "refund" in hits[0].chunk.text


def test_local_store_metadata_filter(embedder: HashingEmbedder) -> None:
    store = LocalVectorStore()
    chunks = _embedded_chunks(embedder, ["alpha", "beta"])
    chunks[0].metadata["tenant"] = "a"
    chunks[1].metadata["tenant"] = "b"
    arun(store.upsert(chunks))
    query = arun(embedder.embed_one("alpha beta"))
    hits = arun(store.search(query, k=5, filter={"tenant": "b"}))
    assert len(hits) == 1
    assert hits[0].chunk.metadata["tenant"] == "b"


def test_local_store_persistence(tmp_path: Path, embedder: HashingEmbedder) -> None:
    path = tmp_path / "store.json"
    store = LocalVectorStore(path)
    arun(store.upsert(_embedded_chunks(embedder, ["persistent chunk"])))
    store.save()
    reloaded = LocalVectorStore(path)
    assert arun(reloaded.count()) == 1


def test_local_store_delete(embedder: HashingEmbedder) -> None:
    store = LocalVectorStore()
    chunks = _embedded_chunks(embedder, ["x", "y"])
    arun(store.upsert(chunks))
    assert arun(store.delete([chunks[0].id])) == 1
    assert arun(store.count()) == 1


def test_hybrid_retriever_fuses(embedder: HashingEmbedder) -> None:
    store = LocalVectorStore()
    arun(
        store.upsert(
            _embedded_chunks(
                embedder,
                [
                    "the hyperdrive matrix operates at 42 teraflops",
                    "banana smoothie recipes for beginners",
                ],
            )
        )
    )
    retriever = Retriever(store, embedder, hybrid=True)
    hits = arun(retriever.retrieve("hyperdrive matrix", k=1))
    assert "hyperdrive" in hits[0].chunk.text


def test_reranker_boosts_overlap() -> None:
    hits = [
        Chunk(text="the cat sat").model_dump(),
    ]
    del hits  # use real objects below
    from aire.rag.types import ScoredChunk

    scored = [
        ScoredChunk(chunk=Chunk(text="unrelated words entirely"), score=0.9),
        ScoredChunk(chunk=Chunk(text="refund policy within thirty days"), score=0.1),
    ]
    reranked = arun(LexicalOverlapReranker().rerank("refund policy", scored, k=2))
    assert "refund" in reranked[0].chunk.text


def test_knowledge_end_to_end(runtime: Runtime, docs: list[str]) -> None:
    knowledge = Knowledge(runtime)
    report = arun(knowledge.ingest(docs))
    assert report.chunks >= len(docs)
    answer = arun(knowledge.ask("How long is the refund window?", model="mock:echo"))
    assert answer.retrieved > 0
    assert answer.citations
    assert any("refund" in c.excerpt.lower() for c in answer.citations)


def test_knowledge_empty_ingestion_raises(runtime: Runtime) -> None:
    from aire.rag import Document

    with pytest.raises(RetrievalError):
        arun(Knowledge(runtime).ingest_documents([Document(text="")]))


def test_knowledge_describe(runtime: Runtime) -> None:
    manifest = Knowledge(runtime).describe()
    assert manifest["kind"] == "knowledge"
    assert manifest["store"]["provider"] == "local"

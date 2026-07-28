"""Contract tests: every provider/store/embedder satisfies the same interface.

When adding a provider, add it to the parametrized lists and it inherits the
whole contract suite.
"""

from __future__ import annotations

from typing import Any

import pytest

from aire.core.types import HealthStatus, Manifest
from aire.models.base import EmbeddingModel, Model
from aire.models.builtin import CallableModel, EchoModel, HashingEmbedder
from aire.models.types import GenerationRequest, GenerationResult
from aire.rag.store import LocalVectorStore, VectorStore
from aire.rag.types import Chunk
from tests.conftest import arun

MODELS: list[Any] = [
    EchoModel(),
    EchoModel("prefixed", prefix=">> "),
    CallableModel("identity", lambda prompt: prompt),
]

EMBEDDERS: list[Any] = [
    HashingEmbedder(),
    HashingEmbedder(dimension=64),
]

STORES: list[Any] = [
    LocalVectorStore(),
]


@pytest.mark.parametrize("model", MODELS, ids=lambda m: m.info.ref)
class TestModelContract:
    def test_info_is_normalized(self, model: Model) -> None:
        info = model.info
        assert info.ref and info.provider
        assert isinstance(info.capabilities, list)

    def test_generate_returns_result(self, model: Model) -> None:
        result = arun(model.generate(GenerationRequest.of("contract probe")))
        assert isinstance(result, GenerationResult)
        assert isinstance(result.text, str)
        assert result.usage.input_tokens >= 0

    def test_stream_yields_chunks(self, model: Model) -> None:
        async def _collect() -> list[str]:
            return [c.text async for c in model.stream(GenerationRequest.of("stream probe"))]

        texts = arun(_collect())
        assert texts and all(isinstance(t, str) for t in texts)

    def test_health_returns_status(self, model: Model) -> None:
        status = arun(model.health())
        assert isinstance(status, HealthStatus)
        assert status.ok

    def test_describe_manifest(self, model: Model) -> None:
        manifest = model.describe()
        assert isinstance(manifest, Manifest)
        assert manifest.kind == "model"


@pytest.mark.parametrize("embedder", EMBEDDERS, ids=lambda e: e.name)
class TestEmbedderContract:
    def test_dimension_consistent(self, embedder: EmbeddingModel) -> None:
        vectors = arun(embedder.embed_texts(["a", "b", "c"]))
        assert len(vectors) == 3
        assert all(len(v) == embedder.dimension for v in vectors)

    def test_describe_manifest(self, embedder: EmbeddingModel) -> None:
        assert embedder.describe().kind == "embedder"


@pytest.mark.parametrize("store", STORES, ids=lambda s: type(s).__name__)
class TestVectorStoreContract:
    def test_upsert_search_delete(self, store: VectorStore, embedder: Any = None) -> None:
        embedder = HashingEmbedder()
        vectors = arun(embedder.embed_texts(["contract text one", "contract text two"]))
        chunks = [
            Chunk(text=t, embedding=v)
            for t, v in zip(["contract text one", "contract text two"], vectors, strict=True)
        ]
        assert arun(store.upsert(chunks)) == 2
        assert arun(store.count()) == 2
        hits = arun(store.search(vectors[0], k=1))
        assert len(hits) == 1
        assert arun(store.delete([chunks[0].id])) == 1
        assert arun(store.count()) == 1

    def test_health(self, store: VectorStore) -> None:
        assert arun(store.health()).ok

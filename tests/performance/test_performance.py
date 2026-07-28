"""Performance gates: import time, embedding throughput, search latency."""

from __future__ import annotations

import subprocess
import sys
import time

import pytest

from aire.models.builtin import HashingEmbedder
from aire.rag.store import LocalVectorStore
from aire.rag.types import Chunk
from tests.conftest import arun


@pytest.mark.performance()
def test_import_time_under_one_second() -> None:
    """`import aire` must stay fast — no heavy deps at import time."""
    started = time.perf_counter()
    subprocess.run(
        [sys.executable, "-c", "import aire"], check=True, capture_output=True, timeout=30
    )
    elapsed = time.perf_counter() - started
    assert elapsed < 1.5, f"cold import took {elapsed:.2f}s (includes interpreter startup)"


@pytest.mark.performance()
def test_no_torch_or_numpy_imported_by_core() -> None:
    code = (
        "import sys, aire; "
        "heavy = [m for m in ('torch', 'tensorflow', 'transformers') if m in sys.modules]; "
        "assert not heavy, heavy"
    )
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, timeout=30)


@pytest.mark.performance()
def test_embedding_throughput() -> None:
    embedder = HashingEmbedder()
    texts = [f"document number {i} with some representative content" for i in range(500)]
    started = time.perf_counter()
    vectors = arun(embedder.embed_texts(texts))
    elapsed = time.perf_counter() - started
    assert len(vectors) == 500
    assert elapsed < 2.0, f"500 embeddings took {elapsed:.2f}s"


@pytest.mark.performance()
def test_local_store_search_latency() -> None:
    embedder = HashingEmbedder()
    store = LocalVectorStore()
    texts = [f"chunk {i} about topic {i % 25}" for i in range(1000)]
    vectors = arun(embedder.embed_texts(texts))
    chunks = [Chunk(text=t, embedding=v) for t, v in zip(texts, vectors, strict=True)]
    arun(store.upsert(chunks))
    query = arun(embedder.embed_one("topic 7"))
    started = time.perf_counter()
    for _ in range(10):
        hits = arun(store.search(query, k=5))
    elapsed = (time.perf_counter() - started) / 10
    assert len(hits) == 5
    assert elapsed < 0.2, f"search over 1000 chunks took {elapsed * 1000:.1f}ms"


@pytest.mark.performance()
def test_workflow_overhead() -> None:
    from aire.workflows import Workflow

    wf = Workflow("perf")
    for i in range(20):
        wf.add(f"n{i}", lambda x, ctx: x + 1)
        if i:
            wf.connect(f"n{i - 1}", f"n{i}")
    started = time.perf_counter()
    result = arun(wf.run(0))
    elapsed = time.perf_counter() - started
    assert result.output == 20
    assert elapsed < 0.5, f"20-node workflow took {elapsed:.2f}s"

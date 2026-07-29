"""Knowledge pipeline offline with mock + hashing."""

from __future__ import annotations

from aire import AI
from aire.models.base import run_sync
from aire.rag.types import Document


def test_knowledge_index_and_ask_offline() -> None:
    embedder = AI.models.embedder_sync("local:hashing")
    kb = AI.rag.create(embedder=embedder)
    report = run_sync(
        kb.ingest(
            [
                Document(
                    text="Authentication uses API keys stored in environment variables.",
                    metadata={"source": "auth.md"},
                ),
                Document(
                    text="aire supports offline mock:echo models for local development.",
                    metadata={"source": "dev.md"},
                ),
            ]
        )
    )
    assert report.chunks >= 1
    answer = run_sync(kb.ask("How does authentication work?", model="mock:echo", k=2))
    assert answer.text
    assert answer.model.startswith("mock:")
    assert answer.retrieved >= 1

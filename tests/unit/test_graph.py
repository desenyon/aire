"""Graph subsystem tests: embedded store, extractors, GraphRAG pipeline."""

from __future__ import annotations

import pytest

from aire.core.runtime import Runtime
from aire.graph import (
    Extraction,
    KnowledgeGraph,
    LexicalGraphExtractor,
    SQLiteGraphStore,
)


def _extraction() -> Extraction:
    return Extraction.model_validate(
        {
            "entities": [
                {"name": "Ada Lovelace", "type": "person"},
                {"name": "Charles Babbage", "type": "person"},
                {"name": "Analytical Engine", "type": "machine"},
            ],
            "relations": [
                {
                    "subject": "Ada Lovelace",
                    "predicate": "worked_with",
                    "object": "Charles Babbage",
                },
                {
                    "subject": "Charles Babbage",
                    "predicate": "designed",
                    "object": "Analytical Engine",
                },
            ],
        }
    )


@pytest.mark.anyio
async def test_sqlite_graph_store_roundtrip() -> None:
    store = SQLiteGraphStore()
    added_e, added_r = await store.upsert(_extraction(), chunk_id="chk_1")
    assert added_e == 3
    assert added_r == 2
    counts = await store.count()
    assert counts == {"entities": 3, "relations": 2}

    matches = await store.match_entities("Who is Ada Lovelace?")
    assert matches and matches[0].name == "Ada Lovelace"

    neighborhood = await store.neighborhood(["Ada Lovelace"], depth=2)
    names = {e.name for e in neighborhood.entities}
    assert "Charles Babbage" in names
    assert "Analytical Engine" in names  # depth 2 traversal
    text = neighborhood.as_context()
    assert "Ada Lovelace —worked_with→ Charles Babbage" in text


@pytest.mark.anyio
async def test_sqlite_graph_store_merge_and_clear() -> None:
    store = SQLiteGraphStore()
    await store.upsert(_extraction())
    added_e, _ = await store.upsert(_extraction())  # re-ingest: no duplicates
    assert added_e == 0
    assert (await store.count())["entities"] == 3
    await store.clear()
    assert (await store.count())["entities"] == 0


@pytest.mark.anyio
async def test_lexical_extractor_finds_entities_and_relations() -> None:
    extractor = LexicalGraphExtractor()
    result = await extractor.extract(
        "Ada Lovelace worked with Charles Babbage. The Analytical Engine was his design."
    )
    names = {e.name for e in result.entities}
    assert "Ada Lovelace" in names
    assert "Charles Babbage" in names
    pairs = {(r.subject, r.object) for r in result.relations}
    assert ("Ada Lovelace", "Charles Babbage") in pairs
    assert all(r.predicate == "related_to" for r in result.relations)


@pytest.mark.anyio
async def test_knowledge_graph_end_to_end_lexical(runtime: Runtime) -> None:
    graph = KnowledgeGraph(runtime)
    report = await graph.ingest(
        [
            "Ada Lovelace wrote notes on the Analytical Engine designed by Charles Babbage.",
            "Charles Babbage designed the Analytical Engine in London.",
        ]
    )
    assert report.chunks >= 2
    assert report.entities > 0
    assert report.extractor == "lexical"

    subgraph = await graph.subgraph("What did Ada Lovelace write about?")
    assert subgraph.relations, "expected graph facts for a matched entity"

    answer = await graph.query("What did Ada Lovelace write about?")
    assert answer.text  # mock:echo answers offline
    assert answer.retrieved > 0
    assert answer.citations  # grounded in indexed chunks

    manifest = graph.describe()
    assert manifest["kind"] == "knowledge_graph"
    assert manifest["extractor"]["type"] == "lexical"


@pytest.mark.anyio
async def test_graph_facade(runtime: Runtime) -> None:
    from aire.ai import _GraphNamespace

    ns = _GraphNamespace(runtime)
    store = ns.store("sqlite:memory")
    assert isinstance(store, SQLiteGraphStore)
    graph = ns.create(store=store)
    assert isinstance(graph, KnowledgeGraph)
    described = ns.describe()
    assert "sqlite" in described["stores"]

"""Graph store offline: SQLiteGraphStore (Neo4j needs AIRE_LIVE_NEO4J / aire[neo4j])."""

from __future__ import annotations

import tempfile
from pathlib import Path

from aire.graph.store import SQLiteGraphStore
from aire.graph.types import ExtractedEntity, ExtractedRelation, Extraction
from aire.models.base import run_sync


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "graph.db"
        store = SQLiteGraphStore(path)
        extraction = Extraction(
            entities=[
                ExtractedEntity(name="Paris", type="place"),
                ExtractedEntity(name="France", type="place"),
            ],
            relations=[
                ExtractedRelation(subject="Paris", predicate="capital_of", object="France"),
            ],
        )
        n_ent, n_rel = run_sync(store.upsert(extraction, chunk_id="c1"))
        print("upserted:", n_ent, "entities,", n_rel, "relations")
        print("store:", store.describe().model_dump(mode="json"))
        print(
            "Neo4j live: set AIRE_LIVE_NEO4J and install aire[neo4j]; "
            "see tests/live/test_live_stores.py"
        )


if __name__ == "__main__":
    main()

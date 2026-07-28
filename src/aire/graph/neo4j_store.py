"""Lazy Neo4j GraphStore adapter (``aire[neo4j]``).

Uses the official ``neo4j`` driver when installed; raises
:class:`~aire.core.errors.ConfigurationError` with an install hint otherwise.
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING, Any

from aire.core.errors import ConfigurationError
from aire.core.types import Manifest
from aire.graph.store import GraphStore
from aire.graph.types import (
    Entity,
    ExtractedEntity,
    ExtractedRelation,
    Extraction,
    Relation,
    Subgraph,
)

if TYPE_CHECKING:
    from aire.core.runtime import Runtime


def _require_neo4j() -> Any:
    if importlib.util.find_spec("neo4j") is None:
        raise ConfigurationError(
            "neo4j is required for Neo4jGraphStore: pip install 'aire[neo4j]'",
            code="graph.neo4j_missing",
            context={"extra": "aire[neo4j]", "package": "neo4j"},
        )
    import neo4j  # type: ignore[import-not-found]

    return neo4j


class Neo4jGraphStore(GraphStore):
    """GraphStore backed by Neo4j (Bolt). Entities are nodes; relations are edges."""

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        *,
        user: str = "neo4j",
        password: str = "",
        database: str = "neo4j",
        name: str = "neo4j",
        driver: Any | None = None,
    ) -> None:
        neo4j = _require_neo4j()
        self._uri = uri
        self._user = user
        self._database = database
        self._name = name
        if driver is not None:
            self._driver = driver
        else:
            auth = (user, password) if password or user else None
            self._driver = neo4j.GraphDatabase.driver(uri, auth=auth)

    def _session(self) -> Any:
        return self._driver.session(database=self._database)

    async def upsert(self, extraction: Extraction, *, chunk_id: str = "") -> tuple[int, int]:
        entities_added = 0
        relations_added = 0
        with self._session() as session:
            for e in extraction.entities:
                entity = Entity(name=e.name, type=e.type)
                result = session.run(
                    """
                    MERGE (n:Entity {key: $key})
                    ON CREATE SET n.name = $name, n.type = $type, n._created = true
                    ON MATCH SET n._created = false
                    RETURN n._created AS created
                    """,
                    key=entity.key,
                    name=entity.name,
                    type=entity.type,
                )
                record = result.single()
                if record and record["created"]:
                    entities_added += 1
            for r in extraction.relations:
                relation = Relation(
                    subject=r.subject.strip(),
                    predicate=r.predicate.strip(),
                    object=r.object.strip(),
                    chunk_id=chunk_id,
                )
                for name in (relation.subject, relation.object):
                    key = name.strip().lower()
                    session.run(
                        "MERGE (n:Entity {key: $key}) "
                        "ON CREATE SET n.name = $name, n.type = 'entity'",
                        key=key,
                        name=name.strip(),
                    )
                session.run(
                    """
                    MATCH (a:Entity {key: $sk}), (b:Entity {key: $ok})
                    MERGE (a)-[r:REL {id: $id}]->(b)
                    SET r.predicate = $pred, r.weight = $weight, r.chunk_id = $chunk_id
                    """,
                    sk=relation.subject.strip().lower(),
                    ok=relation.object.strip().lower(),
                    id=relation.id,
                    pred=relation.predicate,
                    weight=relation.weight,
                    chunk_id=chunk_id,
                )
                relations_added += 1
        return entities_added, relations_added

    async def add_relation(self, relation: Relation) -> None:
        extraction = Extraction(
            entities=[
                ExtractedEntity(name=relation.subject),
                ExtractedEntity(name=relation.object),
            ],
            relations=[
                ExtractedRelation(
                    subject=relation.subject,
                    predicate=relation.predicate,
                    object=relation.object,
                )
            ],
        )
        await self.upsert(extraction, chunk_id=relation.chunk_id)

    async def neighborhood(self, names: list[str], *, depth: int = 1) -> Subgraph:
        keys = [n.strip().lower() for n in names if n.strip()]
        if not keys:
            return Subgraph()
        with self._session() as session:
            result = session.run(
                """
                MATCH (seed:Entity)
                WHERE seed.key IN $keys
                CALL {
                    WITH seed
                    MATCH path = (seed)-[*1..$depth]-(n:Entity)
                    RETURN nodes(path) AS ns, relationships(path) AS rs
                }
                UNWIND ns AS node
                UNWIND rs AS rel
                RETURN collect(DISTINCT node) AS nodes, collect(DISTINCT rel) AS rels
                """,
                keys=keys,
                depth=max(1, depth),
            )
            record = result.single()
            if not record:
                return Subgraph()
            entities = [
                Entity(name=n.get("name", n.get("key", "")), type=n.get("type", "entity"))
                for n in record["nodes"] or []
            ]
            relations: list[Relation] = []
            for rel in record["rels"] or []:
                start = rel.start_node
                end = rel.end_node
                relations.append(
                    Relation(
                        subject=start.get("name", start.get("key", "")),
                        predicate=rel.get("predicate", rel.type),
                        object=end.get("name", end.get("key", "")),
                        weight=float(rel.get("weight", 1.0) or 1.0),
                        chunk_id=str(rel.get("chunk_id", "") or ""),
                    )
                )
            return Subgraph(entities=entities, relations=relations)

    async def match_entities(self, query: str, *, limit: int = 8) -> list[Entity]:
        tokens = [t.lower() for t in query.split() if t.strip()]
        if not tokens:
            return []
        with self._session() as session:
            result = session.run(
                """
                MATCH (n:Entity)
                WHERE any(t IN $tokens WHERE toLower(n.name) CONTAINS t)
                RETURN n LIMIT $limit
                """,
                tokens=tokens,
                limit=limit,
            )
            return [
                Entity(name=r["n"].get("name", ""), type=r["n"].get("type", "entity"))
                for r in result
            ]

    async def entities(self, *, limit: int = 1000) -> list[Entity]:
        with self._session() as session:
            result = session.run(
                "MATCH (n:Entity) RETURN n ORDER BY n.name LIMIT $limit", limit=limit
            )
            return [
                Entity(name=r["n"].get("name", ""), type=r["n"].get("type", "entity"))
                for r in result
            ]

    async def relations(self, *, limit: int = 5000) -> list[Relation]:
        with self._session() as session:
            result = session.run(
                """
                MATCH (a:Entity)-[r:REL]->(b:Entity)
                RETURN a, r, b LIMIT $limit
                """,
                limit=limit,
            )
            return [
                Relation(
                    subject=rec["a"].get("name", ""),
                    predicate=rec["r"].get("predicate", "related"),
                    object=rec["b"].get("name", ""),
                    weight=float(rec["r"].get("weight", 1.0) or 1.0),
                    chunk_id=str(rec["r"].get("chunk_id", "") or ""),
                )
                for rec in result
            ]

    async def count(self) -> dict[str, int]:
        with self._session() as session:
            e = session.run("MATCH (n:Entity) RETURN count(n) AS c").single()["c"]
            r = session.run("MATCH ()-[r:REL]->() RETURN count(r) AS c").single()["c"]
            return {"entities": int(e), "relations": int(r)}

    async def clear(self) -> None:
        with self._session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    async def aclose(self) -> None:
        self._driver.close()

    def describe(self) -> Manifest:
        return Manifest(
            kind="graph_store",
            name=self._name,
            provider="neo4j",
            capabilities=["triples", "bfs-neighborhood", "entity-matching", "persistence"],
            extra={"uri": self._uri, "database": self._database},
        )


def register(runtime: Runtime) -> None:
    """Register ``neo4j:`` graph store factory (lazy import of driver)."""

    def _factory(name: str = "neo4j", *, runtime: Runtime, **options: Any) -> GraphStore:
        uri = options.pop("uri", None) or options.pop("path", None) or "bolt://localhost:7687"
        return Neo4jGraphStore(uri, name=name, **options)

    runtime.registry("graph_store").register("neo4j", _factory, replace=True)


def describe() -> dict[str, Any]:
    available = importlib.util.find_spec("neo4j") is not None
    return {
        "kind": "neo4j_graph_store",
        "available": available,
        "install": "pip install 'aire[neo4j]'",
        "ref": "neo4j:<name>",
    }

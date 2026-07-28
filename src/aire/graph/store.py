"""Graph store interface and the embedded SQLite implementation.

``sqlite:<path>`` (or ``sqlite::memory:``) is the aire-native graph store:
single-file, transactional, zero dependencies (stdlib ``sqlite3``), no server.
Swap in a Neo4j or other adapter through the same :class:`GraphStore`
interface via a plugin.
"""

from __future__ import annotations

import abc
import asyncio
import json
import sqlite3
import threading
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aire.core.types import HealthStatus, Manifest
from aire.graph.types import Entity, Extraction, Relation, Subgraph

if TYPE_CHECKING:
    from aire.core.runtime import Runtime


class GraphStore(abc.ABC):
    """Interface every graph store adapter implements."""

    @abc.abstractmethod
    async def upsert(self, extraction: Extraction, *, chunk_id: str = "") -> tuple[int, int]:
        """Merge an extraction into the graph; returns (entities, relations) added."""

    @abc.abstractmethod
    async def add_relation(self, relation: Relation) -> None: ...

    @abc.abstractmethod
    async def neighborhood(self, names: list[str], *, depth: int = 1) -> Subgraph:
        """BFS neighborhood around the given entity names."""

    @abc.abstractmethod
    async def match_entities(self, query: str, *, limit: int = 8) -> list[Entity]:
        """Find entities mentioned by a free-text query (token overlap)."""

    @abc.abstractmethod
    async def entities(self, *, limit: int = 1000) -> list[Entity]: ...

    @abc.abstractmethod
    async def relations(self, *, limit: int = 5000) -> list[Relation]: ...

    @abc.abstractmethod
    async def count(self) -> dict[str, int]: ...

    @abc.abstractmethod
    async def clear(self) -> None: ...

    async def health(self) -> HealthStatus:
        try:
            await self.count()
        except Exception as exc:
            return HealthStatus.unhealthy(f"{type(exc).__name__}: {exc}")
        return HealthStatus.healthy()

    def describe(self) -> Manifest:
        return Manifest(kind="graph_store", name=type(self).__name__)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS entities (
    key TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT NOT NULL DEFAULT 'entity',
    properties TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS relations (
    id TEXT PRIMARY KEY,
    subject_key TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object_key TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 1.0,
    chunk_id TEXT NOT NULL DEFAULT '',
    properties TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_rel_subject ON relations(subject_key);
CREATE INDEX IF NOT EXISTS idx_rel_object ON relations(object_key);
"""


class SQLiteGraphStore(GraphStore):
    """Embedded, transactional graph store backed by stdlib sqlite3."""

    def __init__(self, path: str | Path = ":memory:", *, name: str = "sqlite") -> None:
        self._path = str(path)
        self._name = name
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self._path, check_same_thread=False)
        self._db.executescript(_SCHEMA)
        self._db.commit()

    # -- synchronous internals (called via asyncio.to_thread) -----------------------

    def _upsert_sync(self, extraction: Extraction, chunk_id: str) -> tuple[int, int]:
        with self._lock, self._db:
            entities_added = 0
            for e in extraction.entities:
                entity = Entity(name=e.name, type=e.type)
                cursor = self._db.execute(
                    "INSERT OR IGNORE INTO entities(key, name, type) VALUES (?, ?, ?)",
                    (entity.key, entity.name, entity.type),
                )
                entities_added += cursor.rowcount
            relations_added = 0
            for r in extraction.relations:
                relation = Relation(
                    subject=r.subject.strip(),
                    predicate=r.predicate.strip(),
                    object=r.object.strip(),
                    chunk_id=chunk_id,
                )
                for name in (relation.subject, relation.object):
                    self._db.execute(
                        "INSERT OR IGNORE INTO entities(key, name) VALUES (?, ?)",
                        (name.lower(), name),
                    )
                cursor = self._db.execute(
                    "INSERT OR REPLACE INTO relations"
                    "(id, subject_key, predicate, object_key, weight, chunk_id, properties)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        relation.id,
                        relation.subject.lower(),
                        relation.predicate,
                        relation.object.lower(),
                        relation.weight,
                        relation.chunk_id,
                        json.dumps(relation.properties),
                    ),
                )
                relations_added += cursor.rowcount
        return entities_added, relations_added

    def _entity_from_row(self, row: tuple[Any, ...]) -> Entity:
        return Entity(
            name=row[1],
            type=row[2],
            properties=json.loads(row[3]) if row[3] else {},
        )

    def _relation_from_row(self, row: tuple[Any, ...]) -> Relation:
        return Relation(
            id=row[0],
            subject=row[1],
            predicate=row[2],
            object=row[3],
            weight=row[4],
            chunk_id=row[5],
            properties=json.loads(row[6]) if row[6] else {},
        )

    def _names_by_keys(self, keys: set[str]) -> dict[str, str]:
        if not keys:
            return {}
        marks = ",".join("?" for _ in keys)
        rows = self._db.execute(
            f"SELECT key, name FROM entities WHERE key IN ({marks})",  # noqa: S608 (parameterized)
            tuple(keys),
        ).fetchall()
        return {row[0]: row[1] for row in rows}

    def _neighborhood_sync(self, names: list[str], depth: int) -> Subgraph:
        with self._lock:
            frontier = deque((name.strip().lower(), 0) for name in names)
            seen: set[str] = set()
            relation_rows: dict[str, tuple[Any, ...]] = {}
            while frontier:
                key, level = frontier.popleft()
                if key in seen or level > depth:
                    continue
                seen.add(key)
                if level == depth:
                    continue
                rows = self._db.execute(
                    "SELECT id, subject_key, predicate, object_key, weight, chunk_id, properties"
                    " FROM relations WHERE subject_key = ? OR object_key = ?",
                    (key, key),
                ).fetchall()
                for row in rows:
                    relation_rows[row[0]] = row
                    frontier.append((row[1], level + 1))
                    frontier.append((row[3], level + 1))
            display = self._names_by_keys(seen)
            entities = [
                Entity(name=name, type=self._type_for(key)) for key, name in sorted(display.items())
            ]
            relations = [
                Relation(
                    id=row[0],
                    subject=display.get(row[1], row[1]),
                    predicate=row[2],
                    object=display.get(row[3], row[3]),
                    weight=row[4],
                    chunk_id=row[5],
                    properties=json.loads(row[6]) if row[6] else {},
                )
                for row in relation_rows.values()
            ]
        return Subgraph(entities=entities, relations=relations)

    def _type_for(self, key: str) -> str:
        row = self._db.execute("SELECT type FROM entities WHERE key = ?", (key,)).fetchone()
        return row[0] if row else "entity"

    # -- interface ----------------------------------------------------------------------

    async def upsert(self, extraction: Extraction, *, chunk_id: str = "") -> tuple[int, int]:
        return await asyncio.to_thread(self._upsert_sync, extraction, chunk_id)

    async def add_relation(self, relation: Relation) -> None:
        from aire.graph.types import ExtractedRelation

        extraction = Extraction(
            relations=[
                ExtractedRelation(
                    subject=relation.subject,
                    predicate=relation.predicate,
                    object=relation.object,
                )
            ]
        )
        await asyncio.to_thread(self._upsert_sync, extraction, relation.chunk_id)

    async def neighborhood(self, names: list[str], *, depth: int = 1) -> Subgraph:
        return await asyncio.to_thread(self._neighborhood_sync, names, depth)

    async def match_entities(self, query: str, *, limit: int = 8) -> list[Entity]:
        from aire.rag.store import tokenize

        terms = set(tokenize(query))

        def _match() -> list[Entity]:
            with self._lock:
                rows = self._db.execute(
                    "SELECT key, name, type, properties FROM entities"
                ).fetchall()
            scored: list[tuple[int, Entity]] = []
            for row in rows:
                name_terms = set(tokenize(row[1]))
                overlap = len(terms & name_terms)
                if overlap:
                    scored.append((overlap, self._entity_from_row(row)))
            scored.sort(key=lambda pair: pair[0], reverse=True)
            return [entity for _, entity in scored[:limit]]

        return await asyncio.to_thread(_match)

    async def entities(self, *, limit: int = 1000) -> list[Entity]:
        def _all() -> list[Entity]:
            with self._lock:
                rows = self._db.execute(
                    "SELECT key, name, type, properties FROM entities ORDER BY name LIMIT ?",
                    (limit,),
                ).fetchall()
            return [self._entity_from_row(row) for row in rows]

        return await asyncio.to_thread(_all)

    async def relations(self, *, limit: int = 5000) -> list[Relation]:
        def _all() -> list[Relation]:
            with self._lock:
                rows = self._db.execute(
                    "SELECT id, subject_key, predicate, object_key, weight, chunk_id, properties"
                    " FROM relations LIMIT ?",
                    (limit,),
                ).fetchall()
            return [self._relation_from_row(row) for row in rows]

        return await asyncio.to_thread(_all)

    async def count(self) -> dict[str, int]:
        def _count() -> dict[str, int]:
            with self._lock:
                entities = self._db.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
                relations = self._db.execute("SELECT COUNT(*) FROM relations").fetchone()[0]
            return {"entities": entities, "relations": relations}

        return await asyncio.to_thread(_count)

    async def clear(self) -> None:
        def _clear() -> None:
            with self._lock, self._db:
                self._db.execute("DELETE FROM relations")
                self._db.execute("DELETE FROM entities")

        await asyncio.to_thread(_clear)

    async def aclose(self) -> None:
        await asyncio.to_thread(self._db.close)

    def describe(self) -> Manifest:
        return Manifest(
            kind="graph_store",
            name=self._name,
            provider="sqlite",
            capabilities=["triples", "bfs-neighborhood", "entity-matching", "persistence"],
            extra={"path": self._path},
        )


def register(runtime: Runtime) -> None:
    """Register the embedded graph store factory on a runtime."""

    def _factory(name: str = "graph", *, runtime: Runtime, **options: Any) -> GraphStore:
        path = options.pop("path", None) or (None if name in {"memory", ":memory:"} else name)
        return SQLiteGraphStore(path or ":memory:", **options)

    runtime.registry("graph_store").register("sqlite", _factory, replace=True)

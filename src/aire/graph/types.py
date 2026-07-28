"""Knowledge graph primitives: entities, relations, extractions, subgraphs."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from aire.core.types import new_id


class Entity(BaseModel):
    """A named thing in the graph (person, org, concept, ...)."""

    id: str = Field(default_factory=lambda: new_id("ent"))
    name: str
    type: str = "entity"
    properties: dict[str, Any] = Field(default_factory=dict)

    @property
    def key(self) -> str:
        """Canonical lookup key (case-insensitive name)."""
        return self.name.strip().lower()


class Relation(BaseModel):
    """A directed, typed edge between two entity names."""

    id: str = Field(default_factory=lambda: new_id("rel"))
    subject: str
    predicate: str
    object: str
    weight: float = 1.0
    chunk_id: str = ""
    properties: dict[str, Any] = Field(default_factory=dict)

    def as_text(self) -> str:
        """Human/agent-readable rendering used in grounded prompts."""
        return f"{self.subject} —{self.predicate}→ {self.object}"


class ExtractedEntity(BaseModel):
    """Model-facing extraction record (what an extractor emits)."""

    name: str
    type: str = "entity"


class ExtractedRelation(BaseModel):
    """Model-facing extracted edge, referencing entity names."""

    subject: str
    predicate: str
    object: str


class Extraction(BaseModel):
    """The structured-output schema models are asked to produce per chunk."""

    entities: list[ExtractedEntity] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)


class Subgraph(BaseModel):
    """A neighborhood slice of the graph — the unit used for grounding."""

    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)

    def as_context(self) -> str:
        """Render relations as context lines for a grounding prompt."""
        lines = [r.as_text() for r in self.relations]
        return "\n".join(dict.fromkeys(lines))

    def describe(self) -> dict[str, Any]:
        return {
            "entities": len(self.entities),
            "relations": len(self.relations),
            "predicates": sorted({r.predicate for r in self.relations}),
        }


class GraphIndexReport(BaseModel):
    """Outcome of graph ingestion."""

    documents: int
    chunks: int
    entities: int
    relations: int
    store: str
    extractor: str
    duration_ms: float = 0.0

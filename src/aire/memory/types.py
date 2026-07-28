"""Long-term memory primitives."""

from __future__ import annotations

import time
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from aire.core.types import new_id


class MemoryKind(StrEnum):
    EPISODIC = "episodic"  # what happened (raw conversation events)
    SEMANTIC = "semantic"  # what is known (facts, preferences, summaries)
    PROCEDURAL = "procedural"  # how to do things (successful plans)


class MemoryEntry(BaseModel):
    """One durable memory with salience-weighted recall."""

    id: str = Field(default_factory=lambda: new_id("mem"))
    kind: MemoryKind = MemoryKind.SEMANTIC
    text: str
    salience: float = 1.0
    created_at: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def describe(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": str(self.kind),
            "salience": self.salience,
            "chars": len(self.text),
            "metadata": self.metadata,
        }

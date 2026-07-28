"""Data primitives: records, lineage and quality reports."""

from __future__ import annotations

import hashlib
from typing import Any

from pydantic import BaseModel, Field

from aire.core.types import new_id


class Record(BaseModel):
    """One normalized data row flowing through pipelines."""

    id: str = Field(default_factory=lambda: new_id("rec"))
    text: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    def fingerprint(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()[:16]


class LineageEntry(BaseModel):
    """One step in a dataset's provenance chain."""

    operation: str
    detail: dict[str, Any] = Field(default_factory=dict)


class QualityReport(BaseModel):
    """Summary of dataset health checks."""

    total: int
    empty: int = 0
    duplicate: int = 0
    avg_length: float = 0.0
    min_length: int = 0
    max_length: int = 0
    pii_suspects: int = 0
    issues: list[str] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues


class DatasetSplit(BaseModel):
    """Named, reproducible partitions of a dataset."""

    train_count: int
    validation_count: int
    test_count: int
    seed: int


class DatasetInfo(BaseModel):
    """Machine-readable dataset manifest."""

    name: str
    count: int
    version: str
    source: str | None = None
    lineage: list[LineageEntry] = Field(default_factory=list)

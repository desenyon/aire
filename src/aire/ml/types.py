"""ML primitives: tasks, features, predictions, fit reports."""

from __future__ import annotations

import time
from enum import StrEnum

from pydantic import BaseModel, Field

from aire.core.types import new_id
from aire.data.types import Record


class TaskType(StrEnum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class FeatureVector(BaseModel):
    """Named numeric features extracted from one record."""

    names: list[str]
    values: list[float]

    def as_dict(self) -> dict[str, float]:
        return dict(zip(self.names, self.values, strict=True))


def extract_features(record: Record) -> dict[str, float]:
    """Feature convention shared by every estimator.

    1. ``record.metadata["features"]`` — explicit numeric feature dict.
    2. Numeric values in ``record.metadata`` (excluding reserved keys).
    3. Text-derived fallback: length, token count, vocabulary richness.
    """
    explicit = record.metadata.get("features")
    if isinstance(explicit, dict):
        return {str(k): float(v) for k, v in explicit.items()}
    reserved = {"expected", "context", "source", "filename", "start", "end"}
    numeric = {
        str(k): float(v)
        for k, v in record.metadata.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool) and k not in reserved
    }
    if numeric:
        return numeric
    from aire.rag.store import tokenize

    tokens = tokenize(record.text)
    return {
        "char_count": float(len(record.text)),
        "token_count": float(len(tokens)),
        "vocab_richness": len(set(tokens)) / len(tokens) if tokens else 0.0,
        "avg_token_len": sum(len(t) for t in tokens) / len(tokens) if tokens else 0.0,
    }


def vectorize(rows: list[dict[str, float]]) -> tuple[list[str], list[list[float]]]:
    """Align feature dicts onto a stable union of names (missing → 0.0)."""
    names = sorted({name for row in rows for name in row})
    matrix = [[row.get(name, 0.0) for name in names] for row in rows]
    return names, matrix


class Prediction(BaseModel):
    """One model prediction with optional per-class probabilities."""

    id: str = Field(default_factory=lambda: new_id("prd"))
    record_id: str = ""
    value: float | str
    probabilities: dict[str, float] = Field(default_factory=dict)
    model: str = "unknown"


class FitReport(BaseModel):
    """Outcome of fitting an estimator."""

    backend: str
    task: str
    samples: int
    features: int
    metrics: dict[str, float] = Field(default_factory=dict)
    feature_names: list[str] = Field(default_factory=list)
    duration_s: float = 0.0
    created_at: float = Field(default_factory=time.time)

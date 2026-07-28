"""Polars bridge — Dataset ⇄ polars DataFrame (lazy optional)."""

from __future__ import annotations

import importlib.util
from typing import Any

from aire.core.errors import ConfigurationError
from aire.data.dataset import Dataset
from aire.data.types import Record
from aire.ml.types import Prediction, extract_features


def _require_polars() -> Any:
    if importlib.util.find_spec("polars") is None:
        raise ConfigurationError(
            "polars is required: pip install 'aire[polars]'",
            code="ml.polars_missing",
        )
    import polars  # type: ignore[import-not-found]

    return polars


def frame_to_dataset(
    frame: Any,
    *,
    target: str | None = "label",
    text_column: str | None = None,
    name: str = "polars",
) -> Dataset:
    """polars DataFrame → aire Dataset (feature columns → metadata['features'])."""
    pl = _require_polars()
    if not isinstance(frame, pl.DataFrame):
        raise ConfigurationError("expected a polars DataFrame", code="ml.polars_type")
    records: list[Record] = []
    columns = list(frame.columns)
    feature_cols = [
        c
        for c in columns
        if c != target and c != text_column and frame[c].dtype.is_numeric()
    ]
    for i, row in enumerate(frame.iter_rows(named=True)):
        features = {c: float(row[c]) for c in feature_cols if row[c] is not None}
        meta: dict[str, Any] = {"features": features}
        if target and target in row and row[target] is not None:
            meta[target] = row[target]
        text = str(row[text_column]) if text_column and row.get(text_column) is not None else ""
        records.append(Record(id=f"{name}-{i}", text=text, metadata=meta))
    return Dataset(name=name, records=records)


def dataset_to_frame(dataset: Dataset, *, target: str | None = None) -> Any:
    """Dataset → polars DataFrame."""
    pl = _require_polars()
    rows: list[dict[str, Any]] = []
    for record in dataset:
        feats = extract_features(record)
        row: dict[str, Any] = dict(feats)
        if target and target in record.metadata:
            row[target] = record.metadata[target]
        row["_id"] = record.id
        rows.append(row)
    return pl.DataFrame(rows)


def predictions_to_frame(predictions: list[Prediction]) -> Any:
    pl = _require_polars()
    return pl.DataFrame(
        [
            {
                "record_id": p.record_id,
                "value": p.value,
                "model": p.model,
                **{f"p_{k}": v for k, v in p.probabilities.items()},
            }
            for p in predictions
        ]
    )

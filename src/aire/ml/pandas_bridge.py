"""pandas bridge: aire Datasets ⇄ DataFrames, lazily imported.

The bridge is how aire plugs into the existing ML ecosystem: pull a DataFrame
from anywhere (SQL, parquet, Spark export, ...), turn it into a Dataset for
training/evaluation, and push predictions back out as a DataFrame.
Requires ``pip install aire[ml]`` (pandas).
"""

from __future__ import annotations

import importlib.util
from typing import Any

from aire.core.errors import ConfigurationError
from aire.data.dataset import Dataset
from aire.data.types import Record
from aire.ml.types import Prediction


def _require_pandas() -> Any:
    if importlib.util.find_spec("pandas") is None:
        raise ConfigurationError(
            "pandas is required for the DataFrame bridge: pip install 'aire[ml]'",
            code="ml.pandas_missing",
            context={"backend": "pandas"},
        )
    import pandas  # type: ignore[import-untyped]

    return pandas


def frame_to_dataset(
    frame: Any,
    *,
    text_field: str | None = None,
    target: str | None = None,
    feature_columns: list[str] | None = None,
    name: str = "pandas",
) -> Dataset:
    """Convert a pandas DataFrame into an aire Dataset.

    Numeric columns become record features (the shared estimator convention);
    ``text_field`` becomes the record text; ``target`` is copied into record
    metadata for fitting.
    """
    pandas = _require_pandas()
    if not isinstance(frame, pandas.DataFrame):
        raise ConfigurationError(
            "frame_to_dataset expects a pandas DataFrame",
            code="ml.frame_invalid",
            context={"type": type(frame).__name__},
        )
    numeric = [
        str(c)
        for c in (feature_columns or frame.columns)
        if str(c) not in {text_field, target} and pandas.api.types.is_numeric_dtype(frame[c])
    ]
    records: list[Record] = []
    for _, row in frame.iterrows():
        metadata: dict[str, Any] = {"features": {col: float(row[col]) for col in numeric}}
        if target is not None:
            metadata[target] = row[target].item() if hasattr(row[target], "item") else row[target]
        text = str(row[text_field]) if text_field else ""
        records.append(Record(text=text, metadata=metadata))
    return Dataset(records, name=name, source="pandas")


def dataset_to_frame(dataset: Dataset, *, target: str | None = None) -> Any:
    """Convert a Dataset into a pandas DataFrame (features as columns)."""
    pandas = _require_pandas()
    from aire.ml.types import extract_features

    rows: list[dict[str, Any]] = []
    for record in dataset:
        row: dict[str, Any] = {"id": record.id, "text": record.text}
        row.update(extract_features(record))
        if target is not None:
            row[target] = record.metadata.get(target)
        rows.append(row)
    return pandas.DataFrame(rows)


def predictions_to_frame(predictions: list[Prediction]) -> Any:
    """Convert predictions into a DataFrame (value + probabilities columns)."""
    pandas = _require_pandas()
    rows = [
        {
            "record_id": p.record_id,
            "value": p.value,
            "model": p.model,
            **{f"p_{label}": prob for label, prob in p.probabilities.items()},
        }
        for p in predictions
    ]
    return pandas.DataFrame(rows)


def available_backends() -> dict[str, bool]:
    """Which ML backends are importable right now (for manifests/doctor)."""
    return {
        "native": True,
        "sklearn": importlib.util.find_spec("sklearn") is not None,
        "torch": importlib.util.find_spec("torch") is not None,
        "keras": importlib.util.find_spec("keras") is not None,
        "xgboost": importlib.util.find_spec("xgboost") is not None,
        "lightgbm": importlib.util.find_spec("lightgbm") is not None,
        "pandas": importlib.util.find_spec("pandas") is not None,
    }

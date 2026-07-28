"""The Estimator: the universal aire contract for creatable ML models.

Every backend — native, scikit-learn, torch, or a plugin — implements this
interface. Fit from an aire :class:`Dataset`, predict on records, evaluate,
persist, and describe. All blocking compute runs in worker threads so the
interface stays async end to end.
"""

from __future__ import annotations

import abc
import asyncio
import json
import time
from pathlib import Path
from typing import Any

from aire.core.errors import ConfigurationError
from aire.core.types import Manifest
from aire.data.dataset import Dataset
from aire.data.types import Record
from aire.ml.types import FitReport, Prediction, TaskType, extract_features, vectorize


class Estimator(abc.ABC):
    """Interface every creatable ML model implements."""

    task: TaskType = TaskType.CLASSIFICATION

    def __init__(self) -> None:
        self.feature_names: list[str] = []
        self.report: FitReport | None = None

    # -- backend hooks ------------------------------------------------------------

    @abc.abstractmethod
    def _fit_sync(self, x: list[list[float]], y: list[Any]) -> dict[str, float]:
        """Fit on an aligned feature matrix + targets; returns training metrics."""

    @abc.abstractmethod
    def _predict_sync(self, x: list[list[float]]) -> list[float | str]:
        """Predict one value per row."""

    def _probabilities_sync(self, x: list[list[float]]) -> list[dict[str, float]] | None:
        return None  # backends that support probabilities override

    @abc.abstractmethod
    def _state(self) -> dict[str, Any]:
        """Serializable fitted parameters."""

    @abc.abstractmethod
    def _restore(self, state: dict[str, Any]) -> None:
        """Restore fitted parameters."""

    # -- shared pipeline --------------------------------------------------------------

    def _prepare(
        self, dataset: Dataset, target: str
    ) -> tuple[list[list[float]], list[Any], list[str]]:
        rows = [extract_features(record) for record in dataset]
        names, x = vectorize(rows)
        y: list[Any] = []
        ids: list[str] = []
        for record in dataset:
            if target not in record.metadata:
                raise ConfigurationError(
                    f"record {record.id} is missing target field {target!r}",
                    code="ml.target_missing",
                    context={"target": target, "record": record.id},
                )
            y.append(record.metadata[target])
            ids.append(record.id)
        if not x:
            raise ConfigurationError("cannot fit on an empty dataset", code="ml.empty_dataset")
        self.feature_names = names
        return x, y, ids

    async def fit(self, dataset: Dataset, *, target: str = "label") -> FitReport:
        """Fit the estimator on a dataset (records carry features + target)."""
        started = time.time()
        x, y, _ = self._prepare(dataset, target)
        metrics = await asyncio.to_thread(self._fit_sync, x, y)
        self.report = FitReport(
            backend=self.backend_name(),
            task=str(self.task),
            samples=len(x),
            features=len(self.feature_names),
            metrics=metrics,
            feature_names=list(self.feature_names),
            duration_s=time.time() - started,
        )
        return self.report

    async def predict(self, inputs: Dataset | list[Record]) -> list[Prediction]:
        """Predict for records (same feature convention as fit)."""
        if self.report is None:
            raise ConfigurationError(
                "estimator is not fitted yet (call fit() or load())",
                code="ml.not_fitted",
            )
        records = list(inputs if isinstance(inputs, list) else list(inputs))
        rows = [extract_features(record) for record in records]
        x = [[row.get(name, 0.0) for name in self.feature_names] for row in rows]
        values = await asyncio.to_thread(self._predict_sync, x)
        probabilities = await asyncio.to_thread(self._probabilities_sync, x)
        return [
            Prediction(
                record_id=record.id,
                value=value,
                probabilities=probabilities[i] if probabilities else {},
                model=self.describe().name,
            )
            for i, (record, value) in enumerate(zip(records, values, strict=True))
        ]

    async def evaluate(self, dataset: Dataset, *, target: str = "label") -> dict[str, float]:
        """Score against labeled records: accuracy (classification) or RMSE/MAE."""
        predictions = await self.predict(list(dataset))
        records = list(dataset)
        truth = [record.metadata.get(target) for record in records]
        if self.task == TaskType.CLASSIFICATION:
            correct = sum(
                1 for p, t in zip(predictions, truth, strict=True) if str(p.value) == str(t)
            )
            return {"accuracy": correct / len(truth) if truth else 0.0, "samples": len(truth)}
        errors = [
            float(p.value) - float(t)  # type: ignore[arg-type]
            for p, t in zip(predictions, truth, strict=True)
        ]
        mae = sum(abs(e) for e in errors) / len(errors) if errors else 0.0
        rmse = (sum(e * e for e in errors) / len(errors)) ** 0.5 if errors else 0.0
        return {"mae": mae, "rmse": rmse, "samples": len(truth)}

    # -- persistence -----------------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """Persist the fitted estimator as portable JSON."""
        if self.report is None:
            raise ConfigurationError("nothing to save: estimator not fitted", code="ml.not_fitted")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "backend": self.backend_name(),
            "task": str(self.task),
            "feature_names": self.feature_names,
            "state": self._state(),
            "report": self.report.model_dump(mode="json"),
        }
        target.write_text(json.dumps(payload, indent=2))
        return target

    def load(self, path: str | Path) -> Estimator:
        """Restore a previously saved estimator (in place; returns self)."""
        payload = json.loads(Path(path).read_text())
        self.feature_names = list(payload["feature_names"])
        self._restore(payload["state"])
        self.report = FitReport.model_validate(payload["report"])
        return self

    # -- introspection -----------------------------------------------------------------

    def backend_name(self) -> str:
        return type(self).__name__

    def describe(self) -> Manifest:
        return Manifest(
            kind="estimator",
            name=self.backend_name(),
            capabilities=[str(self.task), "fit", "predict", "evaluate", "persistence"],
            extra={
                "fitted": self.report is not None,
                "features": len(self.feature_names),
            },
        )

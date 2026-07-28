"""Transform contract: fit/transform stages that compose into Pipelines.

Mirrors the sklearn transformer idea without requiring sklearn. Native
transforms work offline; sklearn transformers wrap lazily.
"""

from __future__ import annotations

import abc
import json
from pathlib import Path
from typing import Any

from aire.core.errors import ConfigurationError
from aire.core.types import Manifest
from aire.data.dataset import Dataset
from aire.ml.types import extract_features, vectorize


class Transform(abc.ABC):
    """One preprocessing stage: fit on a matrix, transform matrices."""

    name: str = "transform"

    def __init__(self) -> None:
        self.feature_names: list[str] = []
        self.fitted: bool = False

    @abc.abstractmethod
    def _fit(self, x: list[list[float]], y: list[Any] | None = None) -> None: ...

    @abc.abstractmethod
    def _transform(self, x: list[list[float]]) -> list[list[float]]: ...

    def fit_matrix(self, x: list[list[float]], y: list[Any] | None = None) -> Transform:
        self._fit(x, y)
        self.fitted = True
        return self

    def transform_matrix(self, x: list[list[float]]) -> list[list[float]]:
        if not self.fitted:
            raise ConfigurationError(
                f"transform {self.name!r} is not fitted", code="ml.not_fitted"
            )
        return self._transform(x)

    def fit_transform_matrix(
        self, x: list[list[float]], y: list[Any] | None = None
    ) -> list[list[float]]:
        return self.fit_matrix(x, y).transform_matrix(x)

    def fit_dataset(self, dataset: Dataset, *, target: str | None = None) -> Transform:
        rows = [extract_features(record) for record in dataset]
        names, x = vectorize(rows)
        self.feature_names = names
        y: list[Any] | None = None
        if target is not None:
            y = [record.metadata.get(target) for record in dataset]
        return self.fit_matrix(x, y)

    def transform_dataset(self, dataset: Dataset) -> list[list[float]]:
        rows = [extract_features(record) for record in dataset]
        x = [[row.get(name, 0.0) for name in self.feature_names] for row in rows]
        return self.transform_matrix(x)

    def describe(self) -> Manifest:
        return Manifest(
            kind="transform",
            name=self.name,
            capabilities=["fit", "transform"],
            extra={"fitted": self.fitted, "features": len(self.feature_names)},
        )


class StandardScaler(Transform):
    """Zero-mean unit-variance scaler (native, offline)."""

    name = "standard_scaler"

    def __init__(self, *, with_mean: bool = True, with_std: bool = True) -> None:
        super().__init__()
        self.with_mean = with_mean
        self.with_std = with_std
        self.means: list[float] = []
        self.stds: list[float] = []

    def _fit(self, x: list[list[float]], y: list[Any] | None = None) -> None:
        n_features = len(x[0]) if x else 0
        n = len(x) or 1
        self.means = [sum(row[i] for row in x) / n for i in range(n_features)]
        self.stds = []
        for i in range(n_features):
            var = sum((row[i] - self.means[i]) ** 2 for row in x) / n
            self.stds.append(var**0.5 if var > 0 else 1.0)

    def _transform(self, x: list[list[float]]) -> list[list[float]]:
        out: list[list[float]] = []
        for row in x:
            new = []
            for i, value in enumerate(row):
                v = value - self.means[i] if self.with_mean else value
                if self.with_std:
                    v = v / self.stds[i]
                new.append(v)
            out.append(new)
        return out

    def state(self) -> dict[str, Any]:
        return {
            "means": self.means,
            "stds": self.stds,
            "with_mean": self.with_mean,
            "with_std": self.with_std,
            "feature_names": self.feature_names,
        }

    def restore(self, state: dict[str, Any]) -> StandardScaler:
        self.means = list(state["means"])
        self.stds = list(state["stds"])
        self.with_mean = bool(state["with_mean"])
        self.with_std = bool(state["with_std"])
        self.feature_names = list(state.get("feature_names", []))
        self.fitted = True
        return self


class MinMaxScaler(Transform):
    """Scale features to [0, 1] (native, offline)."""

    name = "minmax_scaler"

    def __init__(self) -> None:
        super().__init__()
        self.mins: list[float] = []
        self.spans: list[float] = []

    def _fit(self, x: list[list[float]], y: list[Any] | None = None) -> None:
        n_features = len(x[0]) if x else 0
        self.mins = [min(row[i] for row in x) for i in range(n_features)]
        maxs = [max(row[i] for row in x) for i in range(n_features)]
        self.spans = [m - mn if m != mn else 1.0 for mn, m in zip(self.mins, maxs, strict=True)]

    def _transform(self, x: list[list[float]]) -> list[list[float]]:
        return [
            [(row[i] - self.mins[i]) / self.spans[i] for i in range(len(row))] for row in x
        ]

    def state(self) -> dict[str, Any]:
        return {"mins": self.mins, "spans": self.spans, "feature_names": self.feature_names}

    def restore(self, state: dict[str, Any]) -> MinMaxScaler:
        self.mins = list(state["mins"])
        self.spans = list(state["spans"])
        self.feature_names = list(state.get("feature_names", []))
        self.fitted = True
        return self


class SklearnTransform(Transform):
    """Wrap any sklearn transformer (``StandardScaler``, ``PCA``, …)."""

    def __init__(self, name: str = "standard_scaler", **options: Any) -> None:
        super().__init__()
        from aire.ml.sklearn_adapter import resolve_sklearn_transformer

        cls = resolve_sklearn_transformer(name)
        self.name = f"sklearn:{name}"
        self._model = cls(**options)

    def _fit(self, x: list[list[float]], y: list[Any] | None = None) -> None:
        import inspect

        params = inspect.signature(self._model.fit).parameters
        accepts_y = "y" in params or any(
            p.kind is inspect.Parameter.VAR_POSITIONAL for p in params.values()
        )
        if y is not None and accepts_y:
            try:
                self._model.fit(x, y)
                return
            except TypeError:
                pass
        self._model.fit(x)

    def _transform(self, x: list[list[float]]) -> list[list[float]]:
        result = self._model.transform(x)
        return [list(map(float, row)) for row in result]


NATIVE_TRANSFORMS: dict[str, type[Transform]] = {
    "standard_scaler": StandardScaler,
    "minmax_scaler": MinMaxScaler,
}


def create_transform(spec: str, **options: Any) -> Transform:
    """Create a transform from ``native:name``, ``sklearn:name``, or bare native name."""
    if ":" not in spec:
        provider, name = "native", spec
    else:
        provider, _, name = spec.partition(":")
    if provider in ("native", "simple"):
        if name == "column_transformer":
            from aire.ml.compose import ColumnTransformer

            return ColumnTransformer(**options)
        if name == "feature_union":
            from aire.ml.compose import FeatureUnion

            return FeatureUnion(**options)
        try:
            return NATIVE_TRANSFORMS[name](**options)
        except KeyError:
            raise ConfigurationError(
                f"unknown native transform {name!r}",
                code="ml.transform_unknown",
                context={
                    "available": sorted(
                        [*NATIVE_TRANSFORMS, "column_transformer", "feature_union"]
                    )
                },
            ) from None
    if provider == "sklearn":
        return SklearnTransform(name, **options)
    raise ConfigurationError(
        f"unknown transform provider {provider!r}",
        code="ml.transform_provider",
        context={"hint": "use native:* or sklearn:*"},
    )


def save_transform(transform: Transform, path: str | Path) -> Path:
    if not transform.fitted:
        raise ConfigurationError("nothing to save: transform not fitted", code="ml.not_fitted")
    state_fn = getattr(transform, "state", None)
    if not callable(state_fn):
        raise ConfigurationError(
            f"transform {transform.name!r} cannot be serialized by aire",
            code="ml.persistence_delegated",
        )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"name": transform.name, "state": state_fn()}
    target.write_text(json.dumps(payload, indent=2))
    return target

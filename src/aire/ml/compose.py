"""Composable sklearn-style transforms: ColumnTransformer + FeatureUnion."""

from __future__ import annotations

from typing import Any

from aire.core.errors import ConfigurationError
from aire.ml.transform import Transform, create_transform


class ColumnTransformer(Transform):
    """Apply different transforms to named feature subsets, then concatenate.

    ``transformers`` is a list of ``(name, transform_spec|Transform, columns)``
    where ``columns`` is a list of feature names (resolved after first fit from
    input feature_names) or integer indices.
    """

    name = "column_transformer"

    def __init__(
        self,
        transformers: list[tuple[str, str | Transform, list[str | int]]],
        *,
        remainder: str = "drop",
    ) -> None:
        super().__init__()
        if remainder not in ("drop", "passthrough"):
            raise ConfigurationError(
                "remainder must be drop|passthrough", code="ml.remainder"
            )
        self.remainder = remainder
        self._specs = transformers
        self._fitted: list[tuple[str, Transform, list[int]]] = []

    def _resolve_cols(self, columns: list[str | int], names: list[str]) -> list[int]:
        indices: list[int] = []
        for col in columns:
            if isinstance(col, int):
                indices.append(col)
            else:
                if col not in names:
                    raise ConfigurationError(
                        f"unknown column {col!r}",
                        code="ml.column_missing",
                        context={"available": names},
                    )
                indices.append(names.index(col))
        return indices

    def _fit(self, x: list[list[float]], y: list[Any] | None = None) -> None:
        names = self.feature_names or [f"f{i}" for i in range(len(x[0]) if x else 0)]
        self._fitted = []
        used: set[int] = set()
        for name, spec, columns in self._specs:
            transform = create_transform(spec) if isinstance(spec, str) else spec
            idxs = self._resolve_cols(columns, names)
            used.update(idxs)
            subset = [[row[i] for i in idxs] for row in x]
            transform.feature_names = [names[i] for i in idxs]
            transform.fit_matrix(subset, y)
            self._fitted.append((name, transform, idxs))
        self._remainder_idxs = (
            [i for i in range(len(names)) if i not in used]
            if self.remainder == "passthrough"
            else []
        )

    def _transform(self, x: list[list[float]]) -> list[list[float]]:
        parts_per_row: list[list[list[float]]] = [[] for _ in x]
        for _, transform, idxs in self._fitted:
            subset = [[row[i] for i in idxs] for row in x]
            out = transform.transform_matrix(subset)
            for i, row in enumerate(out):
                parts_per_row[i].append(row)
        if self._remainder_idxs:
            for i, row in enumerate(x):
                parts_per_row[i].append([row[j] for j in self._remainder_idxs])
        return [[v for part in parts for v in part] for parts in parts_per_row]


class FeatureUnion(Transform):
    """Run multiple transforms on the full matrix and concatenate outputs."""

    name = "feature_union"

    def __init__(self, transformer_list: list[tuple[str, str | Transform]]) -> None:
        super().__init__()
        self._specs = transformer_list
        self._fitted: list[tuple[str, Transform]] = []

    def _fit(self, x: list[list[float]], y: list[Any] | None = None) -> None:
        self._fitted = []
        for name, spec in self._specs:
            transform = create_transform(spec) if isinstance(spec, str) else spec
            transform.feature_names = list(self.feature_names)
            transform.fit_matrix(x, y)
            self._fitted.append((name, transform))

    def _transform(self, x: list[list[float]]) -> list[list[float]]:
        outs = [t.transform_matrix(x) for _, t in self._fitted]
        if not outs:
            return [[] for _ in x]
        return [
            [v for part in parts for v in part]
            for parts in zip(*outs, strict=True)
        ]

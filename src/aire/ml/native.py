"""Native, zero-dependency estimators (``simple:*`` refs).

Pure-python reference implementations that keep model creation fully
offline-capable and serve as contract examples for plugin backends. They are
real learners (nearest-centroid, k-NN, gradient-descent linear regression),
not mocks.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

from aire.ml.estimator import Estimator
from aire.ml.types import TaskType


def _normalize_fit(x: list[list[float]]) -> tuple[list[float], list[float]]:
    """Per-feature mean/std for z-score normalization."""
    means, stds = [], []
    for col in zip(*x, strict=True):
        mean = sum(col) / len(col)
        var = sum((v - mean) ** 2 for v in col) / len(col)
        means.append(mean)
        stds.append(math.sqrt(var) or 1.0)
    return means, stds


def _apply_norm(row: list[float], means: list[float], stds: list[float]) -> list[float]:
    return [(v - m) / s for v, m, s in zip(row, means, stds, strict=True)]


class MajorityClassifier(Estimator):
    """Baseline: always predicts the most frequent class."""

    def __init__(self) -> None:
        super().__init__()
        self._counts: dict[str, int] = {}
        self._majority: str = ""

    def _fit_sync(self, x: list[list[float]], y: list[Any]) -> dict[str, float]:
        self._counts = dict(Counter(str(v) for v in y))
        self._majority = max(self._counts, key=lambda k: self._counts[k])
        return {"train_accuracy": self._counts[self._majority] / len(y)}

    def _predict_sync(self, x: list[list[float]]) -> list[float | str]:
        return [self._majority for _ in x]

    def _probabilities_sync(self, x: list[list[float]]) -> list[dict[str, float]] | None:
        total = sum(self._counts.values()) or 1
        probs = {label: count / total for label, count in self._counts.items()}
        return [dict(probs) for _ in x]

    def _state(self) -> dict[str, Any]:
        return {"counts": self._counts, "majority": self._majority}

    def _restore(self, state: dict[str, Any]) -> None:
        self._counts = dict(state["counts"])
        self._majority = str(state["majority"])


class CentroidClassifier(Estimator):
    """Nearest class centroid over z-score normalized features."""

    def __init__(self) -> None:
        super().__init__()
        self._centroids: dict[str, list[float]] = {}
        self._means: list[float] = []
        self._stds: list[float] = []

    def _fit_sync(self, x: list[list[float]], y: list[Any]) -> dict[str, float]:
        self._means, self._stds = _normalize_fit(x)
        sums: dict[str, list[float]] = defaultdict(lambda: [0.0] * len(x[0]))
        counts: dict[str, int] = defaultdict(int)
        for row, label in zip(x, y, strict=True):
            normed = _apply_norm(row, self._means, self._stds)
            key = str(label)
            counts[key] += 1
            for i, v in enumerate(normed):
                sums[key][i] += v
        self._centroids = {key: [v / counts[key] for v in total] for key, total in sums.items()}
        predictions = self._predict_sync(x)
        correct = sum(1 for p, t in zip(predictions, y, strict=True) if p == str(t))
        return {"train_accuracy": correct / len(y), "classes": float(len(self._centroids))}

    def _predict_sync(self, x: list[list[float]]) -> list[float | str]:
        predictions: list[float | str] = []
        for row in x:
            normed = _apply_norm(row, self._means, self._stds)
            best = min(
                self._centroids,
                key=lambda label: sum(
                    (a - b) ** 2 for a, b in zip(normed, self._centroids[label], strict=True)
                ),
            )
            predictions.append(best)
        return predictions

    def _state(self) -> dict[str, Any]:
        return {"centroids": self._centroids, "means": self._means, "stds": self._stds}

    def _restore(self, state: dict[str, Any]) -> None:
        self._centroids = {k: list(v) for k, v in state["centroids"].items()}
        self._means = list(state["means"])
        self._stds = list(state["stds"])


class KNNClassifier(Estimator):
    """k-nearest neighbors over normalized features."""

    def __init__(self, k: int = 3) -> None:
        super().__init__()
        self.k = k
        self._train_x: list[list[float]] = []
        self._train_y: list[str] = []
        self._means: list[float] = []
        self._stds: list[float] = []

    def _fit_sync(self, x: list[list[float]], y: list[Any]) -> dict[str, float]:
        self._means, self._stds = _normalize_fit(x)
        self._train_x = [_apply_norm(row, self._means, self._stds) for row in x]
        self._train_y = [str(v) for v in y]
        return {"samples": float(len(x))}

    def _predict_sync(self, x: list[list[float]]) -> list[float | str]:
        predictions: list[float | str] = []
        for row in x:
            normed = _apply_norm(row, self._means, self._stds)
            distances = sorted(
                (
                    sum((a - b) ** 2 for a, b in zip(normed, train, strict=True)),
                    label,
                )
                for train, label in zip(self._train_x, self._train_y, strict=True)
            )
            neighbors = [label for _, label in distances[: self.k]]
            predictions.append(Counter(neighbors).most_common(1)[0][0])
        return predictions

    def _state(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "train_x": self._train_x,
            "train_y": self._train_y,
            "means": self._means,
            "stds": self._stds,
        }

    def _restore(self, state: dict[str, Any]) -> None:
        self.k = int(state["k"])
        self._train_x = [list(r) for r in state["train_x"]]
        self._train_y = list(state["train_y"])
        self._means = list(state["means"])
        self._stds = list(state["stds"])


class LinearRegressor(Estimator):
    """Ordinary linear regression trained by gradient descent (pure python)."""

    task = TaskType.REGRESSION

    def __init__(self, *, epochs: int = 500, learning_rate: float = 0.05) -> None:
        super().__init__()
        self.epochs = epochs
        self.learning_rate = learning_rate
        self._weights: list[float] = []
        self._bias = 0.0
        self._means: list[float] = []
        self._stds: list[float] = []

    def _fit_sync(self, x: list[list[float]], y: list[Any]) -> dict[str, float]:
        self._means, self._stds = _normalize_fit(x)
        normed = [_apply_norm(row, self._means, self._stds) for row in x]
        targets = [float(v) for v in y]
        n_features = len(x[0])
        self._weights = [0.0] * n_features
        self._bias = 0.0
        n = len(x)
        final_loss = 0.0
        for _ in range(self.epochs):
            grad_w = [0.0] * n_features
            grad_b = 0.0
            final_loss = 0.0
            for row, target in zip(normed, targets, strict=True):
                prediction = (
                    sum(w * v for w, v in zip(self._weights, row, strict=True)) + self._bias
                )
                error = prediction - target
                final_loss += error * error
                grad_b += error
                for j in range(n_features):
                    grad_w[j] += error * row[j]
            self._bias -= self.learning_rate * grad_b / n
            for j in range(n_features):
                self._weights[j] -= self.learning_rate * grad_w[j] / n
        return {"train_mse": final_loss / n}

    def _predict_sync(self, x: list[list[float]]) -> list[float | str]:
        return [
            sum(
                w * v
                for w, v in zip(
                    self._weights, _apply_norm(row, self._means, self._stds), strict=True
                )
            )
            + self._bias
            for row in x
        ]

    def _state(self) -> dict[str, Any]:
        return {
            "weights": self._weights,
            "bias": self._bias,
            "means": self._means,
            "stds": self._stds,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
        }

    def _restore(self, state: dict[str, Any]) -> None:
        self._weights = list(state["weights"])
        self._bias = float(state["bias"])
        self._means = list(state["means"])
        self._stds = list(state["stds"])
        self.epochs = int(state["epochs"])
        self.learning_rate = float(state["learning_rate"])


NATIVE_ESTIMATORS: dict[str, type[Estimator]] = {
    "majority": MajorityClassifier,
    "centroid": CentroidClassifier,
    "knn": KNNClassifier,
    "linear_regression": LinearRegressor,
}


def create_native(name: str, **options: Any) -> Estimator:
    """Instantiate a native estimator by short name."""
    from aire.core.errors import ConfigurationError

    try:
        return NATIVE_ESTIMATORS[name](**options)
    except KeyError:
        raise ConfigurationError(
            f"unknown native estimator {name!r}",
            code="ml.estimator_unknown",
            context={"available": sorted(NATIVE_ESTIMATORS)},
        ) from None


def register(runtime: Any) -> None:
    """Register the native estimator factory on a runtime."""

    def _factory(name: str = "centroid", *, runtime: Any = None, **options: Any) -> Estimator:
        return create_native(name, **options)

    runtime.registry("estimator").register("simple", _factory, replace=True)

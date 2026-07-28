"""scikit-learn backend (``sklearn:<estimator>`` refs), lazily imported.

Covers the common estimator zoo by short name; any other estimator class can
be reached with a dotted path (``sklearn:sklearn.ensemble.AdaBoostClassifier``).
Requires ``pip install aire[ml]`` (scikit-learn).
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import Any

from aire.core.errors import ConfigurationError
from aire.core.types import Manifest
from aire.ml.estimator import Estimator
from aire.ml.types import TaskType

_SKLEARN_NAMES: dict[str, tuple[str, str, TaskType]] = {
    # short name -> (module, class, task)
    "logistic_regression": ("sklearn.linear_model", "LogisticRegression", TaskType.CLASSIFICATION),
    "linear_regression": ("sklearn.linear_model", "LinearRegression", TaskType.REGRESSION),
    "ridge": ("sklearn.linear_model", "Ridge", TaskType.REGRESSION),
    "lasso": ("sklearn.linear_model", "Lasso", TaskType.REGRESSION),
    "decision_tree": ("sklearn.tree", "DecisionTreeClassifier", TaskType.CLASSIFICATION),
    "random_forest": ("sklearn.ensemble", "RandomForestClassifier", TaskType.CLASSIFICATION),
    "random_forest_regressor": ("sklearn.ensemble", "RandomForestRegressor", TaskType.REGRESSION),
    "gradient_boosting": (
        "sklearn.ensemble",
        "GradientBoostingClassifier",
        TaskType.CLASSIFICATION,
    ),
    "svm": ("sklearn.svm", "SVC", TaskType.CLASSIFICATION),
    "svr": ("sklearn.svm", "SVR", TaskType.REGRESSION),
    "knn": ("sklearn.neighbors", "KNeighborsClassifier", TaskType.CLASSIFICATION),
    "naive_bayes": ("sklearn.naive_bayes", "GaussianNB", TaskType.CLASSIFICATION),
    "mlp": ("sklearn.neural_network", "MLPClassifier", TaskType.CLASSIFICATION),
}


def _require_sklearn() -> None:
    if importlib.util.find_spec("sklearn") is None:
        raise ConfigurationError(
            "scikit-learn is required for sklearn:* estimators: pip install 'aire[ml]'",
            code="ml.sklearn_missing",
            context={"backend": "sklearn"},
        )


def resolve_sklearn_class(name: str) -> tuple[type[Any], TaskType]:
    """Resolve a short name or dotted path to an estimator class + task."""
    _require_sklearn()
    if name in _SKLEARN_NAMES:
        module_name, class_name, task = _SKLEARN_NAMES[name]
        module = importlib.import_module(module_name)
        return getattr(module, class_name), task
    if "." in name:
        module_name, _, class_name = name.rpartition(".")
        try:
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
        except (ImportError, AttributeError) as exc:
            raise ConfigurationError(
                f"unknown sklearn estimator {name!r}",
                code="ml.estimator_unknown",
                context={"available": sorted(_SKLEARN_NAMES)},
                cause=exc,
            ) from exc
        task = TaskType.REGRESSION if "Regressor" in class_name else TaskType.CLASSIFICATION
        return cls, task
    raise ConfigurationError(
        f"unknown sklearn estimator {name!r}",
        code="ml.estimator_unknown",
        context={"available": sorted(_SKLEARN_NAMES)},
    )


class SklearnEstimator(Estimator):
    """Wraps any scikit-learn estimator behind the aire Estimator contract."""

    def __init__(self, name: str, **hyperparameters: Any) -> None:
        super().__init__()
        cls, task = resolve_sklearn_class(name)
        self.sklearn_name = name
        self.task = task
        self._model = cls(**hyperparameters)

    def backend_name(self) -> str:
        return f"sklearn:{self.sklearn_name}"

    def _fit_sync(self, x: list[list[float]], y: list[Any]) -> dict[str, float]:
        self._model.fit(x, y)
        score = float(self._model.score(x, y))
        metric = "train_accuracy" if self.task == TaskType.CLASSIFICATION else "train_r2"
        return {metric: score}

    def _predict_sync(self, x: list[list[float]]) -> list[float | str]:
        predictions = self._model.predict(x)
        return [v.item() if hasattr(v, "item") else v for v in predictions]

    def _probabilities_sync(self, x: list[list[float]]) -> list[dict[str, float]] | None:
        if not hasattr(self._model, "predict_proba"):
            return None
        classes = [str(c) for c in self._model.classes_]
        return [
            dict(zip(classes, (float(p) for p in row), strict=True))
            for row in self._model.predict_proba(x)
        ]

    @property
    def model(self) -> Any:
        """The underlying sklearn estimator (persist it with skops/joblib)."""
        return self._model

    # sklearn objects are not JSON-serializable, and aire never pickles
    # (security policy): persistence of fitted sklearn models is delegated
    # to the caller via skops.io / joblib on ``estimator.model``.
    def _state(self) -> dict[str, Any]:
        raise ConfigurationError(
            "sklearn models cannot be serialized by aire (no-pickle policy); "
            "persist estimator.model with skops.io or joblib instead",
            code="ml.persistence_delegated",
        )

    def _restore(self, state: dict[str, Any]) -> None:
        raise ConfigurationError(
            "sklearn models cannot be deserialized by aire (no-pickle policy); "
            "load with skops.io or joblib instead",
            code="ml.persistence_delegated",
        )

    def describe(self) -> Manifest:
        manifest = super().describe()
        manifest.extra["hyperparameters"] = {
            k: v
            for k, v in self._model.get_params().items()
            if isinstance(v, (int, float, str, bool))
        }
        return manifest


def register(runtime: Any) -> None:
    """Register the sklearn estimator factory on a runtime."""

    def _factory(name: str = "random_forest", *, runtime: Any = None, **options: Any) -> Estimator:
        return SklearnEstimator(name, **options)

    runtime.registry("estimator").register("sklearn", _factory, replace=True)

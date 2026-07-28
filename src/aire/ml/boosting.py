"""Gradient boosting backends: xgboost, lightgbm, catboost (lazy)."""

from __future__ import annotations

import contextlib
import importlib.util
from typing import Any

from aire.core.errors import ConfigurationError
from aire.core.types import Manifest
from aire.ml.estimator import Estimator
from aire.ml.types import TaskType


def _require(pkg: str, extra: str) -> Any:
    if importlib.util.find_spec(pkg) is None:
        raise ConfigurationError(
            f"{pkg} is required: pip install 'aire[{extra}]'",
            code=f"ml.{pkg}_missing",
            context={"backend": pkg},
        )
    return importlib.import_module(pkg)


class XGBoostEstimator(Estimator):
    """Wraps ``XGBClassifier`` / ``XGBRegressor``."""

    def __init__(
        self,
        name: str = "classifier",
        *,
        task: TaskType | str | None = None,
        **hyperparameters: Any,
    ) -> None:
        super().__init__()
        xgb = _require("xgboost", "xgboost")
        resolved_task = TaskType(task) if task else (
            TaskType.REGRESSION if "regressor" in name else TaskType.CLASSIFICATION
        )
        self.task = resolved_task
        self.name = name
        cls = xgb.XGBRegressor if resolved_task == TaskType.REGRESSION else xgb.XGBClassifier
        # sensible defaults for small datasets / offline demos
        params = {"n_estimators": 50, "max_depth": 3, "verbosity": 0, **hyperparameters}
        if resolved_task == TaskType.CLASSIFICATION:
            params.setdefault("eval_metric", "logloss")
        self._model = cls(**params)

    def backend_name(self) -> str:
        return f"xgboost:{self.name}"

    def _fit_sync(self, x: list[list[float]], y: list[Any]) -> dict[str, float]:
        self._model.fit(x, y)
        score = float(self._model.score(x, y))
        key = "train_accuracy" if self.task == TaskType.CLASSIFICATION else "train_r2"
        return {key: score}

    def _predict_sync(self, x: list[list[float]]) -> list[float | str]:
        preds = self._model.predict(x)
        return [v.item() if hasattr(v, "item") else v for v in preds]

    def _probabilities_sync(self, x: list[list[float]]) -> list[dict[str, float]] | None:
        if not hasattr(self._model, "predict_proba"):
            return None
        classes = [str(c) for c in self._model.classes_]
        return [
            dict(zip(classes, (float(p) for p in row), strict=True))
            for row in self._model.predict_proba(x)
        ]

    def _state(self) -> dict[str, Any]:
        raise ConfigurationError(
            "xgboost persistence delegated to estimator.model.save_model()",
            code="ml.persistence_delegated",
        )

    def _restore(self, state: dict[str, Any]) -> None:
        raise ConfigurationError(
            "xgboost persistence delegated to estimator.model.load_model()",
            code="ml.persistence_delegated",
        )

    @property
    def model(self) -> Any:
        return self._model

    def describe(self) -> Manifest:
        manifest = super().describe()
        manifest.extra["params"] = {
            k: v
            for k, v in self._model.get_params().items()
            if isinstance(v, (int, float, str, bool))
        }
        return manifest


class LightGBMEstimator(Estimator):
    """Wraps ``LGBMClassifier`` / ``LGBMRegressor``."""

    def __init__(
        self,
        name: str = "classifier",
        *,
        task: TaskType | str | None = None,
        **hyperparameters: Any,
    ) -> None:
        super().__init__()
        lgb = _require("lightgbm", "lightgbm")
        resolved_task = TaskType(task) if task else (
            TaskType.REGRESSION if "regressor" in name else TaskType.CLASSIFICATION
        )
        self.task = resolved_task
        self.name = name
        cls = lgb.LGBMRegressor if resolved_task == TaskType.REGRESSION else lgb.LGBMClassifier
        params = {"n_estimators": 50, "max_depth": 3, "verbosity": -1, **hyperparameters}
        self._model = cls(**params)

    def backend_name(self) -> str:
        return f"lightgbm:{self.name}"

    def _fit_sync(self, x: list[list[float]], y: list[Any]) -> dict[str, float]:
        self._model.fit(x, y)
        score = float(self._model.score(x, y))
        key = "train_accuracy" if self.task == TaskType.CLASSIFICATION else "train_r2"
        return {key: score}

    def _predict_sync(self, x: list[list[float]]) -> list[float | str]:
        preds = self._model.predict(x)
        return [v.item() if hasattr(v, "item") else v for v in preds]

    def _probabilities_sync(self, x: list[list[float]]) -> list[dict[str, float]] | None:
        if not hasattr(self._model, "predict_proba"):
            return None
        classes = [str(c) for c in self._model.classes_]
        return [
            dict(zip(classes, (float(p) for p in row), strict=True))
            for row in self._model.predict_proba(x)
        ]

    def _state(self) -> dict[str, Any]:
        raise ConfigurationError(
            "lightgbm persistence delegated to estimator.model.booster_.save_model()",
            code="ml.persistence_delegated",
        )

    def _restore(self, state: dict[str, Any]) -> None:
        raise ConfigurationError(
            "lightgbm persistence delegated to lightgbm.Booster(model_file=...)",
            code="ml.persistence_delegated",
        )

    @property
    def model(self) -> Any:
        return self._model


def register_xgboost(runtime: Any) -> None:
    def _factory(name: str = "classifier", *, runtime: Any = None, **options: Any) -> Estimator:
        return XGBoostEstimator(name, **options)

    runtime.registry("estimator").register("xgboost", _factory, replace=True)


def register_lightgbm(runtime: Any) -> None:
    def _factory(name: str = "classifier", *, runtime: Any = None, **options: Any) -> Estimator:
        return LightGBMEstimator(name, **options)

    runtime.registry("estimator").register("lightgbm", _factory, replace=True)


class CatBoostEstimator(Estimator):
    """Wraps ``CatBoostClassifier`` / ``CatBoostRegressor``."""

    def __init__(
        self,
        name: str = "classifier",
        *,
        task: TaskType | str | None = None,
        **hyperparameters: Any,
    ) -> None:
        super().__init__()
        cb = _require("catboost", "catboost")
        resolved_task = TaskType(task) if task else (
            TaskType.REGRESSION if "regressor" in name else TaskType.CLASSIFICATION
        )
        self.task = resolved_task
        self.name = name
        cls = (
            cb.CatBoostRegressor
            if resolved_task == TaskType.REGRESSION
            else cb.CatBoostClassifier
        )
        params = {
            "iterations": 50,
            "depth": 4,
            "verbose": False,
            "allow_writing_files": False,
            **hyperparameters,
        }
        self._model = cls(**params)

    def backend_name(self) -> str:
        return f"catboost:{self.name}"

    def _fit_sync(self, x: list[list[float]], y: list[Any]) -> dict[str, float]:
        self._model.fit(x, y)
        score = float(self._model.score(x, y))
        key = "train_accuracy" if self.task == TaskType.CLASSIFICATION else "train_r2"
        return {key: score}

    def _predict_sync(self, x: list[list[float]]) -> list[float | str]:
        preds = self._model.predict(x)
        # catboost may return column vectors
        flat = preds.flatten() if hasattr(preds, "flatten") else preds
        return [v.item() if hasattr(v, "item") else v for v in flat]

    def _probabilities_sync(self, x: list[list[float]]) -> list[dict[str, float]] | None:
        if not hasattr(self._model, "predict_proba"):
            return None
        classes = [str(c) for c in self._model.classes_]
        return [
            dict(zip(classes, (float(p) for p in row), strict=True))
            for row in self._model.predict_proba(x)
        ]

    def _state(self) -> dict[str, Any]:
        raise ConfigurationError(
            "catboost persistence delegated to estimator.model.save_model()",
            code="ml.persistence_delegated",
        )

    def _restore(self, state: dict[str, Any]) -> None:
        raise ConfigurationError(
            "catboost persistence delegated to CatBoost().load_model()",
            code="ml.persistence_delegated",
        )

    @property
    def model(self) -> Any:
        return self._model

    def describe(self) -> Manifest:
        manifest = super().describe()
        with contextlib.suppress(Exception):
            manifest.extra["params"] = {
                k: v
                for k, v in self._model.get_params().items()
                if isinstance(v, (int, float, str, bool))
            }
        return manifest


def register_catboost(runtime: Any) -> None:
    def _factory(name: str = "classifier", *, runtime: Any = None, **options: Any) -> Estimator:
        return CatBoostEstimator(name, **options)

    runtime.registry("estimator").register("catboost", _factory, replace=True)

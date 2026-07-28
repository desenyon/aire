"""Keras 3 backend (``keras:mlp``), lazily imported.

Works with any Keras 3 backend (tensorflow / torch / jax). Same Estimator
contract as native/sklearn/torch. Requires ``pip install aire[keras]``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from aire.core.errors import ConfigurationError
from aire.ml.estimator import Estimator
from aire.ml.types import TaskType


def _require_keras() -> Any:
    if importlib.util.find_spec("keras") is None:
        raise ConfigurationError(
            "Keras is required for keras:* estimators: pip install 'aire[keras]'",
            code="ml.keras_missing",
            context={"backend": "keras"},
        )
    import keras  # type: ignore[import-not-found]

    return keras


class KerasEstimator(Estimator):
    """Trains a Keras Sequential MLP (or custom ``model_factory``)."""

    def __init__(
        self,
        name: str = "mlp",
        *,
        task: TaskType | str = TaskType.CLASSIFICATION,
        hidden: tuple[int, ...] = (64, 32),
        epochs: int = 50,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        optimizer: str = "adam",
        loss: str | None = None,
        model_factory: Any | None = None,
        seed: int = 42,
        verbose: int = 0,
    ) -> None:
        super().__init__()
        self.keras = _require_keras()
        import contextlib

        with contextlib.suppress(Exception):  # pragma: no cover - backend variance
            self.keras.utils.set_random_seed(seed)
        self.task = TaskType(task)
        self.name = name
        self.hidden = tuple(hidden)
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.optimizer_name = optimizer
        self.loss_name = loss or (
            "sparse_categorical_crossentropy"
            if self.task == TaskType.CLASSIFICATION
            else "mse"
        )
        self.model_factory = model_factory
        self.verbose = verbose
        self._model: Any = None
        self._classes: list[str] = []

    def backend_name(self) -> str:
        return f"keras:{self.name}"

    def _build(self, n_features: int, n_outputs: int) -> Any:
        keras = self.keras
        if self.model_factory is not None:
            return self.model_factory(n_features, n_outputs)
        layers: list[Any] = [keras.layers.Input(shape=(n_features,))]
        for width in self.hidden:
            layers.append(keras.layers.Dense(width, activation="relu"))
        if self.task == TaskType.CLASSIFICATION:
            layers.append(keras.layers.Dense(n_outputs, activation="softmax"))
        else:
            layers.append(keras.layers.Dense(1))
        model = keras.Sequential(layers)
        opt = keras.optimizers.get(
            {"class_name": self.optimizer_name, "config": {"learning_rate": self.learning_rate}}
        )
        model.compile(optimizer=opt, loss=self.loss_name)
        return model

    def _fit_sync(self, x: list[list[float]], y: list[Any]) -> dict[str, float]:
        if self.task == TaskType.CLASSIFICATION:
            self._classes = sorted({str(v) for v in y})
            targets: list[Any] = [self._classes.index(str(v)) for v in y]
            n_outputs = len(self._classes)
        else:
            self._classes = []
            targets = [[float(v)] for v in y]
            n_outputs = 1
        self._model = self._build(len(x[0]) if x else 0, n_outputs)
        history = self._model.fit(
            x,
            targets,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
        )
        losses = history.history.get("loss", [0.0])
        return {"train_loss": float(losses[-1]), "epochs": float(self.epochs)}

    def _predict_sync(self, x: list[list[float]]) -> list[float | str]:
        assert self._model is not None
        output = self._model.predict(x, verbose=0)
        rows = output.tolist() if hasattr(output, "tolist") else list(output)
        if self.task == TaskType.CLASSIFICATION:
            indices = [int(max(range(len(row)), key=lambda i: row[i])) for row in rows]
            return [self._classes[i] for i in indices]
        flat = rows if not rows or not isinstance(rows[0], list) else [r[0] for r in rows]
        return [float(v) for v in flat]

    def _state(self) -> dict[str, Any]:
        raise ConfigurationError("keras estimators persist via save()", code="ml.state_unavailable")

    def _restore(self, state: dict[str, Any]) -> None:
        raise ConfigurationError("keras estimators persist via load()", code="ml.state_unavailable")

    def save(self, path: str | Path) -> Path:
        if self.report is None or self._model is None:
            raise ConfigurationError("nothing to save: estimator not fitted", code="ml.not_fitted")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        weights_path = target.with_suffix(".weights.h5")
        self._model.save_weights(str(weights_path))
        import json

        target.write_text(
            json.dumps(
                {
                    "backend": self.backend_name(),
                    "task": str(self.task),
                    "feature_names": self.feature_names,
                    "classes": self._classes,
                    "hidden": list(self.hidden),
                    "weights": str(weights_path.name),
                    "report": self.report.model_dump(mode="json"),
                },
                indent=2,
            )
        )
        return target

    def load(self, path: str | Path) -> KerasEstimator:
        import json

        payload = json.loads(Path(path).read_text())
        self.feature_names = list(payload["feature_names"])
        self._classes = list(payload["classes"])
        self.hidden = tuple(payload["hidden"])
        n_outputs = len(self._classes) if self._classes else 1
        self._model = self._build(len(self.feature_names), n_outputs)
        weights = Path(path).with_name(payload["weights"])
        self._model.load_weights(str(weights))
        from aire.ml.types import FitReport

        self.report = FitReport.model_validate(payload["report"])
        return self

    def describe(self) -> Any:
        manifest = super().describe()
        manifest.extra.update(
            {
                "hidden": list(self.hidden),
                "epochs": self.epochs,
                "optimizer": self.optimizer_name,
                "loss": self.loss_name,
                "batch_size": self.batch_size,
            }
        )
        return manifest


def register(runtime: Any) -> None:
    def _factory(name: str = "mlp", *, runtime: Any = None, **options: Any) -> Estimator:
        return KerasEstimator(name, **options)

    runtime.registry("estimator").register("keras", _factory, replace=True)

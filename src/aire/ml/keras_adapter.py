"""Keras 3 backend — full compile/fit/callback surface behind aire Estimator.

``keras:mlp`` or wrap any model via ``model_factory``. Works with any Keras 3
backend (tensorflow / torch / jax). Requires ``pip install aire[keras]``.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
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


def _aire_to_keras_callbacks(callbacks: list[Any], keras: Any) -> list[Any]:
    """Map aire callback objects / keras callbacks / string names to keras objs."""
    out: list[Any] = []
    for cb in callbacks:
        if isinstance(cb, str):
            table = {
                "early_stopping": keras.callbacks.EarlyStopping,
                "reduce_lr": keras.callbacks.ReduceLROnPlateau,
                "model_checkpoint": keras.callbacks.ModelCheckpoint,
                "csv_logger": keras.callbacks.CSVLogger,
                "terminate_on_nan": keras.callbacks.TerminateOnNaN,
            }
            if cb not in table:
                raise ConfigurationError(
                    f"unknown keras callback {cb!r}",
                    code="ml.keras_callback",
                    context={"available": sorted(table)},
                )
            if cb == "model_checkpoint":
                out.append(table[cb]("keras_ckpt.weights.h5", save_weights_only=True))
            elif cb == "csv_logger":
                out.append(table[cb]("keras_train.csv"))
            else:
                out.append(table[cb]())
            continue
        # aire EarlyStopping → keras EarlyStopping
        if type(cb).__name__ == "EarlyStopping" and not hasattr(cb, "set_model"):
            out.append(
                keras.callbacks.EarlyStopping(
                    monitor=getattr(cb, "monitor", "loss").replace("train_", ""),
                    patience=getattr(cb, "patience", 10),
                    min_delta=getattr(cb, "min_delta", 0.0),
                    mode=getattr(cb, "mode", "min"),
                )
            )
            continue
        out.append(cb)
    return out


class KerasEstimator(Estimator):
    """Trains a Keras model (Sequential MLP or custom ``model_factory``)."""

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
        metrics: list[str] | None = None,
        model_factory: Any | None = None,
        callbacks: list[Any] | None = None,
        validation_split: float = 0.0,
        class_weight: dict[Any, float] | None = None,
        seed: int = 42,
        verbose: int = 0,
        run_eagerly: bool = False,
        jit_compile: bool | str = "auto",
    ) -> None:
        super().__init__()
        self.keras = _require_keras()
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
        self.metrics = list(metrics) if metrics is not None else (
            ["accuracy"] if self.task == TaskType.CLASSIFICATION else ["mae"]
        )
        self.model_factory = model_factory
        self.callbacks = list(callbacks or [])
        self.validation_split = validation_split
        self.class_weight = class_weight
        self.verbose = verbose
        self.run_eagerly = run_eagerly
        self.jit_compile = jit_compile
        self._model: Any = None
        self._classes: list[str] = []
        self.history: dict[str, list[float]] = {}

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
            act = "softmax" if n_outputs > 1 else "sigmoid"
            layers.append(keras.layers.Dense(n_outputs, activation=act))
        else:
            layers.append(keras.layers.Dense(n_outputs))
        model = keras.Sequential(layers)
        return model

    def _compile(self, model: Any) -> None:
        keras = self.keras
        try:
            opt = keras.optimizers.get(
                {
                    "class_name": self.optimizer_name,
                    "config": {"learning_rate": self.learning_rate},
                }
            )
        except Exception:
            opt = self.optimizer_name
        compile_kwargs: dict[str, Any] = {
            "optimizer": opt,
            "loss": self.loss_name,
            "metrics": self.metrics,
        }
        with contextlib.suppress(TypeError):
            compile_kwargs["run_eagerly"] = self.run_eagerly
            compile_kwargs["jit_compile"] = self.jit_compile
            model.compile(**compile_kwargs)
            return
        model.compile(optimizer=opt, loss=self.loss_name, metrics=self.metrics)

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
        self._compile(self._model)
        cbs = _aire_to_keras_callbacks(self.callbacks, self.keras)
        history = self._model.fit(
            x,
            targets,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            validation_split=self.validation_split,
            callbacks=cbs,
            class_weight=self.class_weight,
        )
        self.history = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        losses = self.history.get("loss", [0.0])
        out: dict[str, float] = {
            "train_loss": float(losses[-1]),
            "epochs": float(len(losses)),
        }
        if "val_loss" in self.history:
            out["val_loss"] = float(self.history["val_loss"][-1])
        if "accuracy" in self.history:
            out["train_accuracy"] = float(self.history["accuracy"][-1])
        return out

    def _predict_sync(self, x: list[list[float]]) -> list[float | str]:
        assert self._model is not None
        output = self._model.predict(x, verbose=0)
        rows = output.tolist() if hasattr(output, "tolist") else list(output)
        if self.task == TaskType.CLASSIFICATION:
            if rows and isinstance(rows[0], list):
                indices = [int(max(range(len(row)), key=lambda i: row[i])) for row in rows]
            else:
                indices = [round(float(v)) for v in rows]
            return [self._classes[i] for i in indices]
        flat = rows if not rows or not isinstance(rows[0], list) else [r[0] for r in rows]
        return [float(v) for v in flat]

    def _probabilities_sync(self, x: list[list[float]]) -> list[dict[str, float]] | None:
        if self.task != TaskType.CLASSIFICATION or not self._classes:
            return None
        assert self._model is not None
        output = self._model.predict(x, verbose=0)
        rows = output.tolist() if hasattr(output, "tolist") else list(output)
        if not rows:
            return []
        if not isinstance(rows[0], list):
            # binary sigmoid
            return [
                {self._classes[0]: 1.0 - float(v), self._classes[1]: float(v)}
                if len(self._classes) == 2
                else {self._classes[0]: float(v)}
                for v in rows
            ]
        return [
            dict(zip(self._classes, (float(p) for p in row), strict=True)) for row in rows
        ]

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
        payload = {
            "backend": self.backend_name(),
            "task": str(self.task),
            "feature_names": self.feature_names,
            "classes": self._classes,
            "hidden": list(self.hidden),
            "weights": str(weights_path.name),
            "report": self.report.model_dump(mode="json"),
            "optimizer": self.optimizer_name,
            "loss": self.loss_name,
            "metrics": self.metrics,
        }
        target.write_text(json.dumps(payload, indent=2))
        return target

    def load(self, path: str | Path) -> KerasEstimator:
        payload = json.loads(Path(path).read_text())
        self.feature_names = list(payload["feature_names"])
        self._classes = list(payload["classes"])
        self.hidden = tuple(payload["hidden"])
        n_outputs = len(self._classes) if self._classes else 1
        self._model = self._build(len(self.feature_names), n_outputs)
        self._compile(self._model)
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
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "optimizer": self.optimizer_name,
                "loss": self.loss_name,
                "metrics": self.metrics,
                "validation_split": self.validation_split,
            }
        )
        return manifest


def register(runtime: Any) -> None:
    def _factory(name: str = "mlp", *, runtime: Any = None, **options: Any) -> Estimator:
        return KerasEstimator(name, **options)

    runtime.registry("estimator").register("keras", _factory, replace=True)

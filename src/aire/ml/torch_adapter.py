"""PyTorch backend — MLP + custom modules, wired to aire optim/loss/callbacks.

``torch:mlp`` trains a configurable MLP. Pass ``optimizer=``, ``loss=``,
``batch_size=``, ``scheduler=``, ``callbacks=``, or ``module_factory=``.
Requires ``pip install aire[torch]``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aire.core.errors import ConfigurationError
from aire.ml.callbacks import (
    HistoryCallback,
    run_callbacks_begin,
    run_callbacks_end,
    run_callbacks_epoch,
)
from aire.ml.estimator import Estimator
from aire.ml.types import TaskType

if TYPE_CHECKING:
    from collections.abc import Callable


def _require_torch() -> Any:
    if importlib.util.find_spec("torch") is None:
        raise ConfigurationError(
            "PyTorch is required for torch:* estimators: pip install 'aire[torch]'",
            code="ml.torch_missing",
            context={"backend": "torch"},
        )
    import torch  # type: ignore[import-not-found,unused-ignore]

    return torch


class TorchEstimator(Estimator):
    """Trains a torch module behind the aire Estimator contract."""

    def __init__(
        self,
        name: str = "mlp",
        *,
        task: TaskType | str = TaskType.CLASSIFICATION,
        hidden: tuple[int, ...] = (64, 32),
        epochs: int = 200,
        learning_rate: float = 1e-2,
        batch_size: int | None = None,
        optimizer: str = "adam",
        optimizer_options: dict[str, Any] | None = None,
        loss: str | None = None,
        loss_options: dict[str, Any] | None = None,
        scheduler: str | None = None,
        scheduler_options: dict[str, Any] | None = None,
        callbacks: list[Any] | None = None,
        module_factory: Callable[[int, int], Any] | None = None,
        device: str = "cpu",
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.torch = _require_torch()
        self.torch.manual_seed(seed)
        self.task = TaskType(task)
        self.name = name
        self.hidden = tuple(hidden)
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.optimizer_name = optimizer
        self.optimizer_options = dict(optimizer_options or {})
        self.loss_name = loss or (
            "cross_entropy" if self.task == TaskType.CLASSIFICATION else "mse"
        )
        self.loss_options = dict(loss_options or {})
        self.scheduler_name = scheduler
        self.scheduler_options = dict(scheduler_options or {})
        self.callbacks = list(callbacks or [])
        self.module_factory = module_factory
        self.device = device
        self._model: Any = None
        self._classes: list[str] = []
        self.history: list[dict[str, float]] = []

    def backend_name(self) -> str:
        return f"torch:{self.name}"

    def _build(self, n_features: int, n_outputs: int) -> Any:
        if self.module_factory is not None:
            return self.module_factory(n_features, n_outputs).to(self.device)
        nn = self.torch.nn
        layers: list[Any] = []
        width = n_features
        for hidden in self.hidden:
            layers += [nn.Linear(width, hidden), nn.ReLU()]
            width = hidden
        layers.append(nn.Linear(width, n_outputs))
        return nn.Sequential(*layers).to(self.device)

    def _make_optimizer(self) -> Any:
        from aire.ml import optim

        opts = {"lr": self.learning_rate, **self.optimizer_options}
        return optim.create(self.optimizer_name, self._model.parameters(), **opts)

    def _make_loss(self) -> Any:
        from aire.ml import loss

        return loss.create(self.loss_name, **self.loss_options)

    def _make_scheduler(self, optimizer: Any) -> Any | None:
        if not self.scheduler_name:
            return None
        torch = self.torch
        name = self.scheduler_name
        opts = dict(self.scheduler_options)
        if name == "step":
            return torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=opts.pop("step_size", 50), gamma=opts.pop("gamma", 0.1), **opts
            )
        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=opts.pop("T_max", self.epochs), **opts
            )
        if name == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode=opts.pop("mode", "min"), **opts
            )
        raise ConfigurationError(
            f"unknown scheduler {name!r}",
            code="ml.scheduler_unknown",
            context={"available": ["step", "cosine", "plateau"]},
        )

    def _batches(
        self, x_tensor: Any, y_tensor: Any
    ) -> list[tuple[Any, Any]]:
        n = x_tensor.size(0)
        bs = self.batch_size or n
        return [
            (x_tensor[i : i + bs], y_tensor[i : i + bs]) for i in range(0, n, bs)
        ]

    def _fit_sync(self, x: list[list[float]], y: list[Any]) -> dict[str, float]:
        torch = self.torch
        if self.task == TaskType.CLASSIFICATION:
            self._classes = sorted({str(v) for v in y})
            targets = [self._classes.index(str(v)) for v in y]
            n_outputs = len(self._classes)
            y_tensor = torch.tensor(targets, dtype=torch.long, device=self.device)
        else:
            self._classes = []
            n_outputs = 1
            y_tensor = torch.tensor(
                [[float(v)] for v in y], dtype=torch.float32, device=self.device
            )
        self._model = self._build(len(x[0]), n_outputs)
        x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
        optimizer = self._make_optimizer()
        loss_fn = self._make_loss()
        scheduler = self._make_scheduler(optimizer)
        callbacks = list(self.callbacks)
        history_cb = HistoryCallback()
        callbacks.append(history_cb)
        state: dict[str, Any] = {}
        run_callbacks_begin(callbacks, state)
        self._model.train()
        final_loss = 0.0
        epochs_run = 0
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            n_batches = 0
            for xb, yb in self._batches(x_tensor, y_tensor):
                optimizer.zero_grad()
                output = self._model(xb)
                loss = loss_fn(output, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())
                n_batches += 1
            final_loss = epoch_loss / max(n_batches, 1)
            epochs_run = epoch + 1
            logs = {"train_loss": final_loss}
            if scheduler is not None:
                if self.scheduler_name == "plateau":
                    scheduler.step(final_loss)
                else:
                    scheduler.step()
                logs["lr"] = float(optimizer.param_groups[0]["lr"])
            if run_callbacks_epoch(callbacks, epoch, logs, state):
                break
        run_callbacks_end(callbacks, state)
        self.history = list(history_cb.history)
        return {
            "train_loss": final_loss,
            "epochs": float(epochs_run),
            "early_stopped": float(1.0 if state.get("early_stopped") else 0.0),
        }

    def _predict_sync(self, x: list[list[float]]) -> list[float | str]:
        torch = self.torch
        assert self._model is not None
        self._model.eval()
        with torch.no_grad():
            output = self._model(torch.tensor(x, dtype=torch.float32, device=self.device))
        if self.task == TaskType.CLASSIFICATION:
            indices = output.argmax(dim=1).tolist()
            return [self._classes[i] for i in indices]
        return [float(v) for v in output.squeeze(-1).tolist()]

    def _state(self) -> dict[str, Any]:
        raise ConfigurationError("torch estimators persist via save()", code="ml.state_unavailable")

    def _restore(self, state: dict[str, Any]) -> None:
        raise ConfigurationError("torch estimators persist via load()", code="ml.state_unavailable")

    def save(self, path: str | Path) -> Path:
        if self.report is None or self._model is None:
            raise ConfigurationError("nothing to save: estimator not fitted", code="ml.not_fitted")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self.torch.save(
            {
                "backend": self.backend_name(),
                "task": str(self.task),
                "feature_names": self.feature_names,
                "classes": self._classes,
                "hidden": list(self.hidden),
                "state_dict": self._model.state_dict(),
                "report": self.report.model_dump(mode="json"),
            },
            target,
        )
        return target

    def load(self, path: str | Path) -> TorchEstimator:
        payload = self.torch.load(Path(path), weights_only=True, map_location=self.device)
        self.feature_names = list(payload["feature_names"])
        self._classes = list(payload["classes"])
        self.hidden = tuple(payload["hidden"])
        n_outputs = len(self._classes) if self._classes else 1
        self._model = self._build(len(self.feature_names), n_outputs)
        self._model.load_state_dict(payload["state_dict"])
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
                "scheduler": self.scheduler_name,
                "device": self.device,
            }
        )
        return manifest


def register(runtime: Any) -> None:
    def _factory(name: str = "mlp", *, runtime: Any = None, **options: Any) -> Estimator:
        return TorchEstimator(name, **options)

    runtime.registry("estimator").register("torch", _factory, replace=True)

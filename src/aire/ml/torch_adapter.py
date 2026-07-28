"""PyTorch backend — full training loop behind the aire Estimator contract.

Supports ``torch:mlp``, custom ``module_factory``, or ``architecture=`` from
``AI.ml.arch``. Training knobs: optim/loss/scheduler/callbacks, validation
split, AMP, ``torch.compile``, gradient clipping, checkpointing, DataLoader.
Requires ``pip install aire[torch]``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aire.core.errors import ConfigurationError
from aire.ml.callbacks import (
    EarlyStopping,
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
        batch_size: int | None = 32,
        optimizer: str = "adam",
        optimizer_options: dict[str, Any] | None = None,
        loss: str | None = None,
        loss_options: dict[str, Any] | None = None,
        scheduler: str | None = None,
        scheduler_options: dict[str, Any] | None = None,
        callbacks: list[Any] | None = None,
        module_factory: Callable[[int, int], Any] | None = None,
        architecture: Any | None = None,
        device: str = "cpu",
        seed: int = 42,
        validation_split: float = 0.0,
        amp: bool = False,
        compile_model: bool = False,
        compile_mode: str = "default",
        grad_clip: float | None = None,
        num_workers: int = 0,
        shuffle: bool = True,
        checkpoint_path: str | Path | None = None,
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
        self.architecture = architecture
        self.device = device
        self.validation_split = validation_split
        self.amp = amp
        self.compile_model = compile_model
        self.compile_mode = compile_mode
        self.grad_clip = grad_clip
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self._model: Any = None
        self._classes: list[str] = []
        self.history: list[dict[str, float]] = []

    def backend_name(self) -> str:
        return f"torch:{self.name}"

    def _build(self, n_features: int, n_outputs: int) -> Any:
        if self.architecture is not None:
            # LM-style stacks expect token ids; wrap with a linear projector head
            # for tabular use — prefer module_factory for custom nets.
            raise ConfigurationError(
                "pass module_factory= for tabular training with custom modules; "
                "architecture= stacks are for sequence models (use AI.ml.arch directly)",
                code="ml.arch_tabular",
            )
        if self.module_factory is not None:
            model = self.module_factory(n_features, n_outputs).to(self.device)
        else:
            nn = self.torch.nn
            layers: list[Any] = []
            width = n_features
            for hidden in self.hidden:
                layers += [nn.Linear(width, hidden), nn.ReLU()]
                width = hidden
            layers.append(nn.Linear(width, n_outputs))
            model = nn.Sequential(*layers).to(self.device)
        if self.compile_model and hasattr(self.torch, "compile"):
            model = self.torch.compile(model, mode=self.compile_mode)
        return model

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
        table: dict[str, Any] = {
            "step": lambda: torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=opts.pop("step_size", 50),
                gamma=opts.pop("gamma", 0.1),
                **opts,
            ),
            "cosine": lambda: torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=opts.pop("T_max", self.epochs), **opts
            ),
            "plateau": lambda: torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode=opts.pop("mode", "min"), **opts
            ),
            "onecycle": lambda: torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=opts.pop("max_lr", self.learning_rate),
                epochs=opts.pop("epochs", self.epochs),
                steps_per_epoch=opts.pop("steps_per_epoch", 1),
                **opts,
            ),
            "cosine_warm": lambda: torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=opts.pop("T_0", 10), **opts
            ),
        }
        if name not in table:
            raise ConfigurationError(
                f"unknown scheduler {name!r}",
                code="ml.scheduler_unknown",
                context={"available": sorted(table)},
            )
        return table[name]()

    def _dataloader(self, x_tensor: Any, y_tensor: Any) -> Any:
        torch = self.torch
        ds = torch.utils.data.TensorDataset(x_tensor, y_tensor)
        bs = self.batch_size or len(x_tensor)
        return torch.utils.data.DataLoader(
            ds,
            batch_size=bs,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )

    def _split(
        self, x_tensor: Any, y_tensor: Any
    ) -> tuple[Any, Any, Any | None, Any | None]:
        if self.validation_split <= 0:
            return x_tensor, y_tensor, None, None
        n = x_tensor.size(0)
        n_val = max(1, int(n * self.validation_split))
        n_train = n - n_val
        if n_train < 1:
            return x_tensor, y_tensor, None, None
        return (
            x_tensor[:n_train],
            y_tensor[:n_train],
            x_tensor[n_train:],
            y_tensor[n_train:],
        )

    def _eval_loss(self, loss_fn: Any, x_val: Any, y_val: Any) -> float:
        torch = self.torch
        assert self._model is not None
        self._model.eval()
        with torch.no_grad():
            out = self._model(x_val)
            return float(loss_fn(out, y_val).item())

    def _fit_sync(self, x: list[list[float]], y: list[Any]) -> dict[str, float]:  # noqa: C901
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
        x_tr, y_tr, x_val, y_val = self._split(x_tensor, y_tensor)
        optimizer = self._make_optimizer()
        loss_fn = self._make_loss()
        # onecycle needs steps_per_epoch
        if self.scheduler_name == "onecycle":
            bs = self.batch_size or max(len(x_tr), 1)
            self.scheduler_options.setdefault(
                "steps_per_epoch", max(1, (len(x_tr) + bs - 1) // bs)
            )
        scheduler = self._make_scheduler(optimizer)
        callbacks = list(self.callbacks)
        history_cb = HistoryCallback()
        callbacks.append(history_cb)
        state: dict[str, Any] = {}
        run_callbacks_begin(callbacks, state)
        scaler = torch.cuda.amp.GradScaler(enabled=self.amp) if self.amp else None
        loader = self._dataloader(x_tr, y_tr)
        final_loss = 0.0
        best_val = float("inf")
        epochs_run = 0
        for epoch in range(self.epochs):
            self._model.train()
            epoch_loss = 0.0
            n_batches = 0
            for xb, yb in loader:
                optimizer.zero_grad()
                if self.amp and scaler is not None:
                    with torch.cuda.amp.autocast():
                        output = self._model(xb)
                        loss = loss_fn(output, yb)
                    scaler.scale(loss).backward()
                    if self.grad_clip is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self._model.parameters(), self.grad_clip
                        )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    output = self._model(xb)
                    loss = loss_fn(output, yb)
                    loss.backward()
                    if self.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self._model.parameters(), self.grad_clip
                        )
                    optimizer.step()
                if self.scheduler_name == "onecycle" and scheduler is not None:
                    scheduler.step()
                epoch_loss += float(loss.item())
                n_batches += 1
            final_loss = epoch_loss / max(n_batches, 1)
            epochs_run = epoch + 1
            logs: dict[str, float] = {"train_loss": final_loss}
            if x_val is not None and y_val is not None:
                val_loss = self._eval_loss(loss_fn, x_val, y_val)
                logs["val_loss"] = val_loss
                if val_loss < best_val and self.checkpoint_path is not None:
                    best_val = val_loss
                    self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(self._model.state_dict(), self.checkpoint_path)
            if scheduler is not None and self.scheduler_name != "onecycle":
                if self.scheduler_name == "plateau":
                    scheduler.step(logs.get("val_loss", final_loss))
                else:
                    scheduler.step()
                logs["lr"] = float(optimizer.param_groups[0]["lr"])
            if run_callbacks_epoch(callbacks, epoch, logs, state):
                break
        run_callbacks_end(callbacks, state)
        self.history = list(history_cb.history)
        if self.checkpoint_path is not None and self.checkpoint_path.exists():
            self._model.load_state_dict(
                torch.load(self.checkpoint_path, weights_only=True, map_location=self.device)
            )
        return {
            "train_loss": final_loss,
            "epochs": float(epochs_run),
            "early_stopped": float(1.0 if state.get("early_stopped") else 0.0),
            **(
                {"val_loss": float(self.history[-1].get("val_loss", 0.0))}
                if self.history and "val_loss" in self.history[-1]
                else {}
            ),
        }

    def _predict_sync(self, x: list[list[float]]) -> list[float | str]:
        torch = self.torch
        assert self._model is not None
        self._model.eval()
        with torch.no_grad():
            output = self._model(
                torch.tensor(x, dtype=torch.float32, device=self.device)
            )
        if self.task == TaskType.CLASSIFICATION:
            indices = output.argmax(dim=1).tolist()
            return [self._classes[i] for i in indices]
        return [float(v) for v in output.squeeze(-1).tolist()]

    def _probabilities_sync(self, x: list[list[float]]) -> list[dict[str, float]] | None:
        if self.task != TaskType.CLASSIFICATION or not self._classes:
            return None
        torch = self.torch
        assert self._model is not None
        self._model.eval()
        with torch.no_grad():
            logits = self._model(
                torch.tensor(x, dtype=torch.float32, device=self.device)
            )
            probs = torch.softmax(logits, dim=1).tolist()
        return [
            dict(zip(self._classes, (float(p) for p in row), strict=True))
            for row in probs
        ]

    def _state(self) -> dict[str, Any]:
        raise ConfigurationError("torch estimators persist via save()", code="ml.state_unavailable")

    def _restore(self, state: dict[str, Any]) -> None:
        raise ConfigurationError("torch estimators persist via load()", code="ml.state_unavailable")

    def save(self, path: str | Path) -> Path:
        if self.report is None or self._model is None:
            raise ConfigurationError("nothing to save: estimator not fitted", code="ml.not_fitted")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        # unwrap compiled module if needed
        model = self._model
        state_dict = (
            model._orig_mod.state_dict()
            if hasattr(model, "_orig_mod")
            else model.state_dict()
        )
        self.torch.save(
            {
                "backend": self.backend_name(),
                "task": str(self.task),
                "feature_names": self.feature_names,
                "classes": self._classes,
                "hidden": list(self.hidden),
                "state_dict": state_dict,
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
        # disable compile on load for portability
        was = self.compile_model
        self.compile_model = False
        self._model = self._build(len(self.feature_names), n_outputs)
        self.compile_model = was
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
                "amp": self.amp,
                "compile": self.compile_model,
                "validation_split": self.validation_split,
                "grad_clip": self.grad_clip,
            }
        )
        return manifest


def register(runtime: Any) -> None:
    def _factory(name: str = "mlp", *, runtime: Any = None, **options: Any) -> Estimator:
        return TorchEstimator(name, **options)

    runtime.registry("estimator").register("torch", _factory, replace=True)


# re-export for callers that imported EarlyStopping from torch path historically
__all__ = ["EarlyStopping", "TorchEstimator", "register"]

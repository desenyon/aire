"""PyTorch backend (``torch:mlp`` or a user module factory), lazily imported.

``torch:mlp`` trains a configurable multilayer perceptron on the shared
feature convention — classification and regression. Custom architectures plug
in via ``module_factory=lambda n_features, n_outputs: nn.Module``.
Requires ``pip install aire[torch]``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aire.core.errors import ConfigurationError
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
        self.module_factory = module_factory
        self.device = device
        self._model: Any = None
        self._classes: list[str] = []

    def backend_name(self) -> str:
        return f"torch:{self.name}"

    def _build(self, n_features: int, n_outputs: int) -> Any:
        if self.module_factory is not None:
            return self.module_factory(n_features, n_outputs)
        nn = self.torch.nn
        layers: list[Any] = []
        width = n_features
        for hidden in self.hidden:
            layers += [nn.Linear(width, hidden), nn.ReLU()]
            width = hidden
        layers.append(nn.Linear(width, n_outputs))
        return nn.Sequential(*layers).to(self.device)

    def _fit_sync(self, x: list[list[float]], y: list[Any]) -> dict[str, float]:
        torch = self.torch
        if self.task == TaskType.CLASSIFICATION:
            self._classes = sorted({str(v) for v in y})
            targets = [self._classes.index(str(v)) for v in y]
            n_outputs = len(self._classes)
            loss_fn: Any = torch.nn.CrossEntropyLoss()
            y_tensor = torch.tensor(targets, dtype=torch.long, device=self.device)
        else:
            self._classes = []
            n_outputs = 1
            loss_fn = torch.nn.MSELoss()
            y_tensor = torch.tensor(
                [[float(v)] for v in y], dtype=torch.float32, device=self.device
            )
        self._model = self._build(len(x[0]), n_outputs)
        x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)
        self._model.train()
        final_loss = 0.0
        for _ in range(self.epochs):
            optimizer.zero_grad()
            output = self._model(x_tensor)
            loss = loss_fn(output, y_tensor)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.item())
        return {"train_loss": final_loss, "epochs": float(self.epochs)}

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

    # torch modules are not JSON-serializable: persist with torch.save.
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
        # weights_only=True: tensors + primitives only, never executable pickle.
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
                "device": self.device,
            }
        )
        return manifest


def register(runtime: Any) -> None:
    """Register the torch estimator factory on a runtime."""

    def _factory(name: str = "mlp", *, runtime: Any = None, **options: Any) -> Estimator:
        return TorchEstimator(name, **options)

    runtime.registry("estimator").register("torch", _factory, replace=True)

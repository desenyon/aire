"""Framework-independent training contracts.

``FunctionTrainer`` runs any user-supplied training function with epochs,
early stopping and checkpointing — real training orchestration without core
depending on a specific ML framework. Framework adapters (pytorch, jax) are
plugins implementing the same :class:`Trainer` protocol.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from aire.core.serialization import write_json_file
from aire.data.dataset import Dataset


class TrainingConfig(BaseModel):
    """Run configuration shared by all trainer implementations."""

    strategy: str = "supervised"  # supervised | unsupervised | self_supervised | preference
    backend: str = "function"
    epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 1e-4
    early_stopping_patience: int | None = None
    checkpoint_dir: str | None = None
    seed: int = 42
    hyperparameters: dict[str, Any] = Field(default_factory=dict)


class Checkpoint(BaseModel):
    """A durable snapshot of training progress."""

    epoch: int
    metrics: dict[str, float] = Field(default_factory=dict)
    path: str | None = None
    state: dict[str, Any] = Field(default_factory=dict)


class TrainResult(BaseModel):
    """Outcome of a training run."""

    epochs_completed: int
    best_metric: float | None = None
    checkpoints: list[Checkpoint] = Field(default_factory=list)
    history: list[dict[str, float]] = Field(default_factory=list)
    duration_s: float = 0.0
    stopped_early: bool = False


@runtime_checkable
class Trainer(Protocol):
    """The contract every training backend implements."""

    async def fit(self, dataset: Dataset) -> TrainResult: ...

    def describe(self) -> dict[str, Any]: ...


# A step function: (epoch, dataset, config, state) -> (metrics, new_state)
StepFn = Callable[
    [int, Dataset, TrainingConfig, dict[str, Any]],
    tuple[dict[str, float], dict[str, Any]] | Awaitable[tuple[dict[str, float], dict[str, Any]]],
]


class FunctionTrainer:
    """Orchestrates a user-provided step function with full training-loop semantics."""

    def __init__(
        self,
        step: StepFn,
        config: TrainingConfig | None = None,
        *,
        metric: str = "loss",
        minimize: bool = True,
    ) -> None:
        self.step = step
        self.config = config or TrainingConfig()
        self.metric = metric
        self.minimize = minimize

    async def fit(self, dataset: Dataset) -> TrainResult:
        import inspect

        started = time.time()
        state: dict[str, Any] = {}
        history: list[dict[str, float]] = []
        checkpoints: list[Checkpoint] = []
        best: float | None = None
        patience_left = self.config.early_stopping_patience
        stopped_early = False

        for epoch in range(self.config.epochs):
            outcome = self.step(epoch, dataset, self.config, state)
            if inspect.isawaitable(outcome):
                outcome = await outcome
            metrics, state = outcome
            history.append(dict(metrics))
            value = metrics.get(self.metric)
            improved = value is not None and (
                best is None or (value < best if self.minimize else value > best)
            )
            if improved:
                best = value
                patience_left = self.config.early_stopping_patience
                checkpoint = self._save_checkpoint(epoch, metrics, state)
                checkpoints.append(checkpoint)
            elif patience_left is not None:
                patience_left -= 1
                if patience_left <= 0:
                    stopped_early = True
                    break

        return TrainResult(
            epochs_completed=len(history),
            best_metric=best,
            checkpoints=checkpoints,
            history=history,
            duration_s=time.time() - started,
            stopped_early=stopped_early,
        )

    def _save_checkpoint(
        self, epoch: int, metrics: dict[str, float], state: dict[str, Any]
    ) -> Checkpoint:
        checkpoint = Checkpoint(epoch=epoch, metrics=metrics, state=state)
        if self.config.checkpoint_dir:
            path = Path(self.config.checkpoint_dir) / f"checkpoint-epoch{epoch}.json"
            write_json_file(path, checkpoint)
            checkpoint.path = str(path)
        return checkpoint

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "trainer",
            "backend": "function",
            "config": self.config.model_dump(mode="json"),
            "metric": self.metric,
        }

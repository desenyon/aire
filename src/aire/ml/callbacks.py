"""Training callbacks for torch / keras estimators."""

from __future__ import annotations

from typing import Any, Protocol


class Callback(Protocol):
    def on_train_begin(self, state: dict[str, Any]) -> None: ...
    def on_epoch_end(self, epoch: int, logs: dict[str, float], state: dict[str, Any]) -> bool:
        """Return True to stop training early."""
        ...
    def on_train_end(self, state: dict[str, Any]) -> None: ...


class EarlyStopping:
    """Stop when a monitored metric stops improving."""

    def __init__(
        self,
        *,
        monitor: str = "train_loss",
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
    ) -> None:
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best: float | None = None
        self.wait = 0
        self.stopped_epoch = 0

    def on_train_begin(self, state: dict[str, Any]) -> None:
        self.best = None
        self.wait = 0
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch: int, logs: dict[str, float], state: dict[str, Any]) -> bool:
        value = logs.get(self.monitor)
        if value is None:
            return False
        if self.best is None:
            self.best = value
            return False
        improved = (
            value < self.best - self.min_delta
            if self.mode == "min"
            else value > self.best + self.min_delta
        )
        if improved:
            self.best = value
            self.wait = 0
            return False
        self.wait += 1
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            state["early_stopped"] = True
            return True
        return False

    def on_train_end(self, state: dict[str, Any]) -> None:
        return None


class HistoryCallback:
    """Record per-epoch logs."""

    def __init__(self) -> None:
        self.history: list[dict[str, float]] = []

    def on_train_begin(self, state: dict[str, Any]) -> None:
        self.history = []

    def on_epoch_end(self, epoch: int, logs: dict[str, float], state: dict[str, Any]) -> bool:
        entry = {"epoch": float(epoch), **logs}
        self.history.append(entry)
        return False

    def on_train_end(self, state: dict[str, Any]) -> None:
        state["history"] = list(self.history)


def run_callbacks_begin(callbacks: list[Any], state: dict[str, Any]) -> None:
    for cb in callbacks:
        cb.on_train_begin(state)


def run_callbacks_epoch(
    callbacks: list[Any], epoch: int, logs: dict[str, float], state: dict[str, Any]
) -> bool:
    stop = False
    for cb in callbacks:
        stop = cb.on_epoch_end(epoch, logs, state) or stop
    return stop


def run_callbacks_end(callbacks: list[Any], state: dict[str, Any]) -> None:
    for cb in callbacks:
        cb.on_train_end(state)

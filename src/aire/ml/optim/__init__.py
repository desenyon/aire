"""Optimizer contract + registry (torch-backed, lazy)."""

from __future__ import annotations

from typing import Any

from aire.core.errors import ConfigurationError, NotFoundError
from aire.ml.arch.torch_util import require_torch

_OPTIMIZERS: dict[str, Any] = {}


def register(name: str, factory: Any | None = None, *, replace: bool = False) -> Any:
    def _do(fn: Any) -> Any:
        if name in _OPTIMIZERS and not replace:
            raise ConfigurationError(
                f"optimizer {name!r} already registered", code="optim.duplicate"
            )
        _OPTIMIZERS[name] = fn
        return fn

    if factory is not None:
        return _do(factory)
    return _do


def create(name: str, params: Any, **options: Any) -> Any:
    """Build a torch optimizer by name over ``params`` (iterable of tensors)."""
    require_torch()
    _ensure()
    try:
        return _OPTIMIZERS[name](params, **options)
    except KeyError:
        raise NotFoundError("optimizer", name, context={"available": sorted(_OPTIMIZERS)}) from None


def names() -> list[str]:
    _ensure()
    return sorted(_OPTIMIZERS)


def describe() -> dict[str, Any]:
    return {
        "kind": "ml.optim",
        "available": names(),
        "usage": 'AI.ml.optim.create("adamw", model.parameters(), lr=1e-3, weight_decay=0.01)',
    }


def _ensure() -> None:
    if _OPTIMIZERS:
        return
    torch = require_torch()

    @register("sgd")
    def _sgd(
        params: Any, *, lr: float = 1e-2, momentum: float = 0.0, weight_decay: float = 0.0, **_: Any
    ) -> Any:
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

    @register("adam")
    def _adam(
        params: Any,
        *,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        **_: Any,
    ) -> Any:
        return torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

    @register("adamw")
    def _adamw(
        params: Any,
        *,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        **_: Any,
    ) -> Any:
        return torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

    @register("rmsprop")
    def _rmsprop(
        params: Any,
        *,
        lr: float = 1e-2,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
        **_: Any,
    ) -> Any:
        return torch.optim.RMSprop(
            params, lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=momentum
        )

    @register("adagrad")
    def _adagrad(params: Any, *, lr: float = 1e-2, weight_decay: float = 0.0, **_: Any) -> Any:
        return torch.optim.Adagrad(params, lr=lr, weight_decay=weight_decay)

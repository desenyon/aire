"""Loss function contract + registry (torch-backed, lazy)."""

from __future__ import annotations

from typing import Any

from aire.core.errors import ConfigurationError, NotFoundError
from aire.ml.arch.torch_util import require_torch

_LOSSES: dict[str, Any] = {}


def register(name: str, factory: Any | None = None, *, replace: bool = False) -> Any:
    def _do(fn: Any) -> Any:
        if name in _LOSSES and not replace:
            raise ConfigurationError(f"loss {name!r} already registered", code="loss.duplicate")
        _LOSSES[name] = fn
        return fn

    if factory is not None:
        return _do(factory)
    return _do


def create(name: str, **options: Any) -> Any:
    """Build a callable loss module/function by name."""
    require_torch()
    _ensure()
    try:
        return _LOSSES[name](**options)
    except KeyError:
        raise NotFoundError("loss", name, context={"available": sorted(_LOSSES)}) from None


def names() -> list[str]:
    _ensure()
    return sorted(_LOSSES)


def describe() -> dict[str, Any]:
    return {
        "kind": "ml.loss",
        "available": names(),
        "usage": (
            'loss = AI.ml.loss.create("cross_entropy", label_smoothing=0.1); '
            "loss(logits, targets)"
        ),
    }


def _ensure() -> None:  # noqa: C901
    if _LOSSES:
        return
    torch = require_torch()

    @register("cross_entropy")
    def _ce(*, label_smoothing: float = 0.0, ignore_index: int = -100, **_: Any) -> Any:
        return torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing, ignore_index=ignore_index)

    @register("nll")
    def _nll(*, ignore_index: int = -100, **_: Any) -> Any:
        return torch.nn.NLLLoss(ignore_index=ignore_index)

    @register("mse")
    def _mse(**_: Any) -> Any:
        return torch.nn.MSELoss()

    @register("l1")
    def _l1(**_: Any) -> Any:
        return torch.nn.L1Loss()

    @register("huber")
    def _huber(*, delta: float = 1.0, **_: Any) -> Any:
        return torch.nn.HuberLoss(delta=delta)

    @register("smooth_l1")
    def _smooth(**_: Any) -> Any:
        return torch.nn.SmoothL1Loss()

    @register("bce")
    def _bce(**_: Any) -> Any:
        return torch.nn.BCEWithLogitsLoss()

    @register("kl_div")
    def _kl(**_: Any) -> Any:
        return torch.nn.KLDivLoss(reduction="batchmean")

    @register("cosine")
    def _cos(**_: Any) -> Any:
        return torch.nn.CosineEmbeddingLoss()

    @register("ctc")
    def _ctc(**_: Any) -> Any:
        return torch.nn.CTCLoss(blank=0, zero_infinity=True)

    @register("moe_load_balance")
    def _moe_lb(*, n_experts: int = 8, **_: Any) -> Any:
        """Auxiliary MoE load-balancing loss from router probs [..., E]."""

        class MoELoadBalance(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.n_experts = n_experts
                self.kind = "moe_load_balance"

            def forward(self, router_probs: Any) -> Any:
                # router_probs: (..., E) after softmax
                flat = router_probs.reshape(-1, router_probs.size(-1))
                density = flat.mean(dim=0)
                usage = (flat > 0).float().mean(dim=0)
                return self.n_experts * (density * usage).sum()

        return MoELoadBalance()

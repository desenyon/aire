"""Normalization blocks."""

from __future__ import annotations

from typing import Any

from aire.ml.arch.registry import NORM
from aire.ml.arch.torch_util import require_torch


@NORM.register("layernorm")
def build_layernorm(*, n_embd: int, eps: float = 1e-5, **_: Any) -> Any:
    torch = require_torch()
    return torch.nn.LayerNorm(n_embd, eps=eps)


@NORM.register("rmsnorm")
def build_rmsnorm(*, n_embd: int, eps: float = 1e-5, **_: Any) -> Any:
    torch = require_torch()

    class RMSNorm(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(n_embd))
            self.eps = eps
            self.kind = "rmsnorm"

        def forward(self, x: Any) -> Any:
            rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
            return self.weight * x * rms

    return RMSNorm()


@NORM.register("identity")
def build_identity(*, n_embd: int = 0, **_: Any) -> Any:
    torch = require_torch()
    return torch.nn.Identity()

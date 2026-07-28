"""Residual connection strategies."""

from __future__ import annotations

from typing import Any

from aire.ml.arch.registry import RESIDUAL
from aire.ml.arch.torch_util import require_torch


@RESIDUAL.register("add")
def build_add(*, n_embd: int = 0, **_: Any) -> Any:
    """Standard x + f(x) residual (stateless callable module)."""
    torch = require_torch()

    class AddResidual(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.kind = "add"

        def forward(self, x: Any, branch: Any) -> Any:
            return x + branch

        def remix(self, blocks: list[Any], partial: Any | None = None) -> Any:
            raise NotImplementedError("add residual has no depth remix")

    return AddResidual()


@RESIDUAL.register("attn_res")
def build_attn_res(*, n_embd: int, **_: Any) -> Any:
    """Blockwise Attention Residuals — selective depth-wise retrieval."""
    torch = require_torch()

    class AttnRes(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = torch.nn.Linear(n_embd, 1, bias=False)
            self.norm = torch.nn.LayerNorm(n_embd)
            self.kind = "attn_res"

        def forward(self, x: Any, branch: Any) -> Any:
            return x + branch

        def remix(self, blocks: list[Any], partial: Any | None = None) -> Any:
            values = list(blocks)
            if partial is not None:
                values.append(partial)
            stacked = torch.stack(values, dim=0)
            keys = self.norm(stacked)
            logits = torch.einsum("d,nbtd->nbt", self.proj.weight.squeeze(0), keys)
            weights = torch.softmax(logits, dim=0)
            return torch.einsum("nbt,nbtd->btd", weights, stacked)

    return AttnRes()

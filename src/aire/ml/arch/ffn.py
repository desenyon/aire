"""Feed-forward / capacity blocks: MLP, SiTU-MLP, MoE, latent MoE."""

from __future__ import annotations

from typing import Any

from aire.ml.arch.registry import FFN
from aire.ml.arch.torch_util import require_torch


def _situ(x: Any, beta: float = 1.0, linear_beta: float | None = None) -> Any:
    torch = require_torch()
    d = x.shape[-1] // 2
    gate = x[..., :d].to(torch.float32)
    up = x[..., d:].to(torch.float32)
    a = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    return (a * up).to(x.dtype)


@FFN.register("mlp")
def build_mlp(
    *, n_embd: int, dropout: float = 0.0, bias: bool = True, mult: int = 4, **_: Any
) -> Any:
    """GPT-2 style MLP: Linear(mult*d) → GELU → Linear(d)."""
    torch = require_torch()

    class MLP(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            hid = mult * n_embd
            self.fc = torch.nn.Linear(n_embd, hid, bias=bias)
            self.proj = torch.nn.Linear(hid, n_embd, bias=bias)
            self.drop = torch.nn.Dropout(dropout)
            self.kind = "mlp"

        def forward(self, x: Any) -> Any:
            return self.drop(self.proj(torch.nn.functional.gelu(self.fc(x))))

    return MLP()


@FFN.register("swiglu")
def build_swiglu(
    *, n_embd: int, dropout: float = 0.0, bias: bool = True, mult: int = 4, **_: Any
) -> Any:
    torch = require_torch()

    class SwiGLU(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            hid = mult * n_embd
            self.w = torch.nn.Linear(n_embd, 2 * hid, bias=bias)
            self.proj = torch.nn.Linear(hid, n_embd, bias=bias)
            self.drop = torch.nn.Dropout(dropout)
            self.kind = "swiglu"

        def forward(self, x: Any) -> Any:
            u, g = self.w(x).chunk(2, dim=-1)
            return self.drop(self.proj(torch.nn.functional.silu(g) * u))

    return SwiGLU()


@FFN.register("situ_mlp")
def build_situ_mlp(
    *,
    n_embd: int,
    dropout: float = 0.0,
    bias: bool = True,
    mult: int = 4,
    situ_beta: float = 1.0,
    **_: Any,
) -> Any:
    """Dense FFN using SiTU activation (gate||up split)."""
    torch = require_torch()

    class SiTUMLP(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            hid = mult * n_embd
            self.up = torch.nn.Linear(n_embd, 2 * hid, bias=bias)
            self.down = torch.nn.Linear(hid, n_embd, bias=bias)
            self.drop = torch.nn.Dropout(dropout)
            self.beta = situ_beta
            self.kind = "situ_mlp"

        def forward(self, x: Any) -> Any:
            return self.drop(self.down(_situ(self.up(x), beta=self.beta)))

    return SiTUMLP()


def _build_moe(
    *,
    n_embd: int,
    n_experts: int = 8,
    n_shared: int = 2,
    top_k: int = 2,
    dropout: float = 0.0,
    bias: bool = True,
    situ_beta: float = 1.0,
    latent: bool = False,
    latent_dim: int | None = None,
    mult: int = 2,
    kind: str = "moe",
    **_: Any,
) -> Any:
    torch = require_torch()
    width = (latent_dim or max(32, n_embd // 2)) if latent else n_embd
    routed = max(0, n_experts - n_shared)

    class Expert(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            hid = mult * width
            self.up = torch.nn.Linear(width, 2 * hid, bias=bias)
            self.down = torch.nn.Linear(hid, width, bias=bias)

        def forward(self, x: Any) -> Any:
            return self.down(_situ(self.up(x), beta=situ_beta))

    class MoE(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.width = width
            self.top_k = top_k
            self.in_proj = (
                torch.nn.Linear(n_embd, width, bias=bias) if latent else torch.nn.Identity()
            )
            self.out_proj = (
                torch.nn.Linear(width, n_embd, bias=bias) if latent else torch.nn.Identity()
            )
            self.router = torch.nn.Linear(width, routed, bias=False) if routed else None
            self.shared = torch.nn.ModuleList([Expert() for _ in range(n_shared)])
            self.experts = torch.nn.ModuleList([Expert() for _ in range(routed)])
            self.drop = torch.nn.Dropout(dropout)
            self.kind = kind

        def forward(self, x: Any) -> Any:
            h = self.in_proj(x)
            out = torch.zeros_like(h)
            for expert in self.shared:
                out = out + expert(h)
            if self.router is not None and self.experts:
                logits = self.router(h)
                k = min(self.top_k, logits.size(-1))
                weights, indices = torch.topk(logits, k=k, dim=-1)
                weights = torch.softmax(weights, dim=-1)
                flat_h, flat_out = h.reshape(-1, self.width), out.reshape(-1, self.width)
                flat_w, flat_i = weights.reshape(-1, k), indices.reshape(-1, k)
                for eid, expert in enumerate(self.experts):
                    mask = flat_i == eid
                    if not mask.any():
                        continue
                    tok, slot = mask.nonzero(as_tuple=True)
                    flat_out.index_add_(
                        0, tok, expert(flat_h[tok]) * flat_w[tok, slot].unsqueeze(-1)
                    )
                out = flat_out.view_as(h)
            return self.drop(self.out_proj(out))

    return MoE()


@FFN.register("moe")
def build_moe(**options: Any) -> Any:
    return _build_moe(latent=False, kind="moe", **options)


@FFN.register("latent_moe")
def build_latent_moe(**options: Any) -> Any:
    return _build_moe(latent=True, kind="latent_moe", **options)

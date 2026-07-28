"""Attention / associative-memory blocks — each independently constructible."""

from __future__ import annotations

import math
from typing import Any

from aire.ml.arch.registry import ATTENTION
from aire.ml.arch.torch_util import require_torch


def _split(x: Any, b: int, t: int, h: int, dh: int) -> Any:
    return x.view(b, t, h, dh).transpose(1, 2)


def _merge(x: Any, b: int, t: int, d: int) -> Any:
    return x.transpose(1, 2).contiguous().view(b, t, d)


@ATTENTION.register("mha")
def build_mha(
    *,
    n_embd: int,
    n_head: int,
    dropout: float = 0.0,
    bias: bool = True,
    block_size: int = 128,
    **_: Any,
) -> Any:
    """Causal multi-head attention with KV cache."""
    torch = require_torch()
    head_dim = n_embd // n_head

    class CausalMHA(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.n_head, self.head_dim, self.n_embd = n_head, head_dim, n_embd
            self.c_attn = torch.nn.Linear(n_embd, 3 * n_embd, bias=bias)
            self.c_proj = torch.nn.Linear(n_embd, n_embd, bias=bias)
            self.attn_drop = torch.nn.Dropout(dropout)
            self.resid_drop = torch.nn.Dropout(dropout)
            mask = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
            self.register_buffer("bias_mask", mask, persistent=False)
            self.kind = "mha"

        def forward(self, x: Any, cache: Any = None) -> tuple[Any, Any]:
            b, t, c = x.size()
            q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
            q, k, v = (_split(tns, b, t, self.n_head, self.head_dim) for tns in (q, k, v))
            if cache is not None:
                k = torch.cat((cache[0], k), dim=2)
                v = torch.cat((cache[1], v), dim=2)
            past = k.size(2)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            q0 = past - t
            att = att.masked_fill(self.bias_mask[:, :, q0:past, :past] == 0, float("-inf"))
            y = _merge(self.attn_drop(att.softmax(-1)) @ v, b, t, c)
            return self.resid_drop(self.c_proj(y)), (k, v)

    return CausalMHA()


@ATTENTION.register("linear")
def build_linear(
    *, n_embd: int, n_head: int, bias: bool = True, dropout: float = 0.0, **_: Any
) -> Any:
    """ELU+1 feature-map linear attention with fixed DxD state cache."""
    torch = require_torch()
    F = torch.nn.functional
    head_dim = n_embd // n_head

    class LinearAttn(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.n_head, self.head_dim = n_head, head_dim
            self.qkv = torch.nn.Linear(n_embd, 3 * n_embd, bias=bias)
            self.o = torch.nn.Linear(n_embd, n_embd, bias=bias)
            self.drop = torch.nn.Dropout(dropout)
            self.kind = "linear"

        def forward(self, x: Any, cache: Any = None) -> tuple[Any, Any]:
            b, t, d = x.shape
            qkv = self.qkv(x)
            q = F.elu(_split(qkv[:, :, :d], b, t, self.n_head, self.head_dim)) + 1
            k = F.elu(_split(qkv[:, :, d : 2 * d], b, t, self.n_head, self.head_dim)) + 1
            v = _split(qkv[:, :, 2 * d :], b, t, self.n_head, self.head_dim)
            if cache is None:
                s = torch.zeros(b, self.n_head, self.head_dim, self.head_dim, device=x.device)
                z = torch.zeros(b, self.n_head, self.head_dim, device=x.device)
            else:
                s, z = cache
            outs = []
            for i in range(t):
                ki, vi, qi = k[:, :, i], v[:, :, i], q[:, :, i]
                s = s + ki.unsqueeze(-1) @ vi.unsqueeze(-2)
                z = z + ki
                outs.append(
                    (qi.unsqueeze(-2) @ s).squeeze(-2)
                    / (qi * z).sum(-1, keepdim=True).clamp_min(1e-6)
                )
            return self.drop(self.o(_merge(torch.stack(outs, 2), b, t, d))), (s, z)

    return LinearAttn()


def _delta_family(
    *,
    n_embd: int,
    n_head: int,
    bias: bool,
    dropout: float,
    gated: bool,
    per_channel: bool,
    kind: str,
) -> Any:
    torch = require_torch()
    F = torch.nn.functional
    head_dim = n_embd // n_head

    class DeltaFamily(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.n_head, self.head_dim = n_head, head_dim
            self.qkv = torch.nn.Linear(n_embd, 3 * n_embd, bias=bias)
            self.w_beta = torch.nn.Linear(n_embd, n_head, bias=bias)
            self.w_alpha = None
            if gated and per_channel:
                self.w_alpha = torch.nn.Linear(n_embd, n_head * head_dim, bias=bias)
            elif gated:
                self.w_alpha = torch.nn.Linear(n_embd, n_head, bias=bias)
            self.o = torch.nn.Linear(n_embd, n_embd, bias=bias)
            self.drop = torch.nn.Dropout(dropout)
            self.gated, self.per_channel = gated, per_channel
            self.kind = kind

        def forward(self, x: Any, cache: Any = None) -> tuple[Any, Any]:
            b, t, d = x.shape
            qkv = self.qkv(x)
            q = F.normalize(F.silu(_split(qkv[:, :, :d], b, t, self.n_head, self.head_dim)), -1)
            k = F.normalize(
                F.silu(_split(qkv[:, :, d : 2 * d], b, t, self.n_head, self.head_dim)), -1
            )
            v = _split(qkv[:, :, 2 * d :], b, t, self.n_head, self.head_dim)
            beta = torch.sigmoid(self.w_beta(x)).transpose(1, 2).unsqueeze(-1)
            alpha = None
            if self.w_alpha is not None:
                raw = torch.sigmoid(self.w_alpha(x))
                if self.per_channel:
                    alpha = raw.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
                else:
                    alpha = raw.transpose(1, 2).unsqueeze(-1)
            s = (
                cache
                if cache is not None
                else torch.zeros(b, self.n_head, self.head_dim, self.head_dim, device=x.device)
            )
            outs = []
            for i in range(t):
                ki, vi, bi = k[:, :, i : i + 1], v[:, :, i : i + 1], beta[:, :, i : i + 1]
                if alpha is not None:
                    ai = (
                        alpha[:, :, i : i + 1]
                        if not self.per_channel
                        else alpha[:, :, i].unsqueeze(-1)
                    )
                    s = ai * s
                u = bi * (vi - (ki @ s))
                s = s + ki.transpose(-1, -2) @ u
                outs.append(q[:, :, i : i + 1] @ s)
            return self.drop(self.o(_merge(torch.cat(outs, 2), b, t, d))), s

    return DeltaFamily()


@ATTENTION.register("delta")
def build_delta(
    *, n_embd: int, n_head: int, bias: bool = True, dropout: float = 0.0, **_: Any
) -> Any:
    return _delta_family(
        n_embd=n_embd,
        n_head=n_head,
        bias=bias,
        dropout=dropout,
        gated=False,
        per_channel=False,
        kind="delta",
    )


@ATTENTION.register("gated_delta")
def build_gated_delta(
    *, n_embd: int, n_head: int, bias: bool = True, dropout: float = 0.0, **_: Any
) -> Any:
    return _delta_family(
        n_embd=n_embd,
        n_head=n_head,
        bias=bias,
        dropout=dropout,
        gated=True,
        per_channel=False,
        kind="gated_delta",
    )


@ATTENTION.register("kda")
def build_kda(
    *, n_embd: int, n_head: int, bias: bool = True, dropout: float = 0.0, **_: Any
) -> Any:
    """Kimi Delta Attention: per-channel decay + delta rule."""
    return _delta_family(
        n_embd=n_embd,
        n_head=n_head,
        bias=bias,
        dropout=dropout,
        gated=True,
        per_channel=True,
        kind="kda",
    )


@ATTENTION.register("mla")
def build_mla(
    *,
    n_embd: int,
    n_head: int,
    dropout: float = 0.0,
    bias: bool = True,
    block_size: int = 128,
    kv_rank: int | None = None,
    q_lora_rank: int | None = None,
    gated: bool = False,
    **_: Any,
) -> Any:
    """Multi-head Latent Attention with optional q-LoRA and output gating."""
    torch = require_torch()
    head_dim = n_embd // n_head
    kv_rank = kv_rank or max(head_dim, n_embd // 4)

    class MLA(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.n_head, self.head_dim = n_head, head_dim
            self.kv_down = torch.nn.Linear(n_embd, kv_rank, bias=bias)
            self.k_up = torch.nn.Linear(kv_rank, n_head * head_dim, bias=bias)
            self.v_up = torch.nn.Linear(kv_rank, n_head * head_dim, bias=bias)
            if q_lora_rank is not None:
                self.q_down = torch.nn.Linear(n_embd, q_lora_rank, bias=bias)
                self.q_up = torch.nn.Linear(q_lora_rank, n_head * head_dim, bias=bias)
                self.q_proj = None
            else:
                self.q_proj = torch.nn.Linear(n_embd, n_head * head_dim, bias=bias)
                self.q_down = self.q_up = None
            self.o = torch.nn.Linear(n_embd, n_embd, bias=bias)
            self.gate = torch.nn.Linear(n_embd, n_embd, bias=bias) if gated else None
            self.attn_drop = torch.nn.Dropout(dropout)
            mask = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
            self.register_buffer("bias_mask", mask, persistent=False)
            self.kind = "mla"

        def forward(self, x: Any, cache: Any = None) -> tuple[Any, Any]:
            b, t, c = x.size()
            latent = self.kv_down(x)
            if cache is not None:
                latent = torch.cat((cache, latent), dim=1)
            past = latent.size(1)
            k = _split(self.k_up(latent), b, past, self.n_head, self.head_dim)
            v = _split(self.v_up(latent), b, past, self.n_head, self.head_dim)
            if self.q_proj is not None:
                q = _split(self.q_proj(x), b, t, self.n_head, self.head_dim)
            else:
                q = _split(self.q_up(self.q_down(x)), b, t, self.n_head, self.head_dim)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            q0 = past - t
            att = att.masked_fill(self.bias_mask[:, :, q0:past, :past] == 0, float("-inf"))
            y = _merge(self.attn_drop(att.softmax(-1)) @ v, b, t, c)
            y = self.o(y)
            if self.gate is not None:
                y = y * torch.sigmoid(self.gate(x))
            return y, latent

    return MLA()

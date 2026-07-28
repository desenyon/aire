"""Single decoder block: compose any attention + ffn + norms + residual."""

from __future__ import annotations

from typing import Any

from aire.ml.arch.registry import ATTENTION, FFN, NORM, RESIDUAL, ensure_builtins_registered
from aire.ml.arch.specs import LayerSpec
from aire.ml.arch.torch_util import require_torch


def build_block(
    *,
    n_embd: int,
    n_head: int,
    attention: str = "mha",
    ffn: str = "mlp",
    norm: str = "layernorm",
    residual: str = "add",
    dropout: float = 0.0,
    bias: bool = True,
    block_size: int = 128,
    attention_options: dict[str, Any] | None = None,
    ffn_options: dict[str, Any] | None = None,
    norm_options: dict[str, Any] | None = None,
    residual_options: dict[str, Any] | None = None,
) -> Any:
    """Construct one pre-norm decoder block from independently chosen parts."""
    ensure_builtins_registered()
    torch = require_torch()
    attn_opts = {
        "n_embd": n_embd,
        "n_head": n_head,
        "dropout": dropout,
        "bias": bias,
        "block_size": block_size,
        **(attention_options or {}),
    }
    ffn_opts = {
        "n_embd": n_embd,
        "dropout": dropout,
        "bias": bias,
        **(ffn_options or {}),
    }
    norm_opts = {"n_embd": n_embd, **(norm_options or {})}
    res_opts = {"n_embd": n_embd, **(residual_options or {})}

    class DecoderBlock(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.ln_1 = NORM.create(norm, **norm_opts)
            self.attn = ATTENTION.create(attention, **attn_opts)
            self.ln_2 = NORM.create(norm, **norm_opts)
            self.mlp = FFN.create(ffn, **ffn_opts)
            self.residual = RESIDUAL.create(residual, **res_opts)
            self.attention_kind = attention
            self.ffn_kind = ffn
            self.norm_kind = norm
            self.residual_kind = residual

        def forward(self, x: Any, cache: Any = None) -> tuple[Any, Any]:
            attn_out, new_cache = self.attn(self.ln_1(x), cache)
            x = self.residual(x, attn_out)
            x = self.residual(x, self.mlp(self.ln_2(x)))
            return x, new_cache

    return DecoderBlock()


def block_from_spec(
    layer: LayerSpec,
    *,
    n_embd: int,
    n_head: int,
    dropout: float = 0.0,
    bias: bool = True,
    block_size: int = 128,
) -> Any:
    return build_block(
        n_embd=n_embd,
        n_head=n_head,
        attention=layer.attention,
        ffn=layer.ffn,
        norm=layer.norm,
        residual=layer.residual,
        dropout=dropout,
        bias=bias,
        block_size=block_size,
        attention_options=layer.attention_options,
        ffn_options=layer.ffn_options,
        norm_options=layer.norm_options,
        residual_options=layer.residual_options,
    )

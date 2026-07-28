"""Public compose / register API — build architectures from parts, not themes."""

from __future__ import annotations

from typing import Any

from aire.ml.arch.block import build_block
from aire.ml.arch.registry import (
    ARCHITECTURE,
    ATTENTION,
    EMBED,
    FFN,
    HEAD,
    NORM,
    RESIDUAL,
    ensure_builtins_registered,
)
from aire.ml.arch.specs import LayerSpec, StackSpec
from aire.ml.arch.stack import build_stack
from aire.ml.arch.torch_util import require_torch, torch_available


def attention(kind: str, **options: Any) -> Any:
    """Construct a single attention block by registered name."""
    ensure_builtins_registered()
    require_torch()
    return ATTENTION.create(kind, **options)


def ffn(kind: str, **options: Any) -> Any:
    ensure_builtins_registered()
    require_torch()
    return FFN.create(kind, **options)


def norm(kind: str, **options: Any) -> Any:
    ensure_builtins_registered()
    require_torch()
    return NORM.create(kind, **options)


def residual(kind: str, **options: Any) -> Any:
    ensure_builtins_registered()
    require_torch()
    return RESIDUAL.create(kind, **options)


def embed(kind: str, **options: Any) -> Any:
    ensure_builtins_registered()
    require_torch()
    return EMBED.create(kind, **options)


def head(kind: str, **options: Any) -> Any:
    ensure_builtins_registered()
    require_torch()
    return HEAD.create(kind, **options)


def block(**options: Any) -> Any:
    """Construct one decoder block from independently chosen parts."""
    require_torch()
    return build_block(**options)


def compose(
    layers: list[dict[str, Any] | LayerSpec] | list[dict[str, Any]],
    *,
    vocab_size: int = 256,
    n_embd: int = 64,
    n_head: int = 4,
    block_size: int = 128,
    dropout: float = 0.0,
    bias: bool = True,
    embed: str = "learned",
    head: str = "tied",
    attn_res_every: int | None = None,
    name: str = "custom",
    **options: Any,
) -> Any:
    """Compose an arbitrary language model from per-layer block choices.

    Each layer dict may set ``attention``, ``ffn``, ``norm``, ``residual`` and
    ``*_options`` freely — layers need not share a mechanism.
    """
    require_torch()
    layer_specs = [
        layer if isinstance(layer, LayerSpec) else LayerSpec.model_validate(layer)
        for layer in layers
    ]
    spec = StackSpec(
        layers=layer_specs,
        vocab_size=vocab_size,
        n_embd=n_embd,
        n_head=n_head,
        block_size=block_size,
        dropout=dropout,
        bias=bias,
        embed=embed,
        head=head,
        attn_res_every=attn_res_every,
        name=name,
        **{k: v for k, v in options.items() if k in StackSpec.model_fields},
    )
    return build_stack(spec)


def register_attention(name: str, factory: Any | None = None, *, replace: bool = False) -> Any:
    """Register a custom attention factory (decorator or direct)."""
    ensure_builtins_registered()
    return ATTENTION.register(name, factory, replace=replace)


def register_ffn(name: str, factory: Any | None = None, *, replace: bool = False) -> Any:
    ensure_builtins_registered()
    return FFN.register(name, factory, replace=replace)


def register_norm(name: str, factory: Any | None = None, *, replace: bool = False) -> Any:
    ensure_builtins_registered()
    return NORM.register(name, factory, replace=replace)


def register_residual(name: str, factory: Any | None = None, *, replace: bool = False) -> Any:
    ensure_builtins_registered()
    return RESIDUAL.register(name, factory, replace=replace)


def register_architecture(name: str, factory: Any | None = None, *, replace: bool = False) -> Any:
    """Register a full architecture factory under ``AI.ml.arch.create(name)``."""
    ensure_builtins_registered()
    return ARCHITECTURE.register(name, factory, replace=replace)


def create(name: str, **options: Any) -> Any:
    """Build a registered architecture (or fall back to compose if name is ``custom``)."""
    ensure_builtins_registered()
    require_torch()
    if name == "custom":
        layers = options.pop("layers", None)
        if not layers:
            raise ValueError("compose custom architectures with layers=[...]")
        return compose(layers, **options)
    return ARCHITECTURE.create(name, **options)


def available() -> dict[str, list[str]]:
    ensure_builtins_registered()
    return {
        "attention": ATTENTION.names(),
        "ffn": FFN.names(),
        "norm": NORM.names(),
        "residual": RESIDUAL.names(),
        "embed": EMBED.names(),
        "head": HEAD.names(),
        "architecture": ARCHITECTURE.names(),
    }


def describe() -> dict[str, Any]:
    ensure_builtins_registered()
    return {
        "kind": "ml.arch",
        "torch_available": torch_available(),
        "blocks": available(),
        "usage": {
            "attention": 'AI.ml.arch.attention("mha", n_embd=64, n_head=4)',
            "ffn": 'AI.ml.arch.ffn("moe", n_embd=64, n_experts=8)',
            "block": 'AI.ml.arch.block(attention="kda", ffn="latent_moe", n_embd=64, n_head=4)',
            "compose": "AI.ml.arch.compose(layers=[{attention, ffn}, ...], n_embd=64, n_head=4)",
            "register": (
                "@AI.ml.arch.register_attention('mine') / register_ffn / "
                "register_architecture"
            ),
        },
    }

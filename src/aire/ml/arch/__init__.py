"""Composable neural architecture blocks — build any stack from parts.

Create and swap attention, FFN, norm, residual, embed, and head blocks;
compose arbitrary per-layer stacks; register your own factories::

    attn = AI.ml.arch.attention("delta", n_embd=64, n_head=4)
    ffn = AI.ml.arch.ffn("moe", n_embd=64, n_experts=8, top_k=2)
    model = AI.ml.arch.compose(
        layers=[
            {"attention": "mha", "ffn": "mlp"},
            {"attention": "kda", "ffn": "latent_moe", "ffn_options": {"n_experts": 8}},
            {"attention": "mla", "ffn": "swiglu", "attention_options": {"gated": True}},
        ],
        n_embd=64, n_head=4, vocab_size=128,
    )

Requires ``pip install aire[torch]`` to materialize modules.
"""

# Load example composed architectures into the registry.
from aire.ml.arch import recipes as _recipes
from aire.ml.arch.compose import (
    attention,
    available,
    block,
    compose,
    create,
    describe,
    embed,
    ffn,
    head,
    norm,
    register_architecture,
    register_attention,
    register_ffn,
    register_norm,
    register_residual,
    residual,
)
from aire.ml.arch.specs import (
    AttentionSpec,
    EmbedSpec,
    FFNSpec,
    HeadSpec,
    LayerSpec,
    NormSpec,
    ResidualSpec,
    StackSpec,
)
from aire.ml.arch.torch_util import torch_available

_ = _recipes

__all__ = [
    "AttentionSpec",
    "EmbedSpec",
    "FFNSpec",
    "HeadSpec",
    "LayerSpec",
    "NormSpec",
    "ResidualSpec",
    "StackSpec",
    "attention",
    "available",
    "block",
    "compose",
    "create",
    "describe",
    "embed",
    "ffn",
    "head",
    "norm",
    "register_architecture",
    "register_attention",
    "register_ffn",
    "register_norm",
    "register_residual",
    "residual",
    "torch_available",
]

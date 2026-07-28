"""Optional example architectures built BY COMPOSING blocks (not themes).

These exist only as references showing how to assemble parts. Prefer
``AI.ml.arch.compose(layers=[...])`` or ``register_architecture`` for
your own designs.
"""

from __future__ import annotations

from typing import Any

from aire.ml.arch.compose import compose, register_architecture


@register_architecture("uniform_mha")
def uniform_mha(*, n_layer: int = 4, **options: Any) -> Any:
    """All layers: causal MHA + MLP (GPT-2-shaped, fully overrideable)."""
    layers: list[dict[str, Any]] = [{"attention": "mha", "ffn": "mlp"} for _ in range(n_layer)]
    return compose(layers, name="uniform_mha", **options)


@register_architecture("uniform_linear")
def uniform_linear(*, n_layer: int = 4, **options: Any) -> Any:
    layers: list[dict[str, Any]] = [{"attention": "linear", "ffn": "mlp"} for _ in range(n_layer)]
    return compose(layers, name="uniform_linear", **options)


@register_architecture("uniform_delta")
def uniform_delta(*, n_layer: int = 4, attention: str = "delta", **options: Any) -> Any:
    layers: list[dict[str, Any]] = [{"attention": attention, "ffn": "mlp"} for _ in range(n_layer)]
    return compose(layers, name=f"uniform_{attention}", **options)


@register_architecture("hybrid_cycle")
def hybrid_cycle(
    *,
    n_layer: int = 8,
    cycle: list[str] | None = None,
    ffn_first: str = "mlp",
    ffn_rest: str = "moe",
    **options: Any,
) -> Any:
    """Interleave attention kinds in a repeating cycle; first FFN dense, rest MoE.

    Default cycle: kda, kda, kda, mla — override ``cycle`` freely.
    """
    cycle = cycle or ["kda", "kda", "kda", "mla"]
    layers: list[dict[str, Any]] = []
    q_lora = options.pop("q_lora_rank", None)
    for i in range(n_layer):
        attn = cycle[i % len(cycle)]
        attn_opts: dict[str, Any] = {}
        if attn == "mla":
            attn_opts["gated"] = True
            if q_lora is not None:
                attn_opts["q_lora_rank"] = q_lora
        layers.append(
            {
                "attention": attn,
                "ffn": ffn_first if i == 0 else ffn_rest,
                "attention_options": attn_opts,
            }
        )
    return compose(layers, name="hybrid_cycle", **options)

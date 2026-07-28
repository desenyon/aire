"""Pydantic specs for individual architectural blocks (fully customizable)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AttentionSpec(BaseModel):
    """One attention / memory mechanism instance."""

    kind: str = "mha"
    n_embd: int = Field(default=64, ge=8)
    n_head: int = Field(default=4, ge=1)
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)
    bias: bool = True
    block_size: int = Field(default=128, ge=1)
    # mechanism-specific knobs (passed through to the factory)
    options: dict[str, Any] = Field(default_factory=dict)

    def head_dim(self) -> int:
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd ({self.n_embd}) not divisible by n_head ({self.n_head})")
        return self.n_embd // self.n_head


class FFNSpec(BaseModel):
    kind: str = "mlp"
    n_embd: int = Field(default=64, ge=8)
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)
    bias: bool = True
    mult: int = Field(default=4, ge=1)  # hidden = mult * n_embd for dense MLP
    options: dict[str, Any] = Field(default_factory=dict)


class NormSpec(BaseModel):
    kind: str = "layernorm"
    n_embd: int = Field(default=64, ge=8)
    options: dict[str, Any] = Field(default_factory=dict)


class ResidualSpec(BaseModel):
    kind: str = "add"  # add | attn_res
    n_embd: int = Field(default=64, ge=8)
    options: dict[str, Any] = Field(default_factory=dict)


class EmbedSpec(BaseModel):
    kind: str = "learned"  # learned absolute positions
    vocab_size: int = Field(default=256, ge=2)
    n_embd: int = Field(default=64, ge=8)
    block_size: int = Field(default=128, ge=1)
    dropout: float = 0.0
    options: dict[str, Any] = Field(default_factory=dict)


class HeadSpec(BaseModel):
    kind: str = "lm"  # lm | tied
    vocab_size: int = Field(default=256, ge=2)
    n_embd: int = Field(default=64, ge=8)
    bias: bool = False
    options: dict[str, Any] = Field(default_factory=dict)


class LayerSpec(BaseModel):
    """One decoder layer: fully independent attention + ffn + norms."""

    attention: str = "mha"
    ffn: str = "mlp"
    norm: str = "layernorm"
    residual: str = "add"
    attention_options: dict[str, Any] = Field(default_factory=dict)
    ffn_options: dict[str, Any] = Field(default_factory=dict)
    norm_options: dict[str, Any] = Field(default_factory=dict)
    residual_options: dict[str, Any] = Field(default_factory=dict)


class StackSpec(BaseModel):
    """Full stack: arbitrary per-layer specs + shared embed/head/dims."""

    layers: list[LayerSpec]
    vocab_size: int = 256
    n_embd: int = 64
    n_head: int = 4
    block_size: int = 128
    dropout: float = 0.0
    bias: bool = True
    embed: str = "learned"
    head: str = "tied"
    embed_options: dict[str, Any] = Field(default_factory=dict)
    head_options: dict[str, Any] = Field(default_factory=dict)
    # optional depth-wise AttnRes every N layers (None = disabled)
    attn_res_every: int | None = None
    name: str = "custom"

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "stack_spec",
            "name": self.name,
            "n_layer": len(self.layers),
            "dims": {
                "vocab_size": self.vocab_size,
                "n_embd": self.n_embd,
                "n_head": self.n_head,
                "block_size": self.block_size,
            },
            "embed": self.embed,
            "head": self.head,
            "attn_res_every": self.attn_res_every,
            "layers": [layer.model_dump(mode="json") for layer in self.layers],
        }

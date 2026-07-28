"""Embedding blocks."""

from __future__ import annotations

from typing import Any

from aire.ml.arch.registry import EMBED
from aire.ml.arch.torch_util import require_torch


@EMBED.register("learned")
def build_learned(
    *,
    vocab_size: int,
    n_embd: int,
    block_size: int,
    dropout: float = 0.0,
    **_: Any,
) -> Any:
    """Token + learned absolute position embeddings."""
    torch = require_torch()

    class LearnedEmbed(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.wte = torch.nn.Embedding(vocab_size, n_embd)
            self.wpe = torch.nn.Embedding(block_size, n_embd)
            self.drop = torch.nn.Dropout(dropout)
            self.block_size = block_size
            self.kind = "learned"

        def forward(self, idx: Any, *, start: int = 0) -> Any:
            _b, t = idx.shape
            if start + t > self.block_size:
                raise ValueError(f"length {start + t} exceeds block_size {self.block_size}")
            pos = torch.arange(start, start + t, device=idx.device)
            return self.drop(self.wte(idx) + self.wpe(pos))

    return LearnedEmbed()

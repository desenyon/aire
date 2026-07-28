"""Output head blocks."""

from __future__ import annotations

from typing import Any

from aire.ml.arch.registry import HEAD
from aire.ml.arch.torch_util import require_torch


@HEAD.register("lm")
def build_lm(*, vocab_size: int, n_embd: int, bias: bool = False, **_: Any) -> Any:
    torch = require_torch()
    return torch.nn.Linear(n_embd, vocab_size, bias=bias)


@HEAD.register("tied")
def build_tied(*, vocab_size: int, n_embd: int, bias: bool = False, **_: Any) -> Any:
    """LM head intended to share weights with token embeddings (tied later)."""
    return build_lm(vocab_size=vocab_size, n_embd=n_embd, bias=bias)

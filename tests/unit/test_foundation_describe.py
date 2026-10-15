"""Foundation describe honesty."""

from __future__ import annotations

import importlib.util

import pytest

from aire.training.foundation import create_foundation


def test_foundation_describe_kind() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("install the torch profile")
    model = create_foundation("gpt2", n_layer=1, n_embd=16, n_head=2, vocab_size=16, max_seq_len=16)
    desc = model.describe()
    assert desc["kind"] == "foundation_toy_architecture"
    assert "NOT pretrained" in desc.get("honesty", "")

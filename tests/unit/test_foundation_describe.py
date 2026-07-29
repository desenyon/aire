"""Foundation describe honesty."""

from __future__ import annotations

import pytest

from aire.training.foundation import create_foundation


def test_foundation_describe_kind() -> None:
    try:
        model = create_foundation("gpt2", n_layer=2)
    except Exception as exc:  # torch / arch optional
        pytest.skip(f"foundation unavailable: {exc}")
    desc = model.describe()
    assert desc["kind"] == "foundation_toy_architecture"
    assert "NOT pretrained" in desc.get("honesty", "")

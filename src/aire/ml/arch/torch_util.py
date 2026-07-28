"""Shared lazy-torch bootstrap for arch / optim / loss."""

from __future__ import annotations

import importlib.util
from typing import Any

from aire.core.errors import ConfigurationError


def require_torch() -> Any:
    if importlib.util.find_spec("torch") is None:
        raise ConfigurationError(
            "PyTorch is required: pip install 'aire[torch]'",
            code="ml.torch_missing",
            context={"backend": "torch"},
        )
    import torch  # type: ignore[import-not-found,unused-ignore]

    return torch


def torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None

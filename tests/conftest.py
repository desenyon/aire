"""Shared fixtures. Tests use asyncio.run via the `arun` helper — no plugins."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any

import pytest

from aire.core.config import Settings
from aire.core.runtime import Runtime
from aire.models.builtin import EchoModel, HashingEmbedder
from aire.rag.store import register as register_local_store


def arun(coro: Coroutine[Any, Any, Any]) -> Any:
    """Run a coroutine in a fresh event loop."""
    return asyncio.run(coro)


@pytest.fixture()
def runtime() -> Runtime:
    rt = Runtime(Settings(project="test-project"))
    register_local_store(rt)
    return rt


@pytest.fixture()
def echo() -> EchoModel:
    return EchoModel()


@pytest.fixture()
def embedder() -> HashingEmbedder:
    return HashingEmbedder()


@pytest.fixture()
def docs() -> list[str]:
    return [
        "The refund policy allows returns within 30 days of purchase with a receipt.",
        "Authentication uses OAuth2 bearer tokens issued by the identity service.",
        "Rate limits are 100 requests per minute per API key on the standard plan.",
        "Data is encrypted at rest using AES-256 and in transit using TLS 1.3.",
    ]

"""Shared fixtures for aire offline tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from aire import AI
from aire.models.base import Model
from aire.models.builtin import EchoModel, HashingEmbedder


@pytest.fixture
def echo_model() -> Model:
    return EchoModel()


@pytest.fixture
def hashing_embedder() -> HashingEmbedder:
    return HashingEmbedder()


@pytest.fixture
def mock_echo() -> Model:
    return AI.models.use_sync("mock:echo")


@pytest.fixture
def tmp_session_path(tmp_path: Path) -> Path:
    return tmp_path / "session.json"


@pytest.fixture
def sandbox_root(tmp_path: Path) -> Path:
    root = tmp_path / "sandbox"
    root.mkdir()
    (root / "hello.txt").write_text("hello from sandbox\n", encoding="utf-8")
    return root

"""Model protocol contract: EchoModel.generate."""

from __future__ import annotations

from aire.models.base import run_sync
from aire.models.builtin import EchoModel
from aire.models.types import GenerationRequest


def test_echo_model_generate() -> None:
    model = EchoModel()
    result = run_sync(model.generate(GenerationRequest.of("ping")))
    assert result.text == "ping"
    assert result.model.startswith("mock:")
    assert result.usage.input_tokens >= 0

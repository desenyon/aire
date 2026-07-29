"""Offline unit tests for OpenAI media capability routing."""

from __future__ import annotations

from aire.core.types import Capability
from aire.integrations.openai_media import (
    capabilities_for_openai_model,
    is_asr_model,
    is_image_model,
    is_tts_model,
)


def test_tts_model_detection() -> None:
    assert is_tts_model("tts-1")
    assert is_tts_model("tts-1-hd")
    assert not is_tts_model("gpt-4o-mini")


def test_asr_model_detection() -> None:
    assert is_asr_model("whisper-1")
    assert is_asr_model("gpt-4o-transcribe")
    assert not is_asr_model("gpt-4o")


def test_image_model_detection() -> None:
    assert is_image_model("dall-e-3")
    assert is_image_model("gpt-image-1")
    assert not is_image_model("gpt-4o")


def test_capabilities_for_media_models() -> None:
    assert Capability.TEXT_TO_SPEECH in capabilities_for_openai_model("tts-1")
    assert Capability.SPEECH_RECOGNITION in capabilities_for_openai_model("whisper-1")
    assert Capability.IMAGE_GENERATION in capabilities_for_openai_model("dall-e-3")
    assert Capability.VISION_INPUT in capabilities_for_openai_model("gpt-4o")
    assert Capability.TEXT_GENERATION in capabilities_for_openai_model("gpt-4o-mini")


def test_echo_tts_is_stub() -> None:
    from aire.audio.voice import EchoTTSBackend
    from aire.models.base import run_sync

    result = run_sync(EchoTTSBackend().synthesize("hello"))
    assert result.stub is True
    assert result.backend == "echo"


def test_pdf_describe_mentions_ocr() -> None:
    from aire.docs.pdf import describe

    info = describe()
    assert info["ocr"]["extra"] == "aire[ocr]"
    assert "text layer" in info["honesty"]


def test_video_pipeline_offline_stub() -> None:
    from aire.models.base import run_sync
    from aire.vision.video import VideoPipeline

    summary = run_sync(VideoPipeline().summarize("https://example.com/clip.mp4"))
    assert summary.metadata.get("stub") is True
    assert summary.model == "stub"

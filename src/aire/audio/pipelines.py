"""Audio pipelines: speech recognition and text-to-speech delegation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel

from aire.core.content import AudioContent
from aire.core.errors import NotFoundError
from aire.core.types import Capability
from aire.models.base import Model
from aire.multimodal.conversions import transcribe


class TranscriptionResult(BaseModel):
    text: str
    model: str = "unknown"
    duration_s: float | None = None


class AudioPipeline:
    """Speech-to-text and text-to-speech through one interface."""

    def __init__(self, model: Model) -> None:
        self.model = model

    async def transcribe(self, audio: str | Path | AudioContent) -> TranscriptionResult:
        """Transcribe an audio file/content via a speech-recognition model."""
        if not self.model.info.supports(Capability.SPEECH_RECOGNITION):
            raise NotFoundError(
                "capability",
                str(Capability.SPEECH_RECOGNITION),
                context={"model": self.model.info.ref},
            )
        content = audio if isinstance(audio, AudioContent) else _to_audio(audio)
        result = await transcribe(self.model, content)
        return TranscriptionResult(text=result.text, model=self.model.info.ref)

    def describe(self) -> dict[str, Any]:
        return {"kind": "audio_pipeline", "model": self.model.info.ref}


def _to_audio(audio: str | Path) -> AudioContent:
    value = str(audio)
    if value.startswith(("http://", "https://")):
        return AudioContent.from_uri(value)
    return AudioContent.from_file(value)

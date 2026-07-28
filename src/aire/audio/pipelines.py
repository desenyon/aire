"""Audio pipelines: speech recognition and text-to-speech delegation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel

from aire.core.content import AudioContent, Message, TextContent
from aire.core.errors import NotFoundError
from aire.core.types import Capability
from aire.models.base import Model
from aire.models.types import GenerationRequest
from aire.multimodal.conversions import transcribe


class TranscriptionResult(BaseModel):
    text: str
    model: str = "unknown"
    duration_s: float | None = None


class SynthesisResult(BaseModel):
    text: str
    audio_uri: str | None = None
    model: str = "unknown"
    format: str = "wav"


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

    async def synthesize(self, text: str, *, voice: str = "default") -> SynthesisResult:
        """Text-to-speech. Returns a URI when the model emits one; otherwise a stub."""
        if not self.model.info.supports(Capability.TEXT_TO_SPEECH):
            raise NotFoundError(
                "capability",
                str(Capability.TEXT_TO_SPEECH),
                context={"model": self.model.info.ref},
            )
        request = GenerationRequest(
            messages=[
                Message(
                    role="user",
                    content=[
                        TextContent(
                            text=(
                                f"Synthesize speech (voice={voice}) for the following text "
                                "and return an audio URI or description:\n"
                                f"{text}"
                            )
                        )
                    ],
                )
            ]
        )
        result = await self.model.generate(request)
        raw = result.text.strip()
        uri = None
        for prefix in ("https://", "http://", "data:audio/", "file:"):
            if prefix in raw:
                start = raw.index(prefix)
                uri = raw[start:].split()[0].rstrip(".,)\"'")
                break
        return SynthesisResult(text=text, audio_uri=uri, model=self.model.info.ref)

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "audio_pipeline",
            "model": self.model.info.ref,
            "capabilities": [str(c) for c in self.model.info.capabilities],
        }


def _to_audio(audio: str | Path) -> AudioContent:
    value = str(audio)
    if value.startswith(("http://", "https://")):
        return AudioContent.from_uri(value)
    return AudioContent.from_file(value)

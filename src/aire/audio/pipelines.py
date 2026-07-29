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
    stub: bool = False


class SynthesisResult(BaseModel):
    text: str
    audio_uri: str | None = None
    audio: AudioContent | None = None
    model: str = "unknown"
    format: str = "wav"
    stub: bool = False


class AudioPipeline:
    """Speech-to-text and text-to-speech through one interface.

    Prefer OpenAI media models (``openai:whisper-1``, ``openai:tts-1``) which
    hit real ``/audio/*`` endpoints. Capability-gated chat models remain a
    fallback for experimental providers.
    """

    def __init__(self, model: Model, *, api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key

    async def transcribe(self, audio: str | Path | AudioContent) -> TranscriptionResult:
        """Transcribe audio via Whisper / ASR media endpoint or capability path."""
        if self._try_openai_asr():
            from aire.integrations.openai_media import openai_transcribe

            name = self.model.info.ref.split(":", 1)[-1]
            text = await openai_transcribe(
                audio,
                model=name,
                api_key=self.api_key or self._openai_key(),
                base_url=self._openai_base(),
                client=self._openai_http(),
            )
            return TranscriptionResult(text=text, model=self.model.info.ref, stub=False)

        if not self.model.info.supports(Capability.SPEECH_RECOGNITION):
            raise NotFoundError(
                "capability",
                str(Capability.SPEECH_RECOGNITION),
                context={"model": self.model.info.ref},
            )
        content = audio if isinstance(audio, AudioContent) else _to_audio(audio)
        result = await transcribe(self.model, content)
        return TranscriptionResult(text=result.text, model=self.model.info.ref)

    async def synthesize(self, text: str, *, voice: str = "alloy") -> SynthesisResult:
        """Text-to-speech via OpenAI ``/audio/speech`` or capability-gated fallback."""
        if self._try_openai_tts():
            from aire.integrations.openai_media import openai_tts

            name = self.model.info.ref.split(":", 1)[-1]
            audio = await openai_tts(
                text,
                model=name,
                voice=voice,
                api_key=self.api_key or self._openai_key(),
                base_url=self._openai_base(),
                client=self._openai_http(),
            )
            uri = None
            if audio.data is not None:
                import base64

                b64 = base64.b64encode(audio.data).decode("ascii")
                media = audio.media_type or "audio/mpeg"
                uri = f"data:{media};base64,{b64}"
            return SynthesisResult(
                text=text,
                audio_uri=uri,
                audio=audio,
                model=self.model.info.ref,
                format=(audio.media_type or "audio/mpeg").split("/")[-1],
                stub=False,
            )

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
        return SynthesisResult(
            text=text,
            audio_uri=uri,
            model=self.model.info.ref,
            stub=uri is None,
        )

    def _try_openai_tts(self) -> bool:
        from aire.integrations.openai_media import is_tts_model

        ref = self.model.info.ref
        provider, _, name = ref.partition(":")
        return provider in {"openai", "azure"} and is_tts_model(name or ref)

    def _try_openai_asr(self) -> bool:
        from aire.integrations.openai_media import is_asr_model

        ref = self.model.info.ref
        provider, _, name = ref.partition(":")
        return provider in {"openai", "azure"} and is_asr_model(name or ref)

    def _openai_key(self) -> str | None:
        client = getattr(self.model, "_client", None)
        headers = getattr(getattr(client, "raw", None), "headers", None)
        if headers is not None:
            auth = headers.get("Authorization") or headers.get("authorization")
            if auth and str(auth).lower().startswith("bearer "):
                return str(auth).split(" ", 1)[1]
        return None

    def _openai_base(self) -> str | None:
        client = getattr(self.model, "_client", None)
        raw = getattr(client, "raw", None)
        base = getattr(raw, "base_url", None)
        return str(base).rstrip("/") if base else None

    def _openai_http(self) -> Any:
        client = getattr(self.model, "_client", None)
        return getattr(client, "raw", None)

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "audio_pipeline",
            "model": self.model.info.ref,
            "capabilities": [str(c) for c in self.model.info.capabilities],
            "openai_media": self._try_openai_tts() or self._try_openai_asr(),
        }


def _to_audio(audio: str | Path) -> AudioContent:
    value = str(audio)
    if value.startswith(("http://", "https://")):
        return AudioContent.from_uri(value)
    return AudioContent.from_file(value)

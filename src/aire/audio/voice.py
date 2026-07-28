"""Voice agent pipeline: ASR -> agent -> TTS (stub/delegate)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from aire.agents.agent import Agent
from aire.audio.pipelines import AudioPipeline, TranscriptionResult
from aire.core.content import AudioContent
from aire.core.errors import ConfigurationError
from aire.models.base import Model


class VoiceTurn(BaseModel):
    transcript: str
    response_text: str
    audio: AudioContent | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TTSResult(BaseModel):
    text: str
    audio: AudioContent | None = None
    backend: str = "echo"


class TTSBackend:
    """Text-to-speech interface. Default is an offline echo stub."""

    def __init__(self, model: Model | None = None, *, backend: str = "echo") -> None:
        self.model = model
        self.backend = backend

    async def synthesize(self, text: str) -> TTSResult:
        if self.backend == "echo" or self.model is None:
            # Offline stub: encode text as a data URI placeholder (no audio bytes).
            return TTSResult(
                text=text,
                audio=AudioContent.from_uri(f"data:text/plain,{text[:200]}"),
                backend="echo",
            )
        from aire.audio.pipelines import AudioPipeline
        from aire.core.types import Capability

        if self.model.info.supports(Capability.TEXT_TO_SPEECH):
            result = await AudioPipeline(self.model).synthesize(text)
            audio = (
                AudioContent.from_uri(result.audio_uri)
                if result.audio_uri
                else AudioContent.from_uri(f"tts://{self.model.info.ref}")
            )
            return TTSResult(text=text, audio=audio, backend=self.model.info.ref)
        # Last resort: prompt a text model for a URI description.
        ask = getattr(self.model, "ask", None)
        if callable(ask):
            _ = await ask(f"[tts] {text}")
        return TTSResult(
            text=text,
            audio=AudioContent.from_uri(f"tts://{self.model.info.ref}"),
            backend=self.model.info.ref,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "tts",
            "backend": self.backend,
            "model": self.model.info.ref if self.model else None,
        }


class VoiceAgent:
    """ASR -> agent.run -> TTS pipeline."""

    def __init__(
        self,
        agent: Agent,
        *,
        asr: AudioPipeline | Model | None = None,
        tts: TTSBackend | Model | None = None,
    ) -> None:
        self.agent = agent
        self.asr: AudioPipeline | None
        if isinstance(asr, AudioPipeline):
            self.asr = asr
        elif isinstance(asr, Model):
            self.asr = AudioPipeline(asr)
        else:
            self.asr = None
        if isinstance(tts, TTSBackend):
            self.tts = tts
        elif isinstance(tts, Model):
            self.tts = TTSBackend(tts, backend="model")
        else:
            self.tts = TTSBackend()

    async def handle(
        self,
        audio: str | Path | AudioContent | None = None,
        *,
        text: str | None = None,
    ) -> VoiceTurn:
        if text is None:
            if audio is None:
                raise ConfigurationError(
                    "VoiceAgent.handle requires audio or text=",
                    code="audio.voice_input_missing",
                )
            if self.asr is None:
                raise ConfigurationError(
                    "ASR model required for audio input: pass asr=Model or AudioPipeline",
                    code="audio.asr_missing",
                )
            transcript_result: TranscriptionResult = await self.asr.transcribe(audio)
            transcript = transcript_result.text
        else:
            transcript = text
        result = await self.agent.run(transcript)
        tts = await self.tts.synthesize(result.output)
        return VoiceTurn(
            transcript=transcript,
            response_text=result.output,
            audio=tts.audio,
            metadata={"tts_backend": tts.backend, "agent": self.agent.name},
        )

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "voice_agent",
            "agent": self.agent.name,
            "asr": self.asr.describe() if self.asr else None,
            "tts": self.tts.describe(),
        }

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
    stub: bool = False


class EchoTTSBackend:
    """Offline echo / stub text-to-speech — does NOT produce real audio bytes.

    Encodes the input text as a ``data:text/plain`` URI placeholder. Prefer a
    model advertising ``Capability.TEXT_TO_SPEECH`` for real synthesis.
    """

    def __init__(self, model: Model | None = None, *, backend: str = "echo") -> None:
        self.model = model
        self.backend = backend

    async def synthesize(self, text: str) -> TTSResult:
        if self.backend == "echo" or self.model is None:
            return TTSResult(
                text=text,
                audio=AudioContent.from_uri(f"data:text/plain,{text[:200]}"),
                backend="echo",
                stub=True,
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
            return TTSResult(
                text=text,
                audio=audio,
                backend=self.model.info.ref,
                stub=not bool(result.audio_uri),
            )
        # Last resort: prompt a text model for a URI description (still a stub).
        ask = getattr(self.model, "ask", None)
        if callable(ask):
            _ = await ask(f"[tts] {text}")
        return TTSResult(
            text=text,
            audio=AudioContent.from_uri(f"tts://{self.model.info.ref}"),
            backend=self.model.info.ref,
            stub=True,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "echo_tts_backend",
            "stub": True,
            "backend": self.backend,
            "honesty": "offline stub — no real audio bytes unless model has TEXT_TO_SPEECH",
            "model": self.model.info.ref if self.model else None,
        }


# Public aliases: StubTTSBackend is the honesty-forward name; TTSBackend kept for compat.
StubTTSBackend = EchoTTSBackend
TTSBackend = EchoTTSBackend


class OpenAITTSBackend:
    """Real OpenAI ``/audio/speech`` TTS backend (``openai:tts-1`` / ``tts-1-hd``)."""

    def __init__(
        self,
        model: str | Model = "tts-1",
        *,
        voice: str = "alloy",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        if isinstance(model, Model):
            self.model_name = model.info.ref.split(":", 1)[-1]
            self._model: Model | None = model
        else:
            self.model_name = model
            self._model = None
        self.voice = voice
        self.api_key = api_key
        self.base_url = base_url

    async def synthesize(self, text: str) -> TTSResult:
        from aire.integrations.openai_media import openai_tts

        audio = await openai_tts(
            text,
            model=self.model_name,
            voice=self.voice,
            api_key=self.api_key,
            base_url=self.base_url,
            client=self._http(),
        )
        return TTSResult(
            text=text,
            audio=audio,
            backend=f"openai:{self.model_name}",
            stub=False,
        )

    def _http(self) -> Any:
        model = self._model
        if model is None:
            return None
        client = getattr(model, "_client", None)
        return getattr(client, "raw", None)

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "openai_tts_backend",
            "stub": False,
            "model": f"openai:{self.model_name}",
            "voice": self.voice,
        }


class VoiceAgent:
    """ASR -> agent.run -> TTS pipeline."""

    def __init__(
        self,
        agent: Agent,
        *,
        asr: AudioPipeline | Model | None = None,
        tts: EchoTTSBackend | OpenAITTSBackend | Model | None = None,
    ) -> None:
        self.agent = agent
        self.asr: AudioPipeline | None
        if isinstance(asr, AudioPipeline):
            self.asr = asr
        elif isinstance(asr, Model):
            self.asr = AudioPipeline(asr)
        else:
            self.asr = None
        if isinstance(tts, (EchoTTSBackend, OpenAITTSBackend)):
            self.tts: EchoTTSBackend | OpenAITTSBackend = tts
        elif isinstance(tts, Model):
            from aire.integrations.openai_media import is_tts_model

            name = tts.info.ref.split(":", 1)[-1]
            if tts.info.ref.startswith("openai:") and is_tts_model(name):
                self.tts = OpenAITTSBackend(tts)
            else:
                self.tts = EchoTTSBackend(tts, backend="model")
        else:
            self.tts = EchoTTSBackend()

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
            metadata={
                "tts_backend": tts.backend,
                "tts_stub": tts.stub,
                "agent": self.agent.name,
            },
        )

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "voice_agent",
            "agent": self.agent.name,
            "asr": self.asr.describe() if self.asr else None,
            "tts": self.tts.describe(),
        }

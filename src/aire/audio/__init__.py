"""Audio pipelines: transcription and speech through capability-negotiated models."""

from aire.audio.pipelines import AudioPipeline, SynthesisResult, TranscriptionResult
from aire.audio.voice import (
    EchoTTSBackend,
    OpenAITTSBackend,
    StubTTSBackend,
    TTSBackend,
    VoiceAgent,
    VoiceTurn,
)

__all__ = [
    "AudioPipeline",
    "EchoTTSBackend",
    "OpenAITTSBackend",
    "StubTTSBackend",
    "SynthesisResult",
    "TTSBackend",
    "TranscriptionResult",
    "VoiceAgent",
    "VoiceTurn",
]

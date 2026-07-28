"""Audio pipelines: transcription and speech through capability-negotiated models."""

from aire.audio.pipelines import AudioPipeline, TranscriptionResult
from aire.audio.voice import TTSBackend, VoiceAgent, VoiceTurn

__all__ = [
    "AudioPipeline",
    "TTSBackend",
    "TranscriptionResult",
    "VoiceAgent",
    "VoiceTurn",
]

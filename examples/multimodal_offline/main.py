"""Multimodal honesty demo: capability probes without calling paid APIs."""

from __future__ import annotations

from aire import AI
from aire.integrations.openai_media import capabilities_for_openai_model


def main() -> None:
    print("vision:", AI.vision.describe())
    print("audio:", AI.audio.describe())
    print("docs:", AI.docs.describe())
    for name in ("gpt-4o", "tts-1", "whisper-1", "dall-e-3", "gpt-3.5-turbo"):
        caps = sorted(c.value for c in capabilities_for_openai_model(name))
        print(f"  {name}: {caps}")
    print("Real TTS/ASR/image: use openai:* models with API key; see docs/honesty.md")


if __name__ == "__main__":
    main()

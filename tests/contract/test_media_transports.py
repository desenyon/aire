"""Actual media bytes through Aire's HTTP adapters, without paid API calls."""

import base64
import io
import json
import struct
import wave
import zlib

import httpx
import pytest

from aire.core.errors import ProviderError
from aire.integrations.openai_media import (
    image_content_from_result,
    openai_image,
    openai_transcribe,
    openai_tts,
)
from aire.models.base import run_sync


def wav_bytes():
    data = io.BytesIO()
    with wave.open(data, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(8000)
        audio.writeframes(b"\0\0" * 80)
    return data.getvalue()


def png_bytes():
    def chunk(kind, data):
        return (
            struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"\0\0\0\xff"))
        + chunk(b"IEND", b"")
    )


def test_audio_and_image_transport_roundtrip_with_valid_bytes():
    wav, png = wav_bytes(), png_bytes()
    requests = []

    def handle(request):
        requests.append(request)
        assert request.headers["authorization"] == "Bearer qa"
        if request.url.path.endswith("/audio/speech"):
            assert json.loads(request.content)["response_format"] == "wav"
            return httpx.Response(200, content=wav, headers={"content-type": "audio/wav"})
        if request.url.path.endswith("/audio/transcriptions"):
            assert wav in request.content
            assert b"whisper-1" in request.content
            return httpx.Response(200, json={"text": "recognized audio"})
        return httpx.Response(200, json={"data": [{"b64_json": base64.b64encode(png).decode()}]})

    async def check():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            speech = await openai_tts("hello", response_format="wav", api_key="qa", client=client)
            with wave.open(io.BytesIO(speech.data)) as audio:
                assert audio.getnframes() == 80
            assert await openai_transcribe(wav, api_key="qa", client=client) == "recognized audio"
            image = image_content_from_result(
                await openai_image("blue", api_key="qa", client=client)
            )
            assert image.data == png
            assert image.media_type == "image/png"

    run_sync(check())
    assert len(requests) == 3


@pytest.mark.parametrize("mode", ["empty-speech", "missing-transcript", "invalid-image"])
def test_media_adapters_reject_malformed_success_responses(mode):
    async def check():
        response = (
            httpx.Response(200, content=b"")
            if mode == "empty-speech"
            else httpx.Response(
                200,
                json={} if mode == "missing-transcript" else {"data": [{"b64_json": "ZmFrZQ=="}]},
            )
        )
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: response)) as client:
            with pytest.raises(ProviderError):
                if mode == "empty-speech":
                    await openai_tts("hello", api_key="qa", client=client)
                elif mode == "missing-transcript":
                    await openai_transcribe(wav_bytes(), api_key="qa", client=client)
                else:
                    image_content_from_result(
                        await openai_image("blue", api_key="qa", client=client)
                    )

    run_sync(check())

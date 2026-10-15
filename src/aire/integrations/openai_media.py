"""OpenAI media endpoints: TTS, Whisper ASR, and image generation.

These are real HTTP adapters against ``/v1/audio/speech``,
``/v1/audio/transcriptions``, and ``/v1/images/generations``. They work with
any OpenAI-compatible base URL that implements those routes.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any

import httpx

from aire.core.content import AudioContent, ImageContent
from aire.core.errors import AuthenticationError, ConfigurationError, ProviderError
from aire.core.types import Capability
from aire.integrations.http import DEFAULT_TIMEOUT, map_http_error

DEFAULT_BASE_URL = "https://api.openai.com/v1"

_TTS_MODELS = frozenset({"tts-1", "tts-1-hd", "gpt-4o-mini-tts"})
_ASR_MODELS = frozenset({"whisper-1", "gpt-4o-transcribe", "gpt-4o-mini-transcribe"})
_IMAGE_MODELS = frozenset({"dall-e-2", "dall-e-3", "gpt-image-1"})


def is_tts_model(name: str) -> bool:
    return name.lower() in _TTS_MODELS or name.lower().startswith("tts-")


def is_asr_model(name: str) -> bool:
    lower = name.lower()
    return lower in _ASR_MODELS or "whisper" in lower or lower.endswith("-transcribe")


def is_image_model(name: str) -> bool:
    lower = name.lower()
    return lower in _IMAGE_MODELS or lower.startswith("dall-e") or "gpt-image" in lower


def capabilities_for_openai_model(name: str) -> list[Capability]:
    """Capability set for an OpenAI model name (chat + media)."""
    if is_tts_model(name):
        return [Capability.TEXT_TO_SPEECH]
    if is_asr_model(name):
        return [Capability.SPEECH_RECOGNITION, Capability.AUDIO_INPUT]
    if is_image_model(name):
        return [Capability.IMAGE_GENERATION]
    caps = [
        Capability.TEXT_GENERATION,
        Capability.STREAMING,
        Capability.TOOL_CALLING,
        Capability.STRUCTURED_OUTPUT,
    ]
    lower = name.lower()
    vision_tokens = ("gpt-4o", "gpt-4.1", "gpt-4-turbo", "vision", "o1", "o3", "o4")
    if any(tok in lower for tok in vision_tokens):
        caps.append(Capability.VISION_INPUT)
    return caps


def _api_key(explicit: str | None = None) -> str:
    key = explicit or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise AuthenticationError(
            "openai",
            "no API key: set OPENAI_API_KEY or pass api_key=",
        )
    return key


def _base_url(explicit: str | None = None) -> str:
    return (explicit or os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")


def _media_object(response: httpx.Response) -> dict[str, Any]:
    try:
        value = response.json()
    except ValueError as exc:
        raise ProviderError("openai", "invalid media JSON response", retryable=False) from exc
    if not isinstance(value, dict):
        raise ProviderError("openai", "media response must be an object", retryable=False)
    return value


def _decode_image(value: str) -> tuple[bytes, str]:
    try:
        data = base64.b64decode(value, validate=True)
    except (ValueError, TypeError) as exc:
        raise ProviderError("openai", "invalid base64 image response", retryable=False) from exc
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return data, "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return data, "image/jpeg"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return data, "image/webp"
    raise ProviderError("openai", "response is not a supported encoded image", retryable=False)


async def openai_tts(
    text: str,
    *,
    model: str = "tts-1",
    voice: str = "alloy",
    response_format: str = "mp3",
    api_key: str | None = None,
    base_url: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> AudioContent:
    """Call ``POST /audio/speech`` and return inline audio bytes."""
    url = f"{_base_url(base_url)}/audio/speech"
    headers = {"Authorization": f"Bearer {_api_key(api_key)}"}
    payload = {
        "model": model,
        "input": text,
        "voice": voice,
        "response_format": response_format,
    }
    owns = client is None
    http = client or httpx.AsyncClient(timeout=DEFAULT_TIMEOUT)
    try:
        response = await http.post(url, json=payload, headers=headers)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise map_http_error("openai", exc) from exc
    except httpx.HTTPError as exc:
        raise ProviderError("openai", str(exc), cause=exc) from exc
    finally:
        if owns:
            await http.aclose()
    content_type = response.headers.get("content-type", "").split(";", 1)[0]
    if not response.content or (
        content_type
        and not content_type.startswith("audio/")
        and content_type != "application/octet-stream"
    ):
        raise ProviderError("openai", "speech response contains no audio", retryable=False)
    media = f"audio/{response_format}" if "/" not in response_format else response_format
    return AudioContent(
        data=response.content,
        media_type=media,
        metadata={"model": f"openai:{model}", "voice": voice, "provider": "openai"},
    )


async def openai_transcribe(
    audio: str | Path | AudioContent | bytes,
    *,
    model: str = "whisper-1",
    language: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> str:
    """Call ``POST /audio/transcriptions`` and return transcript text."""
    data, filename, media_type = _audio_parts(audio)
    url = f"{_base_url(base_url)}/audio/transcriptions"
    headers = {"Authorization": f"Bearer {_api_key(api_key)}"}
    form: dict[str, Any] = {"model": model}
    if language:
        form["language"] = language
    files = {"file": (filename, data, media_type)}
    owns = client is None
    http = client or httpx.AsyncClient(timeout=DEFAULT_TIMEOUT)
    try:
        response = await http.post(url, data=form, files=files, headers=headers)
        response.raise_for_status()
        payload = _media_object(response)
    except httpx.HTTPStatusError as exc:
        raise map_http_error("openai", exc) from exc
    except httpx.HTTPError as exc:
        raise ProviderError("openai", str(exc), cause=exc) from exc
    finally:
        if owns:
            await http.aclose()
    if not isinstance(payload.get("text"), str):
        raise ProviderError("openai", "transcription response has no text field", retryable=False)
    return str(payload["text"])


async def openai_image(
    prompt: str,
    *,
    model: str = "dall-e-3",
    size: str = "1024x1024",
    n: int = 1,
    api_key: str | None = None,
    base_url: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """Call ``POST /images/generations``; returns ``{uri, b64, revised_prompt}``."""
    url = f"{_base_url(base_url)}/images/generations"
    headers = {"Authorization": f"Bearer {_api_key(api_key)}"}
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "n": n,
        "size": size,
        "response_format": "b64_json",
    }
    owns = client is None
    http = client or httpx.AsyncClient(timeout=DEFAULT_TIMEOUT)
    try:
        response = await http.post(url, json=payload, headers=headers)
        response.raise_for_status()
        body = _media_object(response)
    except httpx.HTTPStatusError as exc:
        raise map_http_error("openai", exc) from exc
    except httpx.HTTPError as exc:
        raise ProviderError("openai", str(exc), cause=exc) from exc
    finally:
        if owns:
            await http.aclose()
    rows = body.get("data") or []
    if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
        raise ProviderError("openai", "images/generations returned no image data", retryable=False)
    row = rows[0]
    b64 = row.get("b64_json")
    uri = row.get("url")
    if b64:
        _, media = _decode_image(b64)
        if not uri:
            uri = f"data:{media};base64,{b64}"
    elif not isinstance(uri, str) or not uri.startswith(("https://", "http://")):
        raise ProviderError(
            "openai", "image response contains neither bytes nor an HTTP URL", retryable=False
        )
    return {
        "uri": uri,
        "b64": b64,
        "revised_prompt": row.get("revised_prompt"),
        "model": f"openai:{model}",
        "raw": row,
    }


def image_content_from_result(result: dict[str, Any]) -> ImageContent:
    b64 = result.get("b64")
    if b64:
        data, media = _decode_image(b64)
        return ImageContent(
            data=data,
            media_type=media,
            uri=result.get("uri"),
            metadata={"model": result.get("model"), "provider": "openai"},
        )
    uri = result.get("uri")
    if uri:
        return ImageContent.from_uri(str(uri), model=result.get("model"), provider="openai")
    raise ConfigurationError(
        "image result has neither b64 nor uri",
        code="vision.image_empty",
    )


def _audio_parts(audio: str | Path | AudioContent | bytes) -> tuple[bytes, str, str]:
    if isinstance(audio, bytes):
        return audio, "audio.wav", "audio/wav"
    if isinstance(audio, AudioContent):
        if audio.data is not None:
            media = audio.media_type or "audio/wav"
            ext = media.split("/")[-1] if "/" in media else "wav"
            return audio.data, f"audio.{ext}", media
        if audio.uri and audio.uri.startswith("file:"):
            path = Path(audio.uri.removeprefix("file://").removeprefix("file:"))
            return path.read_bytes(), path.name, audio.media_type or "audio/wav"
        raise ConfigurationError(
            "AudioContent for transcription needs inline bytes or a file URI",
            code="audio.asr_bytes_required",
        )
    path = Path(audio)
    if not path.is_file():
        raise ConfigurationError(
            f"audio file not found: {path}",
            code="audio.file_not_found",
            context={"path": str(path)},
        )
    media = "audio/wav"
    if path.suffix.lower() in {".mp3", ".mpeg"}:
        media = "audio/mpeg"
    elif path.suffix.lower() == ".m4a":
        media = "audio/mp4"
    elif path.suffix.lower() == ".ogg":
        media = "audio/ogg"
    return path.read_bytes(), path.name, media

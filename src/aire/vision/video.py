"""Video summarization pipeline (frame sampling + model describe)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from aire.core.content import Message, TextContent, VideoContent
from aire.core.errors import ConfigurationError, NotFoundError
from aire.core.types import Capability
from aire.models.base import Model
from aire.models.types import GenerationRequest


class VideoSummary(BaseModel):
    summary: str
    frames_used: int = 0
    model: str = "unknown"
    metadata: dict[str, Any] = Field(default_factory=dict)


class VideoPipeline:
    """Summarize video via model vision/video capability or offline stub."""

    def __init__(self, model: Model | None = None) -> None:
        self.model = model

    async def summarize(
        self,
        video: str | Path | VideoContent,
        *,
        prompt: str = "Summarize this video.",
        max_frames: int = 4,
    ) -> VideoSummary:
        content = _to_video(video)
        if self.model is None:
            loc = content.uri or content.metadata.get("path") or "inline"
            return VideoSummary(
                summary=f"[offline stub] Video at {loc}; "
                f"no model configured. Prompt was: {prompt}",
                frames_used=0,
                model="stub",
                metadata={"prompt": prompt},
            )
        info = self.model.info
        if info.supports(Capability.VISION_INPUT) or "video" in {
            str(c).lower() for c in info.capabilities
        }:
            request = GenerationRequest(
                messages=[
                    Message(
                        role="user",
                        content=[TextContent(text=prompt), content],
                    )
                ]
            )
            result = await self.model.generate(request)
            return VideoSummary(
                summary=result.text,
                frames_used=max_frames,
                model=info.ref,
            )
        # Fallback: ask text-only model with a placeholder frame description.
        frames = _sample_frame_uris(content, max_frames=max_frames)
        frame_note = f"{len(frames)} sampled frame refs: " + ", ".join(frames[:3])
        text = await self.model.ask(f"{prompt}\n\n{frame_note}")
        return VideoSummary(
            summary=str(text),
            frames_used=len(frames),
            model=info.ref,
            metadata={"mode": "text_fallback"},
        )

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "video_pipeline",
            "model": self.model.info.ref if self.model else None,
        }


def _to_video(video: str | Path | VideoContent) -> VideoContent:
    if isinstance(video, VideoContent):
        return video
    value = str(video)
    if value.startswith(("http://", "https://")):
        return VideoContent.from_uri(value)
    path = Path(value)
    if not path.exists():
        raise ConfigurationError(
            f"video not found: {path}",
            code="vision.video_not_found",
        )
    return VideoContent.from_file(value, source=str(path))


def _sample_frame_uris(video: VideoContent, *, max_frames: int) -> list[str]:
    """Offline frame sampling stub (no ffmpeg): return synthetic frame labels."""
    base = video.uri or str(video.metadata.get("path") or "video")
    return [f"{base}#frame={i}" for i in range(max(1, max_frames))]


def require_vision_model(model: Model) -> None:
    if not model.info.supports(Capability.VISION_INPUT):
        raise NotFoundError(
            "capability",
            str(Capability.VISION_INPUT),
            context={"model": model.info.ref},
        )

"""Vision pipelines built on the universal content + model interfaces.

Classification, detection and VQA all delegate to models advertising
``vision-input`` capability; providers map image content to their wire format.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from aire.core.content import ImageContent, Message, TextContent
from aire.core.errors import NotFoundError
from aire.core.types import Capability
from aire.models.base import Model
from aire.models.types import GenerationRequest


class Detection(BaseModel):
    label: str
    confidence: float = 0.0
    box: tuple[float, float, float, float] | None = None


class VisionResult(BaseModel):
    text: str
    labels: list[str] = Field(default_factory=list)
    detections: list[Detection] = Field(default_factory=list)
    model: str = "unknown"


class VisionPipeline:
    """Image classification, description and VQA through one interface."""

    def __init__(self, model: Model) -> None:
        if not model.info.supports(Capability.VISION_INPUT):
            raise NotFoundError(
                "capability",
                str(Capability.VISION_INPUT),
                context={"model": model.info.ref, "hint": "use a vision-capable model"},
            )
        self.model = model

    async def _ask(self, image: ImageContent, prompt: str) -> str:
        request = GenerationRequest(
            messages=[Message(role="user", content=[TextContent(text=prompt), image])]
        )
        return (await self.model.generate(request)).text

    async def describe(
        self, image: str | Path | ImageContent, *, prompt: str = "Describe this image."
    ) -> VisionResult:
        content = _to_image(image)
        text = await self._ask(content, prompt)
        return VisionResult(text=text, model=self.model.info.ref)

    async def classify(self, image: str | Path | ImageContent, labels: list[str]) -> VisionResult:
        """Zero-shot classification against candidate labels."""
        content = _to_image(image)
        prompt = (
            "Classify the image into exactly one of these labels: "
            + ", ".join(labels)
            + ". Respond with only the label."
        )
        text = await self._ask(content, prompt)
        chosen = text.strip().strip(".")
        matched = [label for label in labels if label.lower() in chosen.lower()]
        return VisionResult(text=chosen, labels=matched[:1], model=self.model.info.ref)

    async def vqa(self, image: str | Path | ImageContent, question: str) -> VisionResult:
        """Visual question answering."""
        content = _to_image(image)
        text = await self._ask(content, question)
        return VisionResult(text=text, model=self.model.info.ref)

    def describe_pipeline(self) -> dict[str, Any]:
        return {"kind": "vision_pipeline", "model": self.model.info.ref}


def _to_image(image: str | Path | ImageContent) -> ImageContent:
    if isinstance(image, ImageContent):
        return image
    value = str(image)
    if value.startswith(("http://", "https://")):
        return ImageContent.from_uri(value)
    return ImageContent.from_file(value)

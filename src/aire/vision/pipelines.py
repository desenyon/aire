"""Vision pipelines built on the universal content + model interfaces.

Classification, detection, VQA and image generation all delegate to models
advertising the matching capability; providers map content to their wire format.
"""

from __future__ import annotations

import json
import re
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


class ImageGenerationResult(BaseModel):
    prompt: str
    uri: str | None = None
    b64: str | None = None
    text: str = ""
    model: str = "unknown"


class VisionPipeline:
    """Image classification, description, detection and VQA through one interface."""

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

    async def detect(
        self,
        image: str | Path | ImageContent,
        *,
        labels: list[str] | None = None,
    ) -> VisionResult:
        """Object detection via a vision-capable model (JSON detections).

        Models should return JSON like::

            {"detections": [{"label": "cat", "confidence": 0.9, "box": [0,0,1,1]}]}

        Offline echo models fall back to label-list heuristics from the prompt text.
        """
        content = _to_image(image)
        label_hint = (
            "Restrict labels to: " + ", ".join(labels) + ". "
            if labels
            else "Use free-form labels. "
        )
        prompt = (
            "Detect objects in the image. "
            + label_hint
            + 'Respond with JSON only: {"detections":[{"label":str,"confidence":0-1,'
            '"box":[x1,y1,x2,y2]|null}]}'
        )
        text = await self._ask(content, prompt)
        detections = _parse_detections(text, labels=labels)
        return VisionResult(
            text=text,
            labels=[d.label for d in detections],
            detections=detections,
            model=self.model.info.ref,
        )

    def describe_pipeline(self) -> dict[str, Any]:
        return {"kind": "vision_pipeline", "model": self.model.info.ref}


class ImageGenerationPipeline:
    """Text-to-image generation through models advertising ``image-generation``."""

    def __init__(self, model: Model) -> None:
        if not model.info.supports(Capability.IMAGE_GENERATION):
            raise NotFoundError(
                "capability",
                str(Capability.IMAGE_GENERATION),
                context={"model": model.info.ref, "hint": "use an image-generation model"},
            )
        self.model = model

    async def generate(self, prompt: str, *, size: str = "1024x1024") -> ImageGenerationResult:
        """Generate an image description / URI from a text prompt.

        Providers that return a data URI or https URL in the completion text are
        parsed into ``uri``; otherwise ``text`` holds the raw model response
        (useful for offline stubs).
        """
        request = GenerationRequest.of(
            f"Generate an image ({size}) for this prompt and return a URI or description:\n{prompt}"
        )
        result = await self.model.generate(request)
        text = result.text.strip()
        uri = _extract_uri(text)
        return ImageGenerationResult(
            prompt=prompt,
            uri=uri,
            text=text,
            model=self.model.info.ref,
        )

    def describe(self) -> dict[str, Any]:
        return {"kind": "image_generation_pipeline", "model": self.model.info.ref}


def _to_image(image: str | Path | ImageContent) -> ImageContent:
    if isinstance(image, ImageContent):
        return image
    value = str(image)
    if value.startswith(("http://", "https://")):
        return ImageContent.from_uri(value)
    return ImageContent.from_file(value)


def _extract_uri(text: str) -> str | None:
    match = re.search(r"(https?://\S+|data:image/[^\s\"']+)", text)
    return match.group(1) if match else None


def _parse_detections(text: str, *, labels: list[str] | None) -> list[Detection]:
    # Prefer JSON blob
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            payload = json.loads(text[start : end + 1])
            raw = payload.get("detections", [])
            out: list[Detection] = []
            for item in raw:
                if not isinstance(item, dict):
                    continue
                label = str(item.get("label", "")).strip()
                if not label:
                    continue
                box = item.get("box")
                box_t: tuple[float, float, float, float] | None = None
                if isinstance(box, (list, tuple)) and len(box) == 4:
                    box_t = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
                out.append(
                    Detection(
                        label=label,
                        confidence=float(item.get("confidence", 0.0) or 0.0),
                        box=box_t,
                    )
                )
            if out:
                return out
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    # Heuristic: any requested labels mentioned in the echo/prompt text
    if labels:
        lower = text.lower()
        found = [Detection(label=lab, confidence=0.5) for lab in labels if lab.lower() in lower]
        if found:
            return found
    return []

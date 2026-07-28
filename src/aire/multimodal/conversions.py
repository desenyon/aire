"""Multimodal conversion pipelines.

Converters transform one content kind into another (audio→text, image→text,
text→image, ...). Built-in converters delegate to models advertising the
matching capability, so the registry works with any provider without core
code knowing about vendors.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from aire.core.content import AudioContent, Content, ImageContent, Message, TextContent
from aire.core.errors import NotFoundError
from aire.core.types import Capability
from aire.models.base import Model
from aire.models.types import GenerationRequest


@runtime_checkable
class Converter(Protocol):
    """Transforms content from one kind to another."""

    source_kind: str
    target_kind: str

    async def convert(self, content: Content) -> Content: ...


class ModelConverter:
    """Delegates a conversion to a model with the required capability."""

    def __init__(self, model: Model, source_kind: str, target_kind: str, *, prompt: str) -> None:
        self.model = model
        self.source_kind = source_kind
        self.target_kind = target_kind
        self.prompt = prompt

    async def convert(self, content: Content) -> Content:
        request = GenerationRequest(
            messages=[Message(role="user", content=[TextContent(text=self.prompt), content])]
        )
        result = await self.model.generate(request)
        if self.target_kind == "text":
            return TextContent(text=result.text)
        # Non-text targets require providers returning binary content; the
        # provider adapter is responsible for placing it in result.content.
        for block in result.content:
            if block.kind == self.target_kind:
                return block
        raise NotFoundError(
            "conversion output",
            f"{self.source_kind}->{self.target_kind}",
            context={"model": self.model.info.ref},
        )


class ConversionRegistry:
    """Registry of converters keyed by ``"source->target"``."""

    def __init__(self) -> None:
        self._converters: dict[str, Converter] = {}

    def register(self, converter: Converter, *, replace: bool = True) -> Converter:
        key = f"{converter.source_kind}->{converter.target_kind}"
        if key in self._converters and not replace:
            from aire.core.errors import PluginError

            raise PluginError(f"converter {key!r} already registered", code="registry.duplicate")
        self._converters[key] = converter
        return converter

    def get(self, source_kind: str, target_kind: str) -> Converter:
        key = f"{source_kind}->{target_kind}"
        try:
            return self._converters[key]
        except KeyError:
            raise NotFoundError(
                "converter", key, context={"available": sorted(self._converters)}
            ) from None

    def names(self) -> list[str]:
        return sorted(self._converters)


async def transcribe(
    model: Model, audio: AudioContent, *, prompt: str = "Transcribe this audio."
) -> TextContent:
    """Audio → text via a model with SPEECH_RECOGNITION capability."""
    if not model.info.supports(Capability.SPEECH_RECOGNITION):
        raise NotFoundError(
            "capability", str(Capability.SPEECH_RECOGNITION), context={"model": model.info.ref}
        )
    converter = ModelConverter(model, "audio", "text", prompt=prompt)
    result = await converter.convert(audio)
    assert isinstance(result, TextContent)
    return result


async def describe_image(
    model: Model, image: ImageContent, *, prompt: str = "Describe this image in detail."
) -> TextContent:
    """Image → text via a model with VISION_INPUT capability."""
    if not model.info.supports(Capability.VISION_INPUT):
        raise NotFoundError(
            "capability", str(Capability.VISION_INPUT), context={"model": model.info.ref}
        )
    converter = ModelConverter(model, "image", "text", prompt=prompt)
    result = await converter.convert(image)
    assert isinstance(result, TextContent)
    return result

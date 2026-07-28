"""Normalized multimodal content objects.

These are the library-owned primitives exchanged between every subsystem:
messages, model requests, tool results, retrieval chunks and agent memory all
speak in these types so providers never leak vendor-specific payloads.
"""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field

from aire.core.errors import DataError


class ContentBase(BaseModel):
    """Shared base for all content blocks."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    def describe(self) -> dict[str, Any]:
        """Machine-readable summary for agents and tracing."""
        return {"kind": self.kind, "metadata": self.metadata}


class TextContent(ContentBase):
    """Plain text block."""

    kind: Literal["text"] = "text"
    text: str

    @classmethod
    def of(cls, text: str, **metadata: Any) -> TextContent:
        return cls(text=text, metadata=metadata)


class _BytesContent(ContentBase):
    """Base for binary content carried as bytes or referenced by URI/path."""

    kind: str = "bytes"
    data: bytes | None = None
    uri: str | None = None
    media_type: str | None = None

    @classmethod
    def from_file(cls, path: str | Path, **metadata: Any) -> Self:
        p = Path(path)
        if not p.is_file():
            raise DataError(f"file not found: {p}", context={"path": str(p)})
        media = mimetypes.guess_type(p.name)[0]
        return cls(data=p.read_bytes(), media_type=media, metadata=dict(metadata))

    @classmethod
    def from_uri(cls, uri: str, **metadata: Any) -> Self:
        return cls(uri=uri, metadata=dict(metadata))

    def as_base64(self) -> str:
        if self.data is None:
            raise DataError("content carries no inline bytes (uri-only reference)")
        return base64.b64encode(self.data).decode("ascii")


class ImageContent(_BytesContent):
    """Image block (inline bytes, local file, or remote URI)."""

    kind: Literal["image"] = "image"


class AudioContent(_BytesContent):
    """Audio block."""

    kind: Literal["audio"] = "audio"


class VideoContent(_BytesContent):
    """Video block."""

    kind: Literal["video"] = "video"


class DocumentContent(_BytesContent):
    """Document block (PDF, office formats, ...)."""

    kind: Literal["document"] = "document"
    page_count: int | None = None


class StructuredContent(ContentBase):
    """JSON-like structured data block (records, tool payloads, rows)."""

    kind: Literal["structured"] = "structured"
    data: Any = None
    schema_: dict[str, Any] | None = Field(default=None, alias="schema")


Content = Annotated[
    TextContent | ImageContent | AudioContent | VideoContent | DocumentContent | StructuredContent,
    Field(discriminator="kind"),
]


class Message(BaseModel):
    """A single conversational turn."""

    model_config = ConfigDict(frozen=True)

    role: Literal["system", "user", "assistant", "tool"]
    content: list[Content] = Field(default_factory=list)
    name: str | None = None
    tool_call_id: str | None = None

    @classmethod
    def text(cls, role: Literal["system", "user", "assistant", "tool"], text: str) -> Message:
        return cls(role=role, content=[TextContent(text=text)])

    @property
    def text_content(self) -> str:
        """Concatenated text of all text blocks (empty string if none)."""
        return "".join(c.text for c in self.content if isinstance(c, TextContent))

    def describe(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "kinds": [c.kind for c in self.content],
            "chars": len(self.text_content),
        }


def coerce_messages(value: str | Message | list[Message]) -> list[Message]:
    """Accept a bare string, a message or a list and normalize to a message list."""
    if isinstance(value, str):
        return [Message.text("user", value)]
    if isinstance(value, Message):
        return [value]
    return list(value)

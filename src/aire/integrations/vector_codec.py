"""Lossless chunk identity in vector backends with flat metadata payloads."""

from __future__ import annotations

import json
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from aire.rag.types import Chunk


def chunk_uuid(identifier: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"aire:chunk:{identifier}"))


def pack_metadata(chunk: Chunk) -> dict[str, Any]:
    return {
        **chunk.metadata,
        "__aire_document_id": chunk.document_id,
        "__aire_index": chunk.index,
        "__aire_metadata": json.dumps(chunk.metadata),
    }


def unpack_chunk(identifier: str, text: str, metadata: dict[str, Any]) -> Chunk:
    original = metadata.get("__aire_metadata")
    return Chunk(
        id=identifier,
        text=text,
        document_id=str(metadata.get("__aire_document_id", metadata.get("document_id", ""))),
        index=int(metadata.get("__aire_index", 0)),
        metadata=json.loads(original) if isinstance(original, str) else metadata,
    )

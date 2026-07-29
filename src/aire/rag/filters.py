"""Shared ACL / store-filter helpers for vector store adapters."""

from __future__ import annotations

from typing import Any

from aire.rag.acl import filter_hits, matches_acl
from aire.rag.types import ScoredChunk


def split_acl_filter(
    filter: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Return ``(store_filter_without_acl, acl_dict_or_None)``.

    Callers pass the store filter to the remote API and post-filter hits with
    :func:`apply_acl_to_hits` when ACL is present.
    """
    if not filter:
        return None, None
    acl = filter.get("__acl__") if isinstance(filter.get("__acl__"), dict) else None
    plain = {k: v for k, v in filter.items() if k != "__acl__"}
    return (plain or None), acl


def apply_acl_to_hits(
    hits: list[ScoredChunk],
    acl_filter: dict[str, Any] | None,
) -> list[ScoredChunk]:
    """Post-filter scored hits with :func:`~aire.rag.acl.matches_acl`."""
    return filter_hits(hits, acl_filter)


def matches_metadata(metadata: dict[str, Any], store_filter: dict[str, Any] | None) -> bool:
    """Exact-equality check for non-ACL store filter keys against chunk metadata."""
    if not store_filter:
        return True
    return all(metadata.get(key) == value for key, value in store_filter.items())


def apply_metadata_filter(
    hits: list[ScoredChunk],
    store_filter: dict[str, Any] | None,
) -> list[ScoredChunk]:
    """Client-side metadata equality filter (fallback when native filters are awkward)."""
    if not store_filter:
        return hits
    return [h for h in hits if matches_metadata(h.chunk.metadata, store_filter)]


__all__ = [
    "apply_acl_to_hits",
    "apply_metadata_filter",
    "matches_acl",
    "matches_metadata",
    "split_acl_filter",
]

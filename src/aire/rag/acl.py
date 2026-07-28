"""Access-control filters for retrieval (tenant / role / tag)."""

from __future__ import annotations

from typing import Any

from aire.rag.types import Chunk, ScoredChunk


def matches_acl(metadata: dict[str, Any], acl: dict[str, Any] | None) -> bool:  # noqa: C901
    """Return True if chunk metadata is allowed by ``acl``.

    Supported keys (all optional, AND-combined):
    - ``tenant`` / ``tenant_id``: exact match
    - ``roles``: chunk must list at least one overlapping role
    - ``tags_any``: chunk tags intersect
    - ``tags_all``: chunk tags superset
    - ``allow_public``: if True, chunks with ``public=True`` always pass
    """
    if not acl:
        return True
    if acl.get("allow_public") and metadata.get("public") is True:
        return True
    tenant = acl.get("tenant", acl.get("tenant_id"))
    if tenant is not None:
        chunk_tenant = metadata.get("tenant", metadata.get("tenant_id"))
        if chunk_tenant != tenant:
            return False
    roles = acl.get("roles")
    if roles:
        chunk_roles = set(metadata.get("roles") or metadata.get("allowed_roles") or [])
        if not chunk_roles.intersection(set(roles)):
            return False
    tags_any = acl.get("tags_any")
    if tags_any:
        chunk_tags = set(metadata.get("tags") or [])
        if not chunk_tags.intersection(set(tags_any)):
            return False
    tags_all = acl.get("tags_all")
    if tags_all:
        chunk_tags = set(metadata.get("tags") or [])
        if not set(tags_all).issubset(chunk_tags):
            return False
    # pass through any other exact metadata equals
    reserved = {
        "tenant",
        "tenant_id",
        "roles",
        "tags_any",
        "tags_all",
        "allow_public",
    }
    for key, value in acl.items():
        if key in reserved:
            continue
        if metadata.get(key) != value:
            return False
    return True


def filter_chunks(chunks: list[Chunk], acl: dict[str, Any] | None) -> list[Chunk]:
    return [c for c in chunks if matches_acl(c.metadata, acl)]


def filter_hits(hits: list[ScoredChunk], acl: dict[str, Any] | None) -> list[ScoredChunk]:
    return [h for h in hits if matches_acl(h.chunk.metadata, acl)]


def merge_filter(
    base: dict[str, Any] | None, acl: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Combine a store filter with ACL into one filter dict for adapters."""
    if not base and not acl:
        return None
    out = dict(base or {})
    if acl:
        out["__acl__"] = acl
    return out

"""ACL filter splitting."""

from __future__ import annotations

from aire.rag.filters import split_acl_filter


def test_split_acl_filter_extracts_acl() -> None:
    store, acl = split_acl_filter({"tenant": "acme", "__acl__": {"roles": ["reader"]}})
    assert store == {"tenant": "acme"}
    assert acl == {"roles": ["reader"]}


def test_split_acl_filter_none() -> None:
    assert split_acl_filter(None) == (None, None)
    assert split_acl_filter({}) == (None, None)


def test_split_acl_filter_acl_only() -> None:
    store, acl = split_acl_filter({"__acl__": {"roles": ["admin"]}})
    assert store is None
    assert acl == {"roles": ["admin"]}

"""Semantic chunker honesty via describe()."""

from __future__ import annotations

import warnings

from aire.data.chunking import get_chunker


def test_semantic_chunker_without_embedder_is_honest() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        chunker = get_chunker("semantic")
    assert any(issubclass(w.category, UserWarning) for w in caught)
    desc = chunker.describe()
    assert desc["kind"] == "chunker"
    assert desc["name"] == "semantic"
    assert desc["mode"] == "sentence_approximation"
    note = desc.get("note", "").lower()
    assert "no embedder" in note or "sentence" in note

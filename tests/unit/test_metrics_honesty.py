"""Metric honesty: lexical semantic_overlap; embedding_similarity needs embedder."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from aire.core.errors import ConfigurationError
from aire.evaluation.metrics import embedding_similarity, semantic_overlap
from aire.evaluation.types import EvalCase
from aire.models.base import run_sync


def _ctx(**kwargs: object) -> SimpleNamespace:
    base = dict(
        latency_ms=0.0,
        input_tokens=0,
        output_tokens=0,
        cost_usd=0.0,
        context=None,
        judge=None,
        embedder=None,
    )
    base.update(kwargs)
    return SimpleNamespace(**base)


def test_semantic_overlap_still_callable() -> None:
    case = EvalCase(input="q", expected="hello world")
    result = run_sync(semantic_overlap(case, "hello there world", _ctx()))
    assert result.name == "semantic_overlap"
    assert 0.0 < result.score <= 1.0
    assert result.detail.get("kind") == "lexical_token_f1"


def test_embedding_similarity_requires_embedder() -> None:
    case = EvalCase(input="q", expected="hello")
    with pytest.raises(ConfigurationError) as exc:
        run_sync(embedding_similarity(case, "hello", _ctx(embedder=None)))
    assert "embedder" in str(exc.value).lower()

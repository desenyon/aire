"""Structured error behavior."""

from __future__ import annotations

import pytest

from aire.core.errors import (
    AireError,
    BudgetExceededError,
    NotFoundError,
    PermissionDeniedError,
    ProviderError,
    RateLimitError,
    ensure_aire_error,
)


def test_error_serialization_roundtrip() -> None:
    err = ProviderError("openai", "boom", status=500, context={"model": "gpt"})
    payload = err.to_dict()
    assert payload["code"] == "provider.error"
    assert payload["retryable"] is True
    assert payload["context"]["provider"] == "openai"
    assert payload["context"]["status"] == 500


def test_retryable_classification() -> None:
    assert RateLimitError("p", "slow down").retryable is True
    assert ProviderError("p", "bad request", status=400).retryable is False
    assert ProviderError("p", "overloaded", status=503).retryable is True
    assert BudgetExceededError("limit").retryable is False


def test_not_found_carries_context() -> None:
    err = NotFoundError("tool", "search")
    assert err.context["kind"] == "tool"
    assert err.context["identifier"] == "search"
    assert "search" in str(err)


def test_permission_denied() -> None:
    err = PermissionDeniedError("delete_db", "database.admin")
    assert err.code == "safety.permission_denied"
    assert err.context["permission"] == "database.admin"


def test_wrap_preserves_cause_and_code() -> None:
    original = ValueError("nope")
    wrapped = ensure_aire_error(original)
    assert isinstance(wrapped, AireError)
    assert wrapped.__cause__ is original
    assert ensure_aire_error(wrapped) is wrapped


def test_str_includes_code() -> None:
    assert str(AireError("msg", code="x.y")).startswith("[x.y]")
    with pytest.raises(AireError):
        raise AireError("boom")

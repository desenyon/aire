"""Structured error hierarchy for aire.

Every error raised by the library is an :class:`AireError` (or subclass) with a
stable machine-readable ``code``, human-readable ``message``, arbitrary
structured ``context`` and a ``retryable`` flag so that agents and callers can
react programmatically instead of parsing strings.
"""

from __future__ import annotations

from typing import Any


class AireError(Exception):
    """Base class for all aire errors.

    Attributes:
        code: Stable, machine readable error code (e.g. ``"model.not_found"``).
        message: Human readable description.
        context: Structured details safe to log or serialize.
        retryable: Whether retrying the same operation may succeed.
    """

    code: str = "aire.error"
    retryable: bool = False

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        context: dict[str, Any] | None = None,
        retryable: bool | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        if retryable is not None:
            self.retryable = retryable
        self.context: dict[str, Any] = dict(context or {})
        self.__cause__ = cause

    def to_dict(self) -> dict[str, Any]:
        """Serialize the error to a JSON-safe dictionary."""
        return {
            "error": type(self).__name__,
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
            "context": self.context,
        }

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"


# --- Configuration & resolution -------------------------------------------------


class ConfigurationError(AireError):
    """Invalid, missing or conflicting configuration."""

    code = "config.invalid"


class NotFoundError(AireError):
    """A registry, model, tool or resource could not be resolved."""

    code = "aire.not_found"

    def __init__(self, kind: str, identifier: str, **kw: Any) -> None:
        ctx = dict(kw.pop("context", {}) or {})
        ctx.setdefault("kind", kind)
        ctx.setdefault("identifier", identifier)
        super().__init__(f"{kind} not found: {identifier!r}", context=ctx, **kw)


class PluginError(AireError):
    """Plugin discovery, loading or contract violation."""

    code = "plugin.error"


# --- Providers & models ---------------------------------------------------------


class ProviderError(AireError):
    """An external provider failed or returned an unusable response."""

    code = "provider.error"
    retryable = True

    def __init__(
        self,
        provider: str,
        message: str,
        *,
        status: int | None = None,
        **kw: Any,
    ) -> None:
        ctx = dict(kw.pop("context", {}) or {})
        ctx.setdefault("provider", provider)
        if status is not None:
            ctx.setdefault("status", status)
            kw.setdefault("retryable", status in {408, 409, 425, 429, 500, 502, 503, 504})
        super().__init__(message, context=ctx, **kw)
        self.provider = provider
        self.status = status


class RateLimitError(ProviderError):
    """Provider throttled the request; retry after backoff."""

    code = "provider.rate_limited"
    retryable = True


class AuthenticationError(ProviderError):
    """Missing or invalid provider credentials."""

    code = "provider.auth"
    retryable = False


class ContextLengthError(AireError):
    """Input exceeds the model's context capacity."""

    code = "model.context_length"


class OutputValidationError(AireError):
    """Model output failed structured validation."""

    code = "model.output_invalid"


# --- Tools, agents, workflows -----------------------------------------------------


class ToolError(AireError):
    """Tool execution failed."""

    code = "tool.error"


class PermissionDeniedError(AireError):
    """An action lacked the required permission or approval."""

    code = "safety.permission_denied"
    retryable = False

    def __init__(self, action: str, permission: str, **kw: Any) -> None:
        super().__init__(
            f"action {action!r} requires permission {permission!r}",
            context={"action": action, "permission": permission},
            **kw,
        )


class BudgetExceededError(AireError):
    """Token, cost, step or time budget was exhausted."""

    code = "agent.budget_exceeded"
    retryable = False


class WorkflowError(AireError):
    """Workflow definition or execution failure."""

    code = "workflow.error"


# --- Data & retrieval -------------------------------------------------------------


class DataError(AireError):
    """Dataset loading, validation or processing failure."""

    code = "data.error"


class RetrievalError(AireError):
    """Indexing or retrieval failure."""

    code = "rag.error"


# --- Safety ------------------------------------------------------------------------


class SafetyError(AireError):
    """A guardrail blocked the operation."""

    code = "safety.blocked"
    retryable = False


class TimeoutError(AireError):
    """An operation exceeded its deadline."""

    code = "aire.timeout"
    retryable = True


def ensure_aire_error(exc: BaseException, *, code: str = "aire.internal") -> AireError:
    """Wrap arbitrary exceptions in an :class:`AireError` without hiding the cause."""
    if isinstance(exc, AireError):
        return exc
    return AireError(str(exc) or type(exc).__name__, code=code, cause=exc)

"""The Tool: a universal, self-describing callable.

Wrap any Python function — sync or async — with a schema-derived contract,
permissions, timeout, retry policy, side-effect classification and auditing.
"""

from __future__ import annotations

import asyncio
import builtins
import inspect
import time
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError, create_model
from pydantic.fields import FieldInfo

from aire.core.context import ExecutionContext
from aire.core.errors import AireError, PermissionDeniedError, TimeoutError, ToolError
from aire.core.types import Manifest
from aire.models.types import ToolDefinition
from aire.tools.types import RetryPolicy, SideEffect, ToolResult, ToolSpec


def _schema_from_signature(fn: Callable[..., Any]) -> tuple[type[BaseModel], dict[str, Any]]:
    """Build a pydantic model + JSON schema from a function signature."""
    fields: dict[str, tuple[type, FieldInfo]] = {}
    hints = (
        inspect.get_annotations(fn, eval_str=True) if hasattr(inspect, "get_annotations") else {}
    )
    signature = inspect.signature(fn)
    for name, param in signature.parameters.items():
        if name in {"self", "cls"}:
            continue
        annotation = hints.get(name, param.annotation)
        if annotation is inspect.Parameter.empty:
            annotation = str
        if param.default is inspect.Parameter.empty:
            fields[name] = (annotation, FieldInfo())
        else:
            default_field = FieldInfo(default=param.default)
            fields[name] = (annotation, default_field)
    model = create_model(  # extra=forbid: reject unexpected arguments (injection surface)
        f"{fn.__name__}_args",
        __config__=ConfigDict(extra="forbid"),
        **fields,  # type: ignore[call-overload]
    )
    return model, model.model_json_schema()


class Tool:
    """A function wrapped with its full executable contract."""

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        permissions: list[str] | None = None,
        timeout_seconds: float = 30.0,
        retry: RetryPolicy | None = None,
        side_effect: SideEffect = SideEffect.READ_ONLY,
        cost_estimate_usd: float | None = None,
        audit: bool = True,
    ) -> None:
        self.fn = fn
        args_model, input_schema = _schema_from_signature(fn)
        self._args_model = args_model
        self.spec = ToolSpec(
            name=name or fn.__name__,
            description=description or (inspect.getdoc(fn) or "").strip(),
            input_schema=input_schema,
            permissions=permissions or [],
            timeout_seconds=timeout_seconds,
            retry=retry or RetryPolicy(),
            side_effect=side_effect,
            cost_estimate_usd=cost_estimate_usd,
            audit=audit,
        )

    # -- introspection -----------------------------------------------------------

    @property
    def name(self) -> str:
        return self.spec.name

    def definition(self) -> ToolDefinition:
        """The model-facing view of this tool."""
        return ToolDefinition(
            name=self.spec.name,
            description=self.spec.description,
            parameters=self.spec.input_schema,
        )

    def describe(self) -> Manifest:
        return Manifest(
            kind="tool",
            name=self.spec.name,
            capabilities=[str(self.spec.side_effect)],
            input_schema=self.spec.input_schema,
            extra={
                "permissions": self.spec.permissions,
                "timeout_seconds": self.spec.timeout_seconds,
            },
        )

    # -- execution -----------------------------------------------------------------

    def check_permissions(self, context: ExecutionContext | None = None) -> None:
        """Raise PermissionDeniedError if the context lacks required permissions."""
        if not self.spec.permissions:
            return
        ctx = context or ExecutionContext.current()
        missing = [p for p in self.spec.permissions if p not in ctx.permissions]
        if missing:
            raise PermissionDeniedError(self.spec.name, ", ".join(missing))

    async def execute(
        self,
        arguments: dict[str, Any] | None = None,
        *,
        context: ExecutionContext | None = None,
    ) -> ToolResult:
        """Validate arguments, enforce timeout/retries, invoke, normalize result."""
        started = time.perf_counter()
        try:
            self.check_permissions(context)
        except PermissionDeniedError as exc:
            return ToolResult.failure(exc.message, tool=self.name, code=exc.code)
        try:
            validated = self._args_model.model_validate(arguments or {})
        except ValidationError as exc:
            return ToolResult.failure(
                f"invalid arguments: {exc}", tool=self.name, code="tool.arguments_invalid"
            )
        kwargs = validated.model_dump()
        attempts = max(1, self.spec.retry.attempts)
        last_error: BaseException | None = None
        for attempt in range(attempts):
            try:
                output = await asyncio.wait_for(
                    self._invoke(kwargs), timeout=self.spec.timeout_seconds
                )
                result = ToolResult.success(
                    output, tool=self.name, duration_ms=(time.perf_counter() - started) * 1000.0
                )
                return result
            except builtins.TimeoutError:
                last_error = TimeoutError(
                    f"tool {self.name!r} exceeded timeout of {self.spec.timeout_seconds}s",
                    context={"tool": self.name, "timeout": self.spec.timeout_seconds},
                )
                break  # timeouts are not retried by default
            except AireError as exc:
                last_error = exc
                if not exc.retryable:
                    break
            except Exception as exc:
                last_error = ToolError(
                    f"{type(exc).__name__}: {exc}", context={"tool": self.name}, cause=exc
                )
            if attempt < attempts - 1 and self.spec.retry.backoff_seconds:
                await asyncio.sleep(self.spec.retry.backoff_seconds)
        assert last_error is not None
        code = last_error.code if isinstance(last_error, AireError) else "tool.error"
        return ToolResult.failure(
            str(last_error),
            tool=self.name,
            code=code,
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )

    async def _invoke(self, kwargs: dict[str, Any]) -> Any:
        result = self.fn(**kwargs)
        if inspect.isawaitable(result):
            return await result
        return result


def tool(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    permissions: list[str] | None = None,
    timeout_seconds: float = 30.0,
    retry: RetryPolicy | None = None,
    side_effect: SideEffect | str = SideEffect.READ_ONLY,
    cost_estimate_usd: float | None = None,
    audit: bool = True,
) -> Tool | Callable[[Callable[..., Any]], Tool]:
    """Decorator turning a function into a self-describing :class:`Tool`.

    Usage::

        @tool(permissions=["database.read"], side_effect="read_only")
        async def search_orders(customer_id: str) -> list[dict]:
            '''Find orders for a customer.'''
            ...
    """
    level = SideEffect(side_effect)

    def _wrap(f: Callable[..., Any]) -> Tool:
        return Tool(
            f,
            name=name,
            description=description,
            permissions=permissions,
            timeout_seconds=timeout_seconds,
            retry=retry,
            side_effect=level,
            cost_estimate_usd=cost_estimate_usd,
            audit=audit,
        )

    if fn is not None:
        return _wrap(fn)
    return _wrap

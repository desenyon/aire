"""Ready-made approval policies for agent tool execution.

An :data:`Approver` is ``(ToolCall, ToolSpec) -> bool | Awaitable[bool]``.
These policies cover the common cases: rule-based auto-decisions by side
effect, and interactive human-in-the-loop prompts for everything else.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from aire.models.types import ToolCall
from aire.tools.types import SideEffect, ToolSpec

# Severity order used by RuleApprover's ``auto_approve_below``.
_SEVERITY = [
    SideEffect.READ_ONLY,
    SideEffect.REVERSIBLE_WRITE,
    SideEffect.EXTERNAL_SIDE_EFFECT,
]


class RuleApprover:
    """Auto-approve below a side-effect severity, auto-deny at/above it.

    ``auto_approve_below=SideEffect.REVERSIBLE_WRITE`` approves read-only
    tools and denies everything that writes. Per-tool overrides always win.
    """

    def __init__(
        self,
        *,
        auto_approve_below: SideEffect | str = SideEffect.REVERSIBLE_WRITE,
        allow: set[str] | None = None,
        deny: set[str] | None = None,
        default: bool = False,
    ) -> None:
        self.threshold = SideEffect(auto_approve_below)
        self.allow = set(allow or ())
        self.deny = set(deny or ())
        self.default = default
        self.decisions: list[dict[str, Any]] = []  # audit trail

    def __call__(self, call: ToolCall, spec: ToolSpec) -> bool:
        if call.name in self.allow:
            approved = True
        elif call.name in self.deny:
            approved = False
        else:
            severity = _SEVERITY.index(SideEffect(spec.side_effect))
            approved = severity < _SEVERITY.index(self.threshold)
        self.decisions.append(
            {"tool": call.name, "side_effect": str(spec.side_effect), "approved": approved}
        )
        return approved


class InteractiveApprover:
    """Prompts a human on stdin for each tool call (y/n/always/never).

    ``always``/``never`` are remembered per tool for the session. Runs the
    blocking prompt in a worker thread so the event loop never stalls.
    """

    def __init__(self, *, max_arg_chars: int = 400) -> None:
        self.max_arg_chars = max_arg_chars
        self._remembered: dict[str, bool] = {}

    async def __call__(self, call: ToolCall, spec: ToolSpec) -> bool:
        if call.name in self._remembered:
            return self._remembered[call.name]
        return await asyncio.to_thread(self._prompt, call, spec)

    def _prompt(self, call: ToolCall, spec: ToolSpec) -> bool:
        arguments = json.dumps(call.arguments, default=str)
        if len(arguments) > self.max_arg_chars:
            arguments = arguments[: self.max_arg_chars] + "…"
        print(f"\n[aire] approval requested for tool {call.name!r}")
        print(f"  description: {spec.description or '(none)'}")
        print(f"  side_effect: {spec.side_effect}")
        print(f"  arguments:   {arguments}")
        while True:
            answer = input("  allow? [y]es / [n]o / [a]lways / n[e]ver: ").strip().lower()
            if answer in {"y", "yes"}:
                return True
            if answer in {"n", "no"}:
                return False
            if answer in {"a", "always"}:
                self._remembered[call.name] = True
                return True
            if answer in {"e", "never"}:
                self._remembered[call.name] = False
                return False

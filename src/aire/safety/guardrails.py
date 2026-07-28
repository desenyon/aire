"""Guardrails: first-class, composable safety checks on inputs and outputs."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from aire.core.errors import SafetyError
from aire.safety.patterns import Match, detect_injection, detect_pii, detect_secrets


class GuardrailVerdict(BaseModel):
    """Outcome of one guardrail check."""

    passed: bool
    guardrail: str
    matches: list[dict[str, Any]] = Field(default_factory=list)
    action: str = "block"  # block | warn | redact


@runtime_checkable
class Guardrail(Protocol):
    """Checks text and returns a verdict. Never raises for ordinary findings."""

    name: str

    def check(self, text: str) -> GuardrailVerdict: ...


class PIIGuardrail:
    name = "pii"

    def __init__(self, *, action: str = "warn") -> None:
        self.action = action

    def check(self, text: str) -> GuardrailVerdict:
        matches = detect_pii(text)
        return GuardrailVerdict(
            passed=not matches,
            guardrail=self.name,
            action=self.action,
            matches=[_match_dict(m) for m in matches],
        )


class InjectionGuardrail:
    name = "prompt_injection"

    def __init__(self, *, action: str = "block") -> None:
        self.action = action

    def check(self, text: str) -> GuardrailVerdict:
        matches = detect_injection(text)
        return GuardrailVerdict(
            passed=not matches,
            guardrail=self.name,
            action=self.action,
            matches=[_match_dict(m) for m in matches],
        )


class SecretGuardrail:
    name = "secret"

    def __init__(self, *, action: str = "block") -> None:
        self.action = action

    def check(self, text: str) -> GuardrailVerdict:
        matches = detect_secrets(text)
        return GuardrailVerdict(
            passed=not matches,
            guardrail=self.name,
            action=self.action,
            matches=[_match_dict(m) for m in matches],
        )


class GuardrailChain:
    """Runs guardrails in order; enforces their configured actions."""

    def __init__(self, guardrails: list[Guardrail] | None = None) -> None:
        self.guardrails: list[Guardrail] = guardrails or [
            InjectionGuardrail(),
            SecretGuardrail(),
            PIIGuardrail(),
        ]

    def check(self, text: str, *, stage: str = "input") -> list[GuardrailVerdict]:
        """Check text; raise SafetyError when a blocking guardrail fails."""
        verdicts = [g.check(text) for g in self.guardrails]
        for verdict in verdicts:
            if not verdict.passed and verdict.action == "block":
                raise SafetyError(
                    f"{stage} blocked by guardrail {verdict.guardrail!r}",
                    code="safety.guardrail_blocked",
                    context={
                        "stage": stage,
                        "guardrail": verdict.guardrail,
                        "matches": len(verdict.matches),
                    },
                )
        return verdicts

    def describe(self) -> dict[str, Any]:
        return {"kind": "guardrail_chain", "guardrails": [g.name for g in self.guardrails]}


def _match_dict(m: Match) -> dict[str, Any]:
    return {"kind": m.kind, "span": list(m.span), "preview": m.text[:40]}

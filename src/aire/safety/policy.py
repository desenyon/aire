"""Policy: risk classification and approval requirements for actions.

Deepened in 0.3 with a rule engine over tools and models.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from aire.core.errors import PermissionDeniedError
from aire.tools.types import SideEffect, at_least

RuleAction = Literal["allow", "deny", "require_approval"]


class PolicyRule(BaseModel):
    """One matching rule in the policy engine."""

    name: str
    action: RuleAction = "allow"
    # Matchers (all provided matchers must hold; empty = match all)
    tool: str | None = None
    tool_prefix: str | None = None
    model: str | None = None
    model_prefix: str | None = None
    side_effect_at_or_above: SideEffect | None = None
    permissions_any: list[str] = Field(default_factory=list)
    reason: str = ""


class ApprovalPolicy(BaseModel):
    """Decides which side-effect levels require human approval.

    ``prohibited`` actions are never executable, regardless of approval.
    """

    require_approval_at_or_above: SideEffect = SideEffect.EXTERNAL_SIDE_EFFECT
    trusted_permissions: list[str] = Field(default_factory=list)

    def requires_approval(
        self, side_effect: SideEffect, permissions: list[str] | None = None
    ) -> bool:
        if side_effect == SideEffect.PROHIBITED:
            return True
        granted = set(permissions or [])
        if granted and any(p in self.trusted_permissions for p in granted):
            return False
        return at_least(side_effect, self.require_approval_at_or_above)

    def is_prohibited(self, side_effect: SideEffect) -> bool:
        return side_effect == SideEffect.PROHIBITED

    def describe(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class PolicyEngine:
    """Ordered rule engine over tools/models; first match wins.

    Falls back to :class:`ApprovalPolicy` side-effect thresholds when no rule
    matches.
    """

    def __init__(
        self,
        rules: list[PolicyRule] | None = None,
        *,
        fallback: ApprovalPolicy | None = None,
        default_action: RuleAction = "allow",
    ) -> None:
        self.rules = list(rules or [])
        self.fallback = fallback or ApprovalPolicy()
        self.default_action = default_action
        self.decisions: list[dict[str, Any]] = []

    def add(self, rule: PolicyRule) -> None:
        self.rules.append(rule)

    def decide(
        self,
        *,
        tool: str | None = None,
        model: str | None = None,
        side_effect: SideEffect | str | None = None,
        permissions: list[str] | None = None,
    ) -> RuleAction:
        se = SideEffect(side_effect) if side_effect is not None else None
        for rule in self.rules:
            if not self._matches(
                rule, tool=tool, model=model, side_effect=se, permissions=permissions
            ):
                continue
            self.decisions.append(
                {
                    "rule": rule.name,
                    "action": rule.action,
                    "tool": tool,
                    "model": model,
                    "reason": rule.reason,
                }
            )
            return rule.action
        if se is not None and self.fallback.is_prohibited(se):
            action: RuleAction = "deny"
        elif se is not None and self.fallback.requires_approval(se, permissions):
            action = "require_approval"
        else:
            action = self.default_action
        self.decisions.append(
            {"rule": "_fallback", "action": action, "tool": tool, "model": model}
        )
        return action

    def assert_allowed(self, **kwargs: Any) -> None:
        action = self.decide(**kwargs)
        if action == "deny":
            tool = kwargs.get("tool") or kwargs.get("model") or "action"
            raise PermissionDeniedError(str(tool), "policy.allow")

    def requires_approval(self, **kwargs: Any) -> bool:
        return self.decide(**kwargs) == "require_approval"

    @staticmethod
    def _matches(
        rule: PolicyRule,
        *,
        tool: str | None,
        model: str | None,
        side_effect: SideEffect | None,
        permissions: list[str] | None,
    ) -> bool:
        if rule.tool is not None and tool != rule.tool:
            return False
        if rule.tool_prefix is not None and not (tool or "").startswith(rule.tool_prefix):
            return False
        if rule.model is not None and model != rule.model:
            return False
        if rule.model_prefix is not None and not (model or "").startswith(rule.model_prefix):
            return False
        if rule.side_effect_at_or_above is not None and (
            side_effect is None or not at_least(side_effect, rule.side_effect_at_or_above)
        ):
            return False
        if rule.permissions_any:
            granted = set(permissions or [])
            if not any(p in granted for p in rule.permissions_any):
                return False
        return True

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "policy_engine",
            "rules": [r.model_dump(mode="json") for r in self.rules],
            "fallback": self.fallback.describe(),
            "default_action": self.default_action,
            "decisions": len(self.decisions),
        }


def default_engine() -> PolicyEngine:
    """Sensible defaults: deny prohibited, approve external+high-impact tools."""
    return PolicyEngine(
        [
            PolicyRule(
                name="deny_prohibited",
                action="deny",
                side_effect_at_or_above=SideEffect.PROHIBITED,
                reason="prohibited side effects are never allowed",
            ),
            PolicyRule(
                name="approve_external",
                action="require_approval",
                side_effect_at_or_above=SideEffect.EXTERNAL_SIDE_EFFECT,
                reason="external side effects need approval",
            ),
        ]
    )

"""Policy: risk classification and approval requirements for actions."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from aire.tools.types import SideEffect, at_least


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

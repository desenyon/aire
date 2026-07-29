"""PolicyEngine defaults."""

from __future__ import annotations

from aire.safety.policy import default_engine
from aire.tools.types import SideEffect


def test_deny_prohibited() -> None:
    engine = default_engine()
    assert engine.decide(side_effect=SideEffect.PROHIBITED) == "deny"


def test_require_approval_external() -> None:
    engine = default_engine()
    assert engine.decide(side_effect=SideEffect.EXTERNAL_SIDE_EFFECT) == "require_approval"
    assert engine.requires_approval(side_effect=SideEffect.EXTERNAL_SIDE_EFFECT)


def test_allow_read_only() -> None:
    engine = default_engine()
    assert engine.decide(side_effect=SideEffect.READ_ONLY) == "allow"

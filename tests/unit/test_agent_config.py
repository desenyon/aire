"""AgentConfig defaults."""

from __future__ import annotations

from aire.agents.types import AgentConfig


def test_approval_levels_default() -> None:
    cfg = AgentConfig()
    assert cfg.approval_levels == [
        "external_side_effect",
        "high_impact",
        "prohibited",
    ]


def test_planning_flag_default_false() -> None:
    assert AgentConfig().planning is False
    assert AgentConfig(planning=True).planning is True


def test_max_steps_default() -> None:
    assert AgentConfig().max_steps == 12

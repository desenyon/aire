"""PlanActVerify JSON helpers."""

from __future__ import annotations

from aire.agents.plan import _parse_steps, _parse_verify


def test_parse_steps_from_json() -> None:
    text = (
        'Plan: {"steps":[{"id":"1","action":"add","tool":"calculator",'
        '"args":{"expression":"1+1"}}]}'
    )
    steps = _parse_steps(text)
    assert len(steps) == 1
    assert steps[0].id == "1"
    assert steps[0].action == "add"
    assert steps[0].tool == "calculator"
    assert steps[0].args["expression"] == "1+1"


def test_parse_steps_invalid() -> None:
    assert _parse_steps("no json here") == []


def test_parse_verify_ok() -> None:
    parsed = _parse_verify('{"ok": true, "notes": "done"}')
    assert parsed is not None
    assert parsed["ok"] is True
    assert parsed["notes"] == "done"


def test_parse_verify_false() -> None:
    parsed = _parse_verify('Result: {"ok": false, "notes": "missing"}')
    assert parsed is not None
    assert parsed["ok"] is False

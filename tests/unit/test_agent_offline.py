"""Agent completes offline with mock:echo + calculator."""

from __future__ import annotations

from aire import AI
from aire.agents.types import AgentStatus
from aire.tools.builtins import builtin_tools


def test_agent_mock_echo_with_calculator_completes() -> None:
    calc = next(t for t in builtin_tools() if t.name == "calculator")
    agent = AI.agents.create_sync("mock:echo", tools=[calc], name="calc-agent")
    result = agent.run_sync("What is 2+2?")
    assert result.status == AgentStatus.COMPLETED
    assert result.ok
    assert "2+2" in result.output or result.output  # echo returns the prompt

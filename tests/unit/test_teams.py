"""Multi-agent tests: agent-as-tool composition and supervisor-routed teams."""

from __future__ import annotations

import pytest

from aire.agents import Agent, Team
from aire.core.runtime import Runtime
from aire.models.builtin import CallableModel


def _echo_agent(runtime: Runtime, name: str) -> Agent:
    import aire.models.builtin as builtin

    model = builtin.EchoModel()
    return Agent(model, name=name, runtime=runtime)


def _scripted_supervisor(lines: list[str]) -> CallableModel:
    queue = list(lines)

    def scripted(prompt: str) -> str:
        assert queue, "supervisor called more times than scripted"
        return queue.pop(0)

    return CallableModel("scripted-supervisor", scripted)


@pytest.mark.anyio
async def test_agent_as_tool(runtime: Runtime) -> None:
    agent = _echo_agent(runtime, "researcher")
    tool = agent.as_tool(description="Delegate research tasks.")
    assert tool.name == "researcher"
    assert tool.spec.description == "Delegate research tasks."
    assert "task" in tool.spec.input_schema["properties"]

    result = await tool.execute({"task": "find the capital of France"})
    assert result.ok, result.error
    assert isinstance(result.output, str) and result.output


@pytest.mark.anyio
async def test_team_delegates_then_finishes(runtime: Runtime) -> None:
    supervisor = _scripted_supervisor(
        [
            '{"agent": "researcher", "subtask": "gather facts", "done": false}',
            '{"agent": "writer", "subtask": "write it up", "done": false}',
            '{"done": true, "final_answer": "Synthesized team answer"}',
        ]
    )
    members = {
        "researcher": _echo_agent(runtime, "researcher"),
        "writer": _echo_agent(runtime, "writer"),
    }
    team = Team(members, supervisor)
    result = await team.run("produce a report")
    assert result.answer == "Synthesized team answer"
    assert [d.agent for d in result.delegations] == ["researcher", "writer"]
    assert result.rounds == 3
    assert result.delegations[0].output  # member output was captured as observation


@pytest.mark.anyio
async def test_team_handles_unknown_member(runtime: Runtime) -> None:
    supervisor = _scripted_supervisor(
        [
            '{"agent": "ghost", "subtask": "haunt", "done": false}',
            '{"done": true, "final_answer": "recovered"}',
        ]
    )
    team = Team({"real": _echo_agent(runtime, "real")}, supervisor)
    result = await team.run("task")
    assert result.answer == "recovered"
    assert result.delegations == []  # unknown member was not recorded


@pytest.mark.anyio
async def test_team_max_rounds_finalizes(runtime: Runtime) -> None:
    supervisor = _scripted_supervisor(
        [
            '{"agent": "worker", "subtask": "part 1", "done": false}',
            '{"agent": "worker", "subtask": "part 2", "done": false}',
            "synthesized from all observations",  # finalize() plain ask
        ]
    )
    team = Team({"worker": _echo_agent(runtime, "worker")}, supervisor, max_rounds=2)
    result = await team.run("big task")
    assert result.rounds == 2
    assert len(result.delegations) == 2
    assert result.answer == "synthesized from all observations"


def test_team_facade(runtime: Runtime) -> None:
    from aire.ai import _AgentsNamespace

    ns = _AgentsNamespace(runtime)
    member = ns.create_sync("mock:echo", name="member")
    team = ns.team({"member": member}, supervisor="mock:echo")
    assert isinstance(team, Team)
    described = team.describe()
    assert described["kind"] == "team"
    assert described["members"]["member"] == "mock:echo"


@pytest.mark.anyio
async def test_team_requires_members(runtime: Runtime) -> None:
    from aire.core.errors import AireError

    with pytest.raises(AireError):
        Team({}, _scripted_supervisor(["x"]))

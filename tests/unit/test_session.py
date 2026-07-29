"""DurableSession save / load / hydrate."""

from __future__ import annotations

from pathlib import Path

from aire.agents.session import DurableSession
from aire.agents.types import AgentStatus, AgentStep, StepKind
from aire.core.content import Message


def test_save_load_hydrate(tmp_session_path: Path) -> None:
    session = DurableSession(tmp_session_path)
    session.state.goal = "finish the task"
    session.persist_messages([Message.text("user", "hi"), Message.text("assistant", "hello")])
    session.append_step(
        AgentStep(index=0, kind=StepKind.OBSERVATION, detail={"note": "step0"})
    )
    path = session.save()
    assert path.is_file()

    loaded = DurableSession(tmp_session_path)
    assert loaded.state.goal == "finish the task"
    assert len(loaded.state.messages) == 2
    assert len(loaded.state.steps) == 1

    state = loaded.hydrate_agent_state()
    assert state.input == "finish the task"
    assert len(state.messages) == 2
    assert len(state.steps) == 1
    assert state.status == AgentStatus.RUNNING

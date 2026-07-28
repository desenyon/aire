"""Tool contracts, permissions, builtin tools, and the agent state machine."""

from __future__ import annotations

import pytest

from aire.agents import Agent, AgentConfig, AgentStatus, BufferMemory, JsonlMemory
from aire.core.content import Message
from aire.core.context import ExecutionContext
from aire.core.errors import NotFoundError, PermissionDeniedError
from aire.core.runtime import Runtime
from aire.models.builtin import EchoModel
from aire.models.types import ToolCall
from aire.tools import SideEffect, ToolRegistry, builtin_tools, tool
from tests.conftest import arun


@tool(description="Add two numbers.")
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


def test_tool_schema_from_signature() -> None:
    assert add.spec.input_schema["properties"]["a"]["type"] == "integer"
    assert add.spec.input_schema["required"] == ["a", "b"]
    assert add.spec.description == "Add two numbers."
    definition = add.definition()
    assert definition.name == "add"


def test_tool_execute_success() -> None:
    result = arun(add.execute({"a": 2, "b": 3}))
    assert result.ok and result.output == 5


def test_tool_execute_invalid_args() -> None:
    result = arun(add.execute({"a": "not-an-int", "b": 1}))
    assert not result.ok
    assert result.error_code == "tool.arguments_invalid"


def test_tool_timeout() -> None:
    import asyncio

    @tool(timeout_seconds=0.05)
    async def slow() -> str:
        await asyncio.sleep(5)
        return "done"

    result = arun(slow.execute({}))
    assert not result.ok
    assert "timeout" in (result.error or "").lower()


def test_tool_permissions_enforced() -> None:
    @tool(permissions=["database.admin"])
    def drop_table(name: str) -> str:
        return f"dropped {name}"

    result = arun(drop_table.execute({"name": "users"}))
    assert not result.ok
    assert result.error_code == "safety.permission_denied"

    ctx = ExecutionContext(permissions={"database.admin"})
    ok = arun(drop_table.execute({"name": "users"}, context=ctx))
    assert ok.ok and ok.output == "dropped users"


def test_builtin_calculator_safety() -> None:
    tools = {t.name: t for t in builtin_tools()}
    calc = tools["calculator"]
    assert arun(calc.execute({"expression": "2 + 3 * 4"})).output == 14.0
    evil = arun(calc.execute({"expression": "__import__('os').system('id')"}))
    assert not evil.ok


def test_builtin_read_file_sandbox(tmp_path: object) -> None:
    from pathlib import Path

    root = Path(str(tmp_path))
    (root / "ok.txt").write_text("fine")
    tools = {t.name: t for t in builtin_tools()}
    read = tools["read_file"]
    assert (
        arun(read.execute({"path": str(root / "ok.txt"), "sandbox_root": str(root)})).output
        == "fine"
    )
    blocked = arun(read.execute({"path": "/etc/passwd", "sandbox_root": str(root)}))
    assert not blocked.ok
    assert "sandbox" in (blocked.error or "")


def test_agent_completes_without_tools(runtime: Runtime) -> None:
    agent = Agent(EchoModel(), runtime=runtime, name="simple")
    result = arun(agent.run("say hi"))
    assert result.status == AgentStatus.COMPLETED
    assert result.output == "say hi"
    assert [s.kind for s in result.steps] == ["model_call", "finish"]


def test_agent_tool_calling_loop(runtime: Runtime) -> None:
    model = EchoModel()
    model.scripted_tool_calls = [ToolCall(id="c1", name="add", arguments={"a": 2, "b": 2})]
    agent = Agent(model, tools=[add], runtime=runtime)
    result = arun(agent.run("what is 2+2?"))
    assert result.status == AgentStatus.COMPLETED
    kinds = [s.kind for s in result.steps]
    assert "tool_call" in kinds and "observation" in kinds
    tool_msgs = [m for m in agent.state.messages if m.role == "tool"] or None
    del tool_msgs  # state lives in the executor's run, verified via steps
    tool_steps = [s for s in result.steps if s.kind == "tool_call"]
    assert tool_steps[0].detail["ok"] is True


def test_agent_unknown_tool_observed(runtime: Runtime) -> None:
    model = EchoModel()
    model.scripted_tool_calls = [ToolCall(id="c1", name="ghost", arguments={})]
    agent = Agent(model, runtime=runtime)
    result = arun(agent.run("call ghost"))
    assert result.status == AgentStatus.COMPLETED
    assert any(s.kind == "error" for s in result.steps)


def test_agent_permission_denied_path(runtime: Runtime) -> None:
    @tool(permissions=["top.secret"])
    def classify(doc: str) -> str:
        return "classified"

    model = EchoModel()
    model.scripted_tool_calls = [ToolCall(id="c1", name="classify", arguments={"doc": "x"})]
    agent = Agent(model, tools=[classify], runtime=runtime)
    result = arun(agent.run("classify this"))
    assert any(s.kind == "permission_denied" for s in result.steps)


def test_agent_max_steps_budget(runtime: Runtime) -> None:
    class LoopModel(EchoModel):
        async def generate(self, request):  # type: ignore[override]
            self.scripted_tool_calls = [ToolCall(id="c", name="add", arguments={"a": 1, "b": 1})]
            return await super().generate(request)

    agent = Agent(LoopModel(), tools=[add], config=AgentConfig(max_steps=3), runtime=runtime)
    result = arun(agent.run("loop forever"))
    assert result.status == AgentStatus.MAX_STEPS
    assert len(result.steps) <= 9  # 3 iterations x (model call + tool call + observation)


def test_agent_approval_gate(runtime: Runtime) -> None:
    @tool(side_effect=SideEffect.HIGH_IMPACT)
    def nuke(target: str) -> str:
        return f"nuked {target}"

    model = EchoModel()
    model.scripted_tool_calls = [ToolCall(id="c1", name="nuke", arguments={"target": "db"})]

    denied = Agent(
        model,
        tools=[nuke],
        runtime=runtime,
        config=AgentConfig(approval_levels=["high_impact"]),
    )
    result = arun(denied.run("nuke the db"))
    assert any(s.kind == "permission_denied" and s.detail.get("approval") for s in result.steps)

    model.scripted_tool_calls = [ToolCall(id="c2", name="nuke", arguments={"target": "db"})]
    approved = Agent(
        EchoModel(),
        tools=[nuke],
        runtime=runtime,
        config=AgentConfig(approval_levels=["high_impact"]),
        approver=lambda call, spec: True,
    )
    approved.model.scripted_tool_calls = [
        ToolCall(id="c2", name="nuke", arguments={"target": "db"})
    ]
    result2 = arun(approved.run("nuke the db"))
    assert any(s.kind == "tool_call" and s.detail["ok"] for s in result2.steps)


def test_memory_buffer_window() -> None:
    memory = BufferMemory(window=2)
    for i in range(5):
        arun(memory.add(Message.text("user", f"m{i}")))
    recalled = arun(memory.recall())
    assert [m.text_content for m in recalled] == ["m3", "m4"]


def test_memory_jsonl_persists(tmp_path: object) -> None:
    from pathlib import Path

    path = Path(str(tmp_path)) / "memory.jsonl"
    memory = JsonlMemory(path)
    arun(memory.add(Message.text("user", "remember me")))
    restored = JsonlMemory(path)
    recalled = arun(restored.recall())
    assert recalled[0].text_content == "remember me"


def test_tool_registry_manifests() -> None:
    registry = ToolRegistry()
    registry.register(add)
    assert registry.has("add")
    manifests = registry.manifests()
    assert manifests[0]["name"] == "add"
    assert manifests[0]["input_schema"]["properties"]["b"]["type"] == "integer"
    with pytest.raises(NotFoundError):
        registry.get("missing")


def test_permission_error_type() -> None:
    err = PermissionDeniedError("act", "perm")
    assert err.code == "safety.permission_denied"

"""Cross-subsystem integration: agents with tools, workflows with agents."""

from __future__ import annotations

from aire.agents import Agent, AgentConfig, AgentStatus
from aire.core.runtime import Runtime
from aire.models.builtin import EchoModel
from aire.models.types import ToolCall
from aire.tools import tool
from aire.workflows import Workflow
from tests.conftest import arun


def test_agent_with_multiple_tools(runtime: Runtime) -> None:
    @tool(description="Get the weather for a city.")
    def get_weather(city: str) -> str:
        return f"sunny in {city}"

    @tool(description="Add two numbers.")
    def add(a: int, b: int) -> int:
        return a + b

    model = EchoModel()
    agent = Agent(model, tools=[get_weather, add], runtime=runtime, name="multi")

    model.scripted_tool_calls = [
        ToolCall(id="1", name="get_weather", arguments={"city": "Paris"}),
        ToolCall(id="2", name="add", arguments={"a": 1, "b": 2}),
    ]
    result = arun(agent.run("weather and math please"))
    assert result.status == AgentStatus.COMPLETED
    tool_steps = [s for s in result.steps if s.kind == "tool_call"]
    assert len(tool_steps) == 2
    assert all(s.detail["ok"] for s in tool_steps)
    assert result.usage.input_tokens > 0


def test_agent_as_workflow_node(runtime: Runtime) -> None:
    agent = Agent(EchoModel(), runtime=runtime, name="node-agent")

    async def agent_node(x: str, ctx: dict) -> str:
        return (await agent.run(x)).output

    wf = Workflow("agent-wf")
    wf.add("preprocess", lambda x, ctx: x.strip().upper())
    wf.add("agent", agent_node)
    wf.add("postprocess", lambda x, ctx: f"[{x}]")
    wf.connect("preprocess", "agent")
    wf.connect("agent", "postprocess")
    result = arun(wf.run("  hello workflow  "))
    assert result.ok
    assert result.output == "[HELLO WORKFLOW]"


def test_agent_memory_across_runs(runtime: Runtime) -> None:
    agent = Agent(EchoModel(), runtime=runtime, memory="buffer")
    arun(agent.run("first message"))
    arun(agent.run("second message"))
    recalled = arun(agent.memory.recall())
    texts = [m.text_content for m in recalled]
    assert "first message" in texts and "second message" in texts


def test_agent_describe_manifest(runtime: Runtime) -> None:
    @tool(description="A test tool.")
    def sample(x: int) -> int:
        return x

    agent = Agent(EchoModel(), tools=[sample], runtime=runtime, config=AgentConfig(max_steps=5))
    manifest = agent.describe()
    assert manifest["kind"] == "agent"
    assert manifest["tools"][0]["name"] == "sample"
    assert manifest["config"]["max_steps"] == 5

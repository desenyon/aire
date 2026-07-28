"""Tool-using agent — deterministic, budgeted, permission-checked, offline.

Run:  python examples/chatbot/main.py

The echo model is scripted to call tools so the whole loop runs without a
network. With a real provider ("openai:gpt-4o-mini") the same agent plans and
calls tools on its own.
"""

from __future__ import annotations

from aire import AI
from aire.agents import AgentConfig
from aire.models.builtin import EchoModel
from aire.models.types import ToolCall


@AI.tool(description="Add two integers.", side_effect="read_only")
def add(a: int, b: int) -> int:
    return a + b


@AI.tool(description="Multiply two integers.", side_effect="read_only")
def multiply(a: int, b: int) -> int:
    return a * b


def main() -> None:
    model = EchoModel()
    # Script the offline model: first call a tool, then answer.
    model.scripted_tool_calls = [ToolCall(id="c1", name="add", arguments={"a": 40, "b": 2})]

    agent = AI.agents.create_sync(
        model,
        tools=[add, multiply],
        config=AgentConfig(max_steps=4, token_budget=10_000),
    )
    result = agent.run_sync("what is 40 + 2?")

    print("status:", result.status)
    for step in result.steps:
        print(f"  step {step.index}: {step.kind} {step.detail}")
    print("final output:", result.output or "(answered after tool use)")
    print("tokens used:", result.usage.total_tokens)


if __name__ == "__main__":
    main()

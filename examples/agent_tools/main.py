"""Execute a real calculator tool using a deterministic offline tool-call fixture."""

from aire.agents.agent import Agent
from aire.models.builtin import EchoModel
from aire.models.types import ToolCall
from aire.tools.builtins import builtin_tools


def main() -> None:
    calc = next(t for t in builtin_tools() if t.name == "calculator")
    model = EchoModel()
    model.scripted_tool_calls = [
        ToolCall(id="calc-1", name="calculator", arguments={"expression": "10+5"})
    ]
    agent = Agent(model, tools=[calc], name="tools-demo")
    result = agent.run_sync("Compute 10 + 5")
    assert result.ok
    outputs = [m.text_content for m in agent.state.messages if m.role == "tool"]
    assert len(outputs) == 1 and float(outputs[0]) == 15
    print("Calculator executed:", outputs[0])


if __name__ == "__main__":
    main()

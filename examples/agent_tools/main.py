"""Agent with calculator tool, offline mock:echo."""

from aire import AI
from aire.tools.builtins import builtin_tools


def main() -> None:
    calc = next(t for t in builtin_tools() if t.name == "calculator")
    agent = AI.agents.create_sync("mock:echo", tools=[calc], name="tools-demo")
    result = agent.run_sync("Please compute 10 + 5")
    print("status:", result.status)
    print("output:", result.output)
    print("steps:", len(result.steps))


if __name__ == "__main__":
    main()

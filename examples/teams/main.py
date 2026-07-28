"""Multi-agent teams offline: specialist agents + a supervisor that routes.

The supervisor is a scripted callable model here so the example (and tests)
are deterministic. Point it at any real model ref for open-ended routing:

    team = AI.agents.team(members, supervisor="ollama:llama3.2")
"""

import asyncio

from aire import AI
from aire.agents import Team
from aire.models.builtin import CallableModel


async def run() -> None:
    researcher = await AI.agents.create("mock:echo", name="researcher")
    writer = await AI.agents.create("mock:echo", name="writer")

    # 1. agent-as-tool: any agent becomes a Tool other agents can call.
    research_tool = researcher.as_tool(description="Gather facts on a topic.")
    print("tool contract:", research_tool.definition().model_dump())

    # 2. supervisor-routed team: a model decides who handles each subtask.
    decisions = iter(
        [
            '{"agent": "researcher", "subtask": "gather key facts", "done": false}',
            '{"agent": "writer", "subtask": "write the summary", "done": false}',
            '{"done": true, "final_answer": "Report complete: facts gathered and written up."}',
        ]
    )
    supervisor = CallableModel("supervisor", lambda prompt: next(decisions))

    team = Team({"researcher": researcher, "writer": writer}, supervisor)
    result = await team.run("produce a one-page report on GraphRAG")

    print(f"\nanswer: {result.answer}")
    print(f"rounds: {result.rounds}")
    for record in result.delegations:
        print(f"  {record.agent} ← {record.subtask!r} → {record.output[:60]!r}")

    # 3. long-term memory: the writer remembers across runs.
    memory = AI.memory.create()
    await memory.remember("The user prefers one-page reports", salience=2.0)
    recalled = await memory.recall_semantic("report format preference", k=1)
    print(f"\nlong-term memory recall: {[m.text for m in recalled]}")


if __name__ == "__main__":
    asyncio.run(run())

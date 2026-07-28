"""Multi-agent topologies beyond supervisor-team: swarm, debate, auction, blackboard."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from aire.agents.agent import Agent
from aire.models.types import GenerationRequest


class TopologyResult(BaseModel):
    output: str
    mode: str
    rounds: int = 0
    transcripts: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


async def swarm(
    agents: list[Agent],
    goal: str,
    *,
    aggregator: Agent | None = None,
) -> TopologyResult:
    """Independent workers in parallel, then aggregate."""
    import asyncio

    results = await asyncio.gather(*[a.run(goal) for a in agents])
    transcripts = [
        {"agent": a.name, "output": r.output} for a, r in zip(agents, results, strict=True)
    ]
    if aggregator is None:
        combined = "\n\n".join(f"## {t['agent']}\n{t['output']}" for t in transcripts)
        return TopologyResult(output=combined, mode="swarm", rounds=1, transcripts=transcripts)
    brief = "\n\n".join(f"[{t['agent']}]: {t['output']}" for t in transcripts)
    final = await aggregator.run(f"Synthesize the best answer for: {goal}\n\n{brief}")
    return TopologyResult(
        output=final.output,
        mode="swarm",
        rounds=1,
        transcripts=[*transcripts, {"agent": aggregator.name, "output": final.output}],
    )


async def debate(
    agents: list[Agent],
    goal: str,
    *,
    rounds: int = 2,
    judge: Agent | None = None,
) -> TopologyResult:
    """Agents argue in turns; optional judge picks the winner."""
    if len(agents) < 2:
        raise ValueError("debate needs at least 2 agents")
    transcripts: list[dict[str, Any]] = []
    last = goal
    for r in range(rounds):
        for agent in agents:
            prompt = (
                f"Debate round {r + 1}. Goal: {goal}\n"
                f"Previous: {last}\nArgue your position clearly."
            )
            result = await agent.run(prompt)
            transcripts.append({"round": r + 1, "agent": agent.name, "output": result.output})
            last = result.output
    if judge is None:
        return TopologyResult(
            output=last, mode="debate", rounds=rounds, transcripts=transcripts
        )
    brief = "\n".join(f"[{t['agent']} r{t['round']}]: {t['output']}" for t in transcripts)
    verdict = await judge.run(f"Judge this debate on: {goal}\n\n{brief}\n\nFinal answer:")
    return TopologyResult(
        output=verdict.output,
        mode="debate",
        rounds=rounds,
        transcripts=[*transcripts, {"agent": judge.name, "output": verdict.output}],
    )


async def auction(
    agents: list[Agent],
    goal: str,
    *,
    auctioneer: Agent | None = None,
) -> TopologyResult:
    """Each agent bids a plan + confidence; auctioneer (or heuristic) awards the job."""
    bids: list[dict[str, Any]] = []
    for agent in agents:
        result = await agent.run(
            f"Bid on this task. Reply with a plan and a confidence 0-1.\nTask: {goal}"
        )
        conf = _extract_confidence(result.output)
        bids.append(
            {"agent": agent.name, "bid": result.output, "confidence": conf, "agent_obj": agent}
        )
    if auctioneer is not None:
        brief = "\n\n".join(f"[{b['agent']} conf={b['confidence']}]: {b['bid']}" for b in bids)
        pick = await auctioneer.run(
            f"Pick the best bidder for: {goal}\n\n{brief}\nReply with the agent name first."
        )
        winner_name = pick.output.strip().split()[0].strip("[]:.,")
        winner = next(
            (b for b in bids if b["agent"] == winner_name),
            max(bids, key=lambda b: b["confidence"]),
        )
    else:
        winner = max(bids, key=lambda b: float(b["confidence"]))
    final = await winner["agent_obj"].run(f"You won the auction. Execute fully:\n{goal}")
    return TopologyResult(
        output=final.output,
        mode="auction",
        rounds=1,
        transcripts=[
            *[{"agent": b["agent"], "bid": b["bid"], "confidence": b["confidence"]} for b in bids],
            {"agent": winner["agent"], "output": final.output},
        ],
        metadata={"winner": winner["agent"]},
    )


async def blackboard(
    agents: list[Agent],
    goal: str,
    *,
    rounds: int = 3,
) -> TopologyResult:
    """Shared scratchpad; agents read/write contributions each round."""
    board: list[str] = [f"GOAL: {goal}"]
    transcripts: list[dict[str, Any]] = []
    for r in range(rounds):
        for agent in agents:
            state = "\n".join(board[-20:])
            result = await agent.run(
                f"Blackboard round {r + 1}. Add one useful note or partial answer.\n"
                f"Current board:\n{state}"
            )
            note = result.output.strip()
            board.append(f"[{agent.name}] {note}")
            transcripts.append({"round": r + 1, "agent": agent.name, "output": note})
    # final synthesis via first agent
    synth = await agents[0].model.generate(
        GenerationRequest.of(
            f"Produce the final answer from this blackboard for: {goal}\n\n"
            + "\n".join(board)
        )
    )
    return TopologyResult(
        output=synth.text,
        mode="blackboard",
        rounds=rounds,
        transcripts=transcripts,
        metadata={"board": board},
    )


def _extract_confidence(text: str) -> float:
    import re

    match = re.search(r"confidence[:\s]+([01](?:\.\d+)?)", text, re.I)
    if match:
        return float(match.group(1))
    match = re.search(r"\b(0?\.\d+|1(?:\.0+)?)\b", text)
    if match:
        val = float(match.group(1))
        if 0.0 <= val <= 1.0:
            return val
    return 0.5

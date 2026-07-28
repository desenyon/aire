"""Multi-agent teams: a supervisor model routes tasks to specialist agents.

The supervisor decides each step with validated structured output: delegate a
subtask to a named member, or finish with a final answer. Member outputs feed
back as observations, so routing decisions stay grounded in what specialists
actually returned. Works offline with any model (including scripted
``callable:`` models for deterministic tests).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from aire.agents.agent import Agent
from aire.core.errors import AireError
from aire.core.types import Usage
from aire.models.base import Model

SUPERVISOR_PROMPT = (
    "You are the supervisor of a team of specialist agents.\n"
    "Team members:\n{roster}\n\n"
    "Decide the next step for the user's task. Either delegate one subtask to "
    "exactly one member (set agent + subtask), or, if the task is fully answered "
    "by the observations so far, set done=true and write final_answer.\n\n"
    "Task: {task}\n\nObservations so far:\n{observations}"
)

_FINALIZE_PROMPT = (
    "Synthesize the final answer to the task from the specialist observations.\n\n"
    "Task: {task}\n\nObservations:\n{observations}"
)


class Delegation(BaseModel):
    """The supervisor's structured decision for one round."""

    agent: str | None = None
    subtask: str = ""
    done: bool = False
    final_answer: str = ""


class DelegationRecord(BaseModel):
    """One executed delegation (auditable handoff)."""

    agent: str
    subtask: str
    output: str
    ok: bool = True


class TeamResult(BaseModel):
    """Outcome of a team run."""

    answer: str
    delegations: list[DelegationRecord] = Field(default_factory=list)
    rounds: int = 0
    usage: Usage = Field(default_factory=Usage)


class Team:
    """A supervisor-routed group of agents."""

    def __init__(
        self,
        members: dict[str, Agent] | list[Agent],
        supervisor: Model,
        *,
        max_rounds: int = 6,
        prompt_template: str = SUPERVISOR_PROMPT,
    ) -> None:
        if isinstance(members, list):
            members = {agent.name: agent for agent in members}
        if not members:
            raise AireError("a team needs at least one member agent", code="team.empty")
        self.members = dict(members)
        self.supervisor = supervisor
        self.max_rounds = max_rounds
        self.prompt_template = prompt_template

    def _roster(self) -> str:
        lines = []
        for name, agent in self.members.items():
            tools = ", ".join(t.name for t in agent.registry) or "no tools"
            lines.append(f"- {name}: {tools}")
        return "\n".join(lines)

    async def run(self, task: str) -> TeamResult:
        """Run the team: supervisor routes, members execute, answer is synthesized."""
        observations: list[str] = []
        records: list[DelegationRecord] = []
        usage = Usage()
        for round_index in range(self.max_rounds):
            prompt = self.prompt_template.format(
                roster=self._roster(),
                task=task,
                observations="\n".join(observations) or "(none yet)",
            )
            decision = Delegation.model_validate(
                await self.supervisor.ask_structured(prompt, Delegation)
            )
            if decision.done or not decision.agent:
                return TeamResult(
                    answer=decision.final_answer or await self._finalize(task, observations),
                    delegations=records,
                    rounds=round_index + 1,
                    usage=usage,
                )
            member = self.members.get(decision.agent)
            if member is None:
                observations.append(
                    f"supervisor picked unknown member {decision.agent!r}; "
                    f"available: {', '.join(self.members)}"
                )
                continue
            result = await member.run(decision.subtask)
            usage = usage + result.usage
            records.append(
                DelegationRecord(
                    agent=decision.agent,
                    subtask=decision.subtask,
                    output=result.output,
                    ok=result.status == "completed",
                )
            )
            observations.append(f"[{decision.agent}] {result.output}")
        return TeamResult(
            answer=await self._finalize(task, observations),
            delegations=records,
            rounds=self.max_rounds,
            usage=usage,
        )

    async def _finalize(self, task: str, observations: list[str]) -> str:
        prompt = _FINALIZE_PROMPT.format(
            task=task, observations="\n".join(observations) or "(none)"
        )
        return await self.supervisor.ask(prompt)

    def describe(self) -> dict[str, Any]:
        """Machine-readable team manifest — for agents."""
        return {
            "kind": "team",
            "members": {name: agent.describe()["model"] for name, agent in self.members.items()},
            "supervisor": self.supervisor.info.ref,
            "max_rounds": self.max_rounds,
        }

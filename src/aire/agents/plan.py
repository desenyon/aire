"""Plan → act → verify agent loop (deterministic, budgeted)."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field

from aire.agents.agent import Agent
from aire.agents.types import AgentResult
from aire.models.types import GenerationRequest


class PlanStep(BaseModel):
    id: str
    action: str
    tool: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)
    done: bool = False
    observation: str = ""


class Plan(BaseModel):
    goal: str
    steps: list[PlanStep] = Field(default_factory=list)
    verified: bool = False
    notes: str = ""


class PlanActVerify:
    """Three-phase agent: draft a plan, execute steps, verify the goal."""

    def __init__(self, agent: Agent, *, max_repair: int = 2) -> None:
        self.agent = agent
        self.max_repair = max_repair

    async def run(self, goal: str) -> AgentResult:
        plan = await self._draft_plan(goal)
        for step in plan.steps:
            if step.tool:
                prompt = (
                    f"Goal: {goal}\nExecute plan step {step.id}: {step.action}\n"
                    f"Prefer tool {step.tool} with args {json.dumps(step.args)}."
                )
            else:
                prompt = f"Goal: {goal}\nExecute plan step {step.id}: {step.action}"
            partial = await self.agent.run(prompt)
            step.observation = partial.output
            step.done = True
        verified, notes = await self._verify(goal, plan)
        plan.verified = verified
        plan.notes = notes
        repairs = 0
        while not verified and repairs < self.max_repair:
            repairs += 1
            repair = await self.agent.run(
                f"Goal not met ({notes}). Goal: {goal}. "
                f"Plan so far: {plan.model_dump_json()}. Fix the remaining gaps."
            )
            plan.notes = repair.output
            verified, notes = await self._verify(goal, plan)
            plan.verified = verified
            plan.notes = notes
        # Final synthesis
        result = await self.agent.run(
            f"Summarize the completed plan for: {goal}\n"
            f"Plan JSON: {plan.model_dump_json()}\nVerified: {plan.verified}"
        )
        result.metadata = {
            **(result.metadata or {}),
            "plan": plan.model_dump(mode="json"),
            "verified": plan.verified,
        }
        return result

    async def _draft_plan(self, goal: str) -> Plan:
        prompt = (
            "Draft a short JSON plan to achieve the goal. "
            'Format: {"steps":[{"id":"1","action":"...","tool":null,"args":{}}]}\n'
            f"Goal: {goal}"
        )
        text = (await self.agent.model.generate(GenerationRequest.of(prompt))).text
        steps = _parse_steps(text)
        if not steps:
            steps = [PlanStep(id="1", action=goal)]
        return Plan(goal=goal, steps=steps)

    async def _verify(self, goal: str, plan: Plan) -> tuple[bool, str]:
        prompt = (
            f"Did we achieve the goal?\nGoal: {goal}\n"
            f"Plan: {plan.model_dump_json()}\n"
            'Reply JSON: {"ok": true|false, "notes": "..."}'
        )
        text = (await self.agent.model.generate(GenerationRequest.of(prompt))).text
        ok = "true" in text.lower().split("ok")[-1][:20] if "ok" in text.lower() else False
        # also accept YES at start
        if text.strip().lower().startswith("yes") or '"ok": true' in text.lower():
            ok = True
        if text.strip().lower().startswith("no") or '"ok": false' in text.lower():
            ok = False
        return ok, text.strip()[:500]


def _parse_steps(text: str) -> list[PlanStep]:
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return []
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    raw = payload.get("steps") if isinstance(payload, dict) else None
    if not isinstance(raw, list):
        return []
    steps: list[PlanStep] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        steps.append(
            PlanStep(
                id=str(item.get("id", i + 1)),
                action=str(item.get("action", "")),
                tool=item.get("tool"),
                args=dict(item.get("args") or {}),
            )
        )
    return steps

"""Agent streaming events — yield each state-machine step as it happens."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel, Field

from aire.agents.runtime import AgentExecutor
from aire.agents.types import AgentResult, AgentState, AgentStatus, AgentStep
from aire.core.content import Message
from aire.core.context import Budget, ExecutionContext
from aire.core.errors import BudgetExceededError
from aire.models.types import GenerationRequest


class AgentEvent(BaseModel):
    """One streamed event from an agent run."""

    type: str  # step | status | output | error | done
    step: AgentStep | None = None
    status: str | None = None
    output: str | None = None
    error: str | None = None
    run_id: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


async def run_stream(
    executor: AgentExecutor,
    input: str,
    *,
    state: AgentState | None = None,
) -> AsyncIterator[AgentEvent]:
    """Stream agent steps. Terminal event is ``type=done`` with full result fields."""
    ctx = ExecutionContext(
        permissions=set(executor.config.permissions),
        budget=Budget(
            max_tokens=executor.config.token_budget,
            max_cost_usd=executor.config.cost_budget_usd,
            max_steps=executor.config.max_steps,
        ),
    )
    agent_state = state or AgentState(input=input)
    if not agent_state.messages:
        if executor.config.system_prompt:
            agent_state.messages.append(Message.text("system", executor.config.system_prompt))
        if executor.memory is not None:
            agent_state.messages.extend(await executor.memory.recall())
        agent_state.messages.append(Message.text("user", input))
    agent_state.status = AgentStatus.RUNNING
    yield AgentEvent(type="status", status=str(agent_state.status), run_id=agent_state.id)

    while agent_state.status == AgentStatus.RUNNING:
        before = len(agent_state.steps)
        try:
            ctx.tick()
        except BudgetExceededError as exc:
            agent_state.status = (
                AgentStatus.MAX_STEPS if "step" in exc.message else AgentStatus.BUDGET_EXCEEDED
            )
            agent_state.error = exc.message
            yield AgentEvent(
                type="error",
                error=exc.message,
                status=str(agent_state.status),
                run_id=agent_state.id,
            )
            break
        await executor._transition(agent_state, ctx)
        for step in agent_state.steps[before:]:
            yield AgentEvent(type="step", step=step, run_id=agent_state.id)
        # status is mutated inside _transition; re-read without while-narrowing
        finished = agent_state.status != AgentStatus.RUNNING
        if finished and agent_state.output:
            yield AgentEvent(
                type="output",
                output=agent_state.output,
                status=str(agent_state.status),
                run_id=agent_state.id,
            )

    result = AgentResult(
        output=agent_state.output or "",
        status=agent_state.status,
        steps=list(agent_state.steps),
        usage=agent_state.usage,
        run_id=agent_state.id,
        error=agent_state.error,
    )
    yield AgentEvent(
        type="done",
        status=str(result.status),
        output=result.output,
        error=result.error,
        run_id=result.run_id,
        metadata={"steps": len(result.steps), "usage": result.usage.model_dump()},
    )


async def stream_tokens(
    executor: AgentExecutor,
    input: str,
) -> AsyncIterator[str]:
    """Stream final-answer text.

    When tools are registered, runs the full tool loop via :func:`run_stream`
    and yields the completed answer (models cannot stream mid-tool-loop).
    Without tools, streams model tokens directly when the provider supports it.
    """
    if executor.tools.definitions():
        final = ""
        async for event in run_stream(executor, input):
            if (event.type == "output" and event.output) or (event.type == "done" and event.output):
                final = event.output
        if final:
            yield final
        return

    messages: list[Message] = []
    if executor.config.system_prompt:
        messages.append(Message.text("system", executor.config.system_prompt))
    if executor.memory is not None:
        messages.extend(await executor.memory.recall())
    messages.append(Message.text("user", input))
    request = GenerationRequest(messages=messages, temperature=executor.config.temperature)
    async for chunk in executor.model.stream(request):
        if chunk.text:
            yield chunk.text

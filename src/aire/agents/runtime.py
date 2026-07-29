"""The deterministic agent execution engine.

The agent loop is an explicit state machine — never unbounded recursion:

    RUNNING
      → MODEL_CALL        (model proposes text or tool calls)
      → PERMISSION_DENIED (policy blocked a proposed action; fed back as observation)
      → TOOL_CALL         (approved tool executes)
      → OBSERVATION       (tool result appended to state)
      → FINISH            (model produced a final answer)
      → MAX_STEPS | BUDGET_EXCEEDED | FAILED

Every transition is recorded as an :class:`AgentStep`, so executions are fully
auditable, replayable and traceable.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from aire.agents.types import (
    AgentConfig,
    AgentResult,
    AgentState,
    AgentStatus,
    AgentStep,
    StepKind,
)
from aire.core.content import Message
from aire.core.context import Budget, ExecutionContext
from aire.core.errors import BudgetExceededError
from aire.core.types import Usage
from aire.models.base import Model
from aire.models.types import GenerationRequest, ToolCall
from aire.tools.registry import ToolRegistry
from aire.tools.types import ToolSpec

if TYPE_CHECKING:
    from aire.agents.memory import Memory
    from aire.core.runtime import Runtime
    from aire.safety.policy import PolicyEngine

Approver = Callable[[ToolCall, ToolSpec], bool | Awaitable[bool]]


def _deny_by_default(call: ToolCall, spec: ToolSpec) -> bool:
    """Default approval policy: nothing sensitive is auto-approved."""
    return False


def _resolve_policy(policy: PolicyEngine | None, config: AgentConfig) -> PolicyEngine | None:
    """Attach ``default_engine()`` when approval_levels include external-or-higher."""
    if policy is not None:
        return policy
    levels = {str(x) for x in config.approval_levels}
    if levels & {"external_side_effect", "high_impact", "prohibited"}:
        from aire.safety.policy import default_engine

        return default_engine()
    return None


class AgentExecutor:
    """Runs an agent (model + tools + memory + policy) to completion."""

    def __init__(
        self,
        model: Model,
        tools: ToolRegistry,
        *,
        config: AgentConfig | None = None,
        memory: Memory | None = None,
        runtime: Runtime | None = None,
        approver: Approver | None = None,
        policy: PolicyEngine | None = None,
    ) -> None:
        self.model = model
        self.tools = tools
        self.config = config or AgentConfig()
        self.memory = memory
        self.runtime = runtime
        self.approver = approver or _deny_by_default
        self.policy = _resolve_policy(policy, self.config)

    async def run(
        self,
        input: str,
        *,
        context: ExecutionContext | None = None,
        state: AgentState | None = None,
    ) -> AgentResult:
        """Execute the state machine until a terminal state. Fully resumable via ``state``."""
        ctx = context or ExecutionContext(
            permissions=set(self.config.permissions),
            budget=Budget(
                max_tokens=self.config.token_budget,
                max_cost_usd=self.config.cost_budget_usd,
                max_steps=self.config.max_steps,
            ),
        )
        agent_state = state or AgentState(input=input)
        if not agent_state.messages:
            if self.config.system_prompt:
                agent_state.messages.append(Message.text("system", self.config.system_prompt))
            if self.memory is not None:
                agent_state.messages.extend(await self.memory.recall())
            agent_state.messages.append(Message.text("user", input))
        agent_state.status = AgentStatus.RUNNING

        while agent_state.status == AgentStatus.RUNNING:
            try:
                ctx.tick()
            except BudgetExceededError as exc:
                agent_state.status = (
                    AgentStatus.MAX_STEPS if "step" in exc.message else AgentStatus.BUDGET_EXCEEDED
                )
                agent_state.error = exc.message
                break
            await self._transition(agent_state, ctx)

        return AgentResult(
            output=agent_state.output or "",
            status=agent_state.status,
            steps=list(agent_state.steps),
            usage=agent_state.usage,
            run_id=agent_state.id,
            error=agent_state.error,
        )

    # -- state machine ----------------------------------------------------------------

    async def _transition(self, state: AgentState, ctx: ExecutionContext) -> None:
        """Advance the machine by exactly one model call + any resulting tool calls."""
        request = GenerationRequest(
            messages=state.messages,
            tools=self.tools.definitions() or None,
            temperature=self.config.temperature,
        )
        self._emit("agent.model_call", {"run_id": state.id, "messages": len(state.messages)})
        result = await self.model.generate(request)
        state.usage = state.usage + result.usage
        try:
            ctx.record_usage(result.usage)
        except BudgetExceededError as exc:
            state.status = AgentStatus.BUDGET_EXCEEDED
            state.error = exc.message
            return
        self._append_step(
            state,
            StepKind.MODEL_CALL,
            usage=result.usage,
            text=result.text[:500],
            tool_calls=[c.name for c in result.tool_calls],
        )

        if not result.tool_calls:
            state.output = result.text
            state.status = AgentStatus.COMPLETED
            self._append_step(state, StepKind.FINISH, chars=len(result.text))
            self._emit("agent.finished", {"run_id": state.id, "steps": len(state.steps)})
            return

        state.messages.append(Message(role="assistant", content=result.content))
        if self.config.parallel_tools and len(result.tool_calls) > 1:
            await self._execute_tools_parallel(state, ctx, result.tool_calls)
        else:
            for call in result.tool_calls:
                if state.status != AgentStatus.RUNNING:
                    break
                await self._execute_tool(state, ctx, call)

    async def _policy_gate(
        self, call: ToolCall, spec: ToolSpec, ctx: ExecutionContext
    ) -> tuple[bool, str | None]:
        """Apply PolicyEngine if set. Returns (allowed_to_execute, deny_message)."""
        if self.policy is None:
            return True, None
        action = self.policy.decide(
            tool=call.name,
            side_effect=spec.side_effect,
            permissions=list(ctx.permissions) or list(self.config.permissions),
        )
        if action == "deny":
            return False, "error: permission denied (policy)"
        if action == "require_approval":
            approved = await self._approve(call, spec)
            if not approved:
                return False, "error: action requires human approval (denied)"
        return True, None

    async def _execute_tools_parallel(  # noqa: C901
        self,
        state: AgentState,
        ctx: ExecutionContext,
        calls: list[ToolCall],
    ) -> None:
        """Run independent tool bodies concurrently; append observations in call order."""

        async def _body(call: ToolCall) -> tuple[ToolCall, str, dict[str, Any]]:
            detail: dict[str, Any] = {"tool": call.name}
            if not self.tools.has(call.name):
                return (
                    call,
                    f"error: unknown tool {call.name!r}",
                    {**detail, "reason": "unknown_tool"},
                )
            tool = self.tools.get(call.name)
            missing = [p for p in tool.spec.permissions if p not in ctx.permissions]
            if missing:
                return (
                    call,
                    f"error: permission denied (missing: {', '.join(missing)})",
                    {**detail, "missing": missing, "kind": "permission_denied"},
                )
            # Policy engine (deny / require_approval) when present
            if self.policy is not None:
                allowed, deny_msg = await self._policy_gate(call, tool.spec, ctx)
                if not allowed:
                    return (
                        call,
                        deny_msg or "error: permission denied (policy)",
                        {**detail, "policy": True, "kind": "permission_denied"},
                    )
            elif str(tool.spec.side_effect) in self.config.approval_levels:
                approved = await self._approve(call, tool.spec)
                if not approved:
                    return (
                        call,
                        "error: action requires human approval (denied)",
                        {**detail, "approval": True, "kind": "permission_denied"},
                    )
            self._emit("agent.tool_call", {"run_id": state.id, "tool": call.name})
            try:
                outcome = await tool.execute(call.arguments, context=ctx)
            except Exception as exc:
                return (
                    call,
                    f"error: {exc}",
                    {**detail, "ok": False, "error": str(exc), "kind": "tool_call"},
                )
            observation = outcome.output if outcome.ok else f"error: {outcome.error}"
            return (
                call,
                observation if isinstance(observation, str) else _to_text(observation),
                {
                    **detail,
                    "ok": outcome.ok,
                    "duration_ms": outcome.duration_ms,
                    "call": call.model_dump(mode="json"),
                    "kind": "tool_call",
                },
            )

        outcomes = await asyncio.gather(*[_body(call) for call in calls], return_exceptions=False)
        for call, observation, detail in outcomes:
            kind = detail.pop("kind", "tool_call")
            if kind == "permission_denied":
                self._append_step(state, StepKind.PERMISSION_DENIED, **detail)
            elif detail.get("reason") == "unknown_tool":
                self._append_step(state, StepKind.ERROR, **detail)
            else:
                self._append_step(state, StepKind.TOOL_CALL, **detail)
            self._observe(state, call, observation)

    async def _execute_tool(self, state: AgentState, ctx: ExecutionContext, call: ToolCall) -> None:
        """Permission check → policy/approval check → execute → observe."""
        if not self.tools.has(call.name):
            self._observe(state, call, f"error: unknown tool {call.name!r}")
            self._append_step(state, StepKind.ERROR, tool=call.name, reason="unknown_tool")
            return
        tool = self.tools.get(call.name)

        missing = [p for p in tool.spec.permissions if p not in ctx.permissions]
        if missing:
            self._append_step(state, StepKind.PERMISSION_DENIED, tool=call.name, missing=missing)
            self._observe(state, call, f"error: permission denied (missing: {', '.join(missing)})")
            return

        if self.policy is not None:
            allowed, deny_msg = await self._policy_gate(call, tool.spec, ctx)
            if not allowed:
                self._append_step(
                    state, StepKind.PERMISSION_DENIED, tool=call.name, policy=True
                )
                self._observe(state, call, deny_msg or "error: permission denied (policy)")
                return
        elif str(tool.spec.side_effect) in self.config.approval_levels:
            approved = await self._approve(call, tool.spec)
            if not approved:
                self._append_step(state, StepKind.PERMISSION_DENIED, tool=call.name, approval=True)
                self._observe(state, call, "error: action requires human approval (denied)")
                return

        self._emit("agent.tool_call", {"run_id": state.id, "tool": call.name})
        outcome = await tool.execute(call.arguments, context=ctx)
        self._append_step(
            state,
            StepKind.TOOL_CALL,
            call=call.model_dump(mode="json"),
            ok=outcome.ok,
            duration_ms=outcome.duration_ms,
        )
        observation = outcome.output if outcome.ok else f"error: {outcome.error}"
        self._observe(state, call, observation)

    def _observe(self, state: AgentState, call: ToolCall, observation: Any) -> None:
        text = observation if isinstance(observation, str) else _to_text(observation)
        state.messages.append(
            Message(role="tool", content=[], name=call.name, tool_call_id=call.id).model_copy(
                update={"content": [_text_content(text)]}
            )
        )
        self._append_step(state, StepKind.OBSERVATION, tool=call.name, chars=len(text))

    def _append_step(
        self, state: AgentState, kind: StepKind, usage: Usage | None = None, **detail: Any
    ) -> AgentStep:
        step = AgentStep(index=len(state.steps), kind=kind, detail=detail, usage=usage or Usage())
        state.steps.append(step)
        return step

    async def _approve(self, call: ToolCall, spec: ToolSpec) -> bool:
        decision = self.approver(call, spec)
        if inspect.isawaitable(decision):
            return bool(await decision)
        return bool(decision)

    def _emit(self, topic: str, data: dict[str, Any]) -> None:
        if self.runtime is not None:
            self.runtime.events.emit(topic, data, source="agent")


def _text_content(text: str) -> Any:
    from aire.core.content import TextContent

    return TextContent(text=text)


def _to_text(value: Any) -> str:
    import json

    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return str(value)

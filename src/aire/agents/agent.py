"""The Agent: model + tools + memory + policy, with a fluent builder."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aire.agents.memory import Memory, resolve_memory
from aire.agents.runtime import AgentExecutor, Approver
from aire.agents.session import DurableSession
from aire.agents.types import AgentConfig, AgentResult, AgentState, AgentStatus
from aire.core.content import Message
from aire.models.base import Model
from aire.tools.registry import ToolRegistry
from aire.tools.tool import Tool

if TYPE_CHECKING:
    from aire.agents.streaming import AgentEvent
    from aire.core.runtime import Runtime
    from aire.safety.policy import PolicyEngine


class Agent:
    """A composable agent. Create via ``AI.agents.create(...)`` or directly."""

    def __init__(
        self,
        model: Model,
        *,
        tools: list[Tool] | ToolRegistry | None = None,
        memory: Memory | str | None = None,
        config: AgentConfig | None = None,
        runtime: Runtime | None = None,
        approver: Approver | None = None,
        policy: PolicyEngine | None = None,
        name: str = "agent",
        session: DurableSession | str | Path | None = None,
    ) -> None:
        self.model = model
        self.name = name
        self.runtime = runtime
        self.registry = tools if isinstance(tools, ToolRegistry) else ToolRegistry()
        if isinstance(tools, list):
            for t in tools:
                self.registry.register(t)
        self.memory = resolve_memory(memory)
        self.config = config or AgentConfig()
        self.approver = approver
        self.policy = policy
        self.state = AgentState()
        self._skills: list[str] = []
        if isinstance(session, DurableSession):
            self.session: DurableSession | None = session
        elif session is not None:
            self.session = DurableSession(session)
        else:
            self.session = None

    def _executor(self) -> AgentExecutor:
        return AgentExecutor(
            self.model,
            self.registry,
            config=self.config,
            memory=self.memory,
            runtime=self.runtime,
            approver=self.approver,
            policy=self.policy,
        )

    # -- execution -----------------------------------------------------------------

    async def run(  # noqa: C901
        self, input: str, *, use_planning: bool | None = None
    ) -> AgentResult:
        """Run the agent to a terminal state and persist memory (+ optional session).

        ``use_planning`` overrides ``config.planning``. When planning is on,
        delegates to :class:`~aire.agents.plan.PlanActVerify` with
        ``use_planning=False`` on nested calls to avoid recursion.
        """
        planning = self.config.planning if use_planning is None else use_planning
        if planning:
            from aire.agents.plan import PlanActVerify

            return await PlanActVerify(self).run(input)

        # Resume from durable session when paused/running with prior messages
        resumed = False
        if (
            self.session is not None
            and self.session.state.status in ("paused", "running")
            and self.session.state.messages
        ):
            self.state = self.session.hydrate_agent_state(input)
            # Continuing turn: append new user input if not already the last message
            if input and (
                not self.state.messages
                or self.state.messages[-1].role != "user"
                or self.state.messages[-1].text_content != input
            ):
                self.state.messages.append(Message.text("user", input))
            self.state.status = AgentStatus.RUNNING
            resumed = True
        else:
            self.state = AgentState(input=input)

        if self.session is not None:
            self.session.state.goal = input if not resumed else (self.session.state.goal or input)
            self.session.state.status = "running"
            self.session.save()

        executor = self._executor()
        try:
            result = await executor.run(input, state=self.state)
        except Exception as exc:
            if self.session is not None:
                self.session.fail(str(exc))
            raise
        await self.memory.add(Message.text("user", input))
        for message in self.state.messages:
            if message.role == "tool":
                await self.memory.add(message)
        if result.output:
            await self.memory.add(Message.text("assistant", result.output))
        if self.session is not None:
            self.session.persist_messages(self.state.messages)
            for step in result.steps:
                # Avoid duplicating steps already in session on resume
                if not any(s.get("index") == step.index for s in self.session.state.steps):
                    self.session.append_step(step)
            self.session.complete(result)
        return result

    async def pause(self) -> None:
        """Save current messages to the session and mark it paused."""
        if self.session is None:
            return
        self.session.persist_messages(self.state.messages)
        if self.state.steps:
            # Keep steps in sync
            existing = {s.get("index") for s in self.session.state.steps}
            for step in self.state.steps:
                if step.index not in existing:
                    self.session.state.steps.append(step.model_dump(mode="json"))
        self.session.pause()

    async def ask(self, input: str) -> str:
        """Convenience: run and return just the final text."""
        return (await self.run(input)).output

    async def run_stream(self, input: str) -> AsyncIterator[AgentEvent]:
        """Stream each agent step as an :class:`~aire.agents.streaming.AgentEvent`."""
        from aire.agents.streaming import run_stream

        self.state = AgentState(input=input)
        self._session_start(input)
        executor = self._executor()
        final: AgentResult | None = None
        try:
            async for event in run_stream(executor, input, state=self.state):
                if event.type == "done":
                    final = self._result_from_done(event)
                yield event
        except Exception as exc:
            if self.session is not None:
                self.session.fail(str(exc))
            raise
        await self._persist_turn(input, final)

    def _session_start(self, input: str) -> None:
        if self.session is None:
            return
        self.session.state.goal = input
        self.session.state.status = "running"
        self.session.save()

    def _result_from_done(self, event: AgentEvent) -> AgentResult:
        return AgentResult(
            output=event.output or "",
            status=AgentStatus(event.status) if event.status else self.state.status,
            steps=list(self.state.steps),
            usage=self.state.usage,
            run_id=event.run_id or self.state.id,
            error=event.error,
        )

    async def _persist_turn(self, input: str, final: AgentResult | None) -> None:
        await self.memory.add(Message.text("user", input))
        for message in self.state.messages:
            if message.role == "tool":
                await self.memory.add(message)
        if final and final.output:
            await self.memory.add(Message.text("assistant", final.output))
        if self.session is not None and final is not None:
            self.session.persist_messages(self.state.messages)
            for step in final.steps:
                if not any(s.get("index") == step.index for s in self.session.state.steps):
                    self.session.append_step(step)
            self.session.complete(final)

    def run_sync(self, input: str, *, use_planning: bool | None = None) -> AgentResult:
        from aire.models.base import run_sync

        return run_sync(self.run(input, use_planning=use_planning))

    def reset(self) -> None:
        """Start a fresh conversation (keeps tools and config, clears memory)."""
        from aire.models.base import run_sync

        self.state = AgentState()
        run_sync(self.memory.clear())

    def attach_session(self, path: str | Path | DurableSession) -> DurableSession:
        """Attach or create a durable session for checkpoint/resume."""
        self.session = path if isinstance(path, DurableSession) else DurableSession(path)
        return self.session

    @classmethod
    def from_session(
        cls,
        path: str | Path,
        model: Model,
        **kwargs: Any,
    ) -> Agent:
        """Rebuild an agent bound to an existing session file; hydrate state."""
        session = DurableSession(path)
        agent = cls(model, session=session, **kwargs)
        agent.state = session.to_agent_state()
        return agent

    # -- composition ------------------------------------------------------------------

    def as_tool(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        side_effect: str | Any = None,
    ) -> Tool:
        """Wrap this agent as a :class:`Tool` — the agent-as-tool building block.

        The returned tool takes ``task: str`` and returns the agent's final
        text, so any other agent (or :class:`~aire.agents.team.Team`) can
        delegate to it through the standard tool contract.
        """
        from aire.tools.types import SideEffect

        agent = self

        async def _delegate(task: str) -> str:
            result = await agent.run(task)
            return result.output

        _delegate.__doc__ = description or f"Delegate a task to the {agent.name} agent."
        return Tool(
            _delegate,
            name=name or agent.name,
            description=description,
            side_effect=SideEffect(side_effect) if side_effect else SideEffect.READ_ONLY,
        )

    # -- introspection --------------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "agent",
            "name": self.name,
            "model": self.model.info.ref,
            "tools": [t.spec.describe() for t in self.registry],
            "memory": self.memory.describe(),
            "config": self.config.model_dump(mode="json"),
            "skills": list(self._skills),
            "session": self.session.describe() if self.session else None,
            "policy": self.policy.describe() if self.policy is not None else None,
        }

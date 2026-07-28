"""The Agent: model + tools + memory + policy, with a fluent builder."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aire.agents.memory import Memory, resolve_memory
from aire.agents.runtime import AgentExecutor, Approver
from aire.agents.types import AgentConfig, AgentResult, AgentState
from aire.models.base import Model
from aire.tools.registry import ToolRegistry
from aire.tools.tool import Tool

if TYPE_CHECKING:
    from aire.core.runtime import Runtime


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
        name: str = "agent",
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
        self.state = AgentState()

    # -- execution -----------------------------------------------------------------

    async def run(self, input: str) -> AgentResult:
        """Run the agent to a terminal state and persist memory."""
        self.state = AgentState(input=input)
        executor = AgentExecutor(
            self.model,
            self.registry,
            config=self.config,
            memory=self.memory,
            runtime=self.runtime,
            approver=self.approver,
        )
        from aire.core.content import Message

        # Executor owns history assembly (recall + current user turn once).
        result = await executor.run(input, state=self.state)
        await self.memory.add(Message.text("user", input))
        for message in self.state.messages:
            if message.role == "tool":
                await self.memory.add(message)
        if result.output:
            await self.memory.add(Message.text("assistant", result.output))
        return result

    async def ask(self, input: str) -> str:
        """Convenience: run and return just the final text."""
        return (await self.run(input)).output

    def run_sync(self, input: str) -> AgentResult:
        from aire.models.base import run_sync

        return run_sync(self.run(input))

    def reset(self) -> None:
        """Start a fresh conversation (keeps tools and config, clears memory)."""
        from aire.models.base import run_sync

        self.state = AgentState()
        run_sync(self.memory.clear())

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
        }

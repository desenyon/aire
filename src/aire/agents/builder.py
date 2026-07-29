"""Fluent AgentBuilder — construct complex agents with one readable chain.

Example::

    agent = (
        AI.agents.builder("research")
        .model("mock:echo")
        .system("You are a careful researcher.")
        .tools(["http_get", "calculator"])
        .skill("research")
        .memory("buffer")
        .budget(max_steps=20, cost_usd=1.0)
        .planning(True)
        .build_sync()
    )
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aire.agents.types import AgentConfig
from aire.core.errors import ConfigurationError
from aire.models.base import Model, run_sync
from aire.tools.tool import Tool

if TYPE_CHECKING:
    from aire.agents.agent import Agent
    from aire.core.runtime import Runtime


class AgentBuilder:
    """Accumulate agent options, then materialize via :meth:`build` / :meth:`build_sync`."""

    def __init__(self, name: str = "agent", *, runtime: Runtime | None = None) -> None:
        self._name = name
        self._runtime = runtime
        self._model: str | Model | None = None
        self._tools: list[Tool | str] = []
        self._skills: list[str] = []
        self._memory: str | Any | None = None
        self._session: str | Path | Any | None = None
        self._system: str | None = None
        self._planning = False
        self._max_steps = 12
        self._token_budget: int | None = None
        self._cost_budget: float | None = None
        self._temperature: float | None = None
        self._permissions: list[str] = []
        self._approval_levels: list[str] = [
            "external_side_effect",
            "high_impact",
            "prohibited",
        ]
        self._approver: Any = None
        self._policy: Any = None
        self._builtins = False
        self._parallel_tools = False
        self._metadata: dict[str, Any] = {}

    def model(self, spec: str | Model) -> AgentBuilder:
        self._model = spec
        return self

    def system(self, prompt: str) -> AgentBuilder:
        self._system = prompt
        return self

    def tools(self, tools: list[Tool | str] | Tool | str | Sequence[Tool | str]) -> AgentBuilder:
        if isinstance(tools, (Tool, str)):
            self._tools.append(tools)
        else:
            self._tools.extend(list(tools))
        return self

    def skill(self, name: str) -> AgentBuilder:
        self._skills.append(name)
        return self

    def skills(self, names: list[str]) -> AgentBuilder:
        self._skills.extend(names)
        return self

    def memory(self, spec: str | Any) -> AgentBuilder:
        self._memory = spec
        return self

    def session(self, path: str | Path | Any) -> AgentBuilder:
        self._session = path
        return self

    def planning(self, enabled: bool = True) -> AgentBuilder:
        self._planning = enabled
        return self

    def budget(
        self,
        *,
        max_steps: int | None = None,
        tokens: int | None = None,
        cost_usd: float | None = None,
    ) -> AgentBuilder:
        if max_steps is not None:
            self._max_steps = max_steps
        if tokens is not None:
            self._token_budget = tokens
        if cost_usd is not None:
            self._cost_budget = cost_usd
        return self

    def temperature(self, value: float) -> AgentBuilder:
        self._temperature = value
        return self

    def permissions(self, *perms: str) -> AgentBuilder:
        self._permissions.extend(perms)
        return self

    def approval(self, *levels: str) -> AgentBuilder:
        self._approval_levels = list(levels)
        return self

    def approver(self, policy: Any) -> AgentBuilder:
        self._approver = policy
        return self

    def policy(self, engine: Any) -> AgentBuilder:
        """Attach a :class:`~aire.safety.policy.PolicyEngine` for tool gating."""
        self._policy = engine
        return self

    def builtins(self, enabled: bool = True) -> AgentBuilder:
        self._builtins = enabled
        return self

    def parallel_tools(self, enabled: bool = True) -> AgentBuilder:
        """Hint: enable concurrent tool execution when the executor supports it."""
        self._parallel_tools = enabled
        return self

    def meta(self, **kwargs: Any) -> AgentBuilder:
        self._metadata.update(kwargs)
        return self

    def config(self) -> AgentConfig:
        return AgentConfig(
            max_steps=self._max_steps,
            token_budget=self._token_budget,
            cost_budget_usd=self._cost_budget,
            planning=self._planning,
            system_prompt=self._system,
            approval_levels=list(self._approval_levels),
            permissions=list(self._permissions),
            temperature=self._temperature,
            parallel_tools=self._parallel_tools,
        )

    async def build(self) -> Agent:
        from aire.ai import AI

        if self._model is None:
            raise ConfigurationError(
                "AgentBuilder requires .model(...)",
                code="agents.builder_model",
            )
        agent = await AI.agents.create(
            self._model,
            tools=self._tools or None,
            memory=self._memory if isinstance(self._memory, str) else None,
            config=self.config(),
            approver=self._approver,
            name=self._name,
            builtins=self._builtins,
            skills=self._skills or None,
            session=self._session,
        )
        if self._policy is not None:
            agent.policy = self._policy
        if self._memory is not None and not isinstance(self._memory, str):
            agent.memory = self._memory
        agent._builder_meta = {  # type: ignore[attr-defined]
            **self._metadata,
            "parallel_tools": self._parallel_tools,
        }
        return agent

    def build_sync(self) -> Agent:
        return run_sync(self.build())

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "agent_builder",
            "name": self._name,
            "model": self._model if isinstance(self._model, str) else getattr(
                getattr(self._model, "info", None), "ref", type(self._model).__name__
            ),
            "tools": [t if isinstance(t, str) else t.spec.name for t in self._tools],
            "skills": list(self._skills),
            "config": self.config().model_dump(mode="json"),
            "parallel_tools": self._parallel_tools,
            "metadata": dict(self._metadata),
        }

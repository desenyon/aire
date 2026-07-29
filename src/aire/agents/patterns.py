"""Reusable agent patterns — research, coder, critic, planner, RAG."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from aire.agents.builder import AgentBuilder
from aire.core.errors import ConfigurationError


class AgentPattern(BaseModel):
    """Declarative defaults applied onto an :class:`AgentBuilder`."""

    name: str
    system: str
    tools: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    max_steps: int = 16
    planning: bool = False
    builtins: bool = False
    parallel_tools: bool = False
    permissions: list[str] = Field(default_factory=list)
    description: str = ""


_PATTERNS: dict[str, AgentPattern] = {
    "research": AgentPattern(
        name="research",
        system=(
            "You are a careful researcher. Gather evidence via web_search / http_get, "
            "cite sources when available, distinguish facts from speculation, and "
            "produce a structured summary."
        ),
        tools=["http_get", "http_post", "web_search", "calculator"],
        skills=["research"],
        max_steps=20,
        planning=True,
        builtins=True,
        description="Research agent with web_search/http builtins and planning",
    ),
    "coder": AgentPattern(
        name="coder",
        system=(
            "You are a senior software engineer. Write correct, typed, tested code. "
            "Prefer small diffs, explain trade-offs briefly, and never invent APIs. "
            "Use calculator, read_file, and list_files when helpful."
        ),
        # Builtin filesystem + calculator (code_toolkit adds extras when applied separately)
        tools=["calculator", "read_file", "list_files"],
        skills=["code"],
        max_steps=24,
        planning=True,
        builtins=True,
        parallel_tools=True,
        description="Coding agent with calculator + filesystem builtins and planning",
    ),
    "critic": AgentPattern(
        name="critic",
        system=(
            "You are a rigorous critic. Find flaws, risks, missing edge cases, and "
            "suggest concrete improvements. Be precise and adversarial but fair."
        ),
        max_steps=10,
        planning=False,
        description="Adversarial review agent",
    ),
    "planner": AgentPattern(
        name="planner",
        system=(
            "You are a planning specialist. Decompose goals into ordered steps with "
            "dependencies, risks, and success criteria. Do not execute tools unless asked."
        ),
        max_steps=8,
        planning=True,
        description="Goal decomposition planner",
    ),
    "rag": AgentPattern(
        name="rag",
        system=(
            "You answer using retrieved context only. If evidence is missing, say so. "
            "Quote short passages and keep answers grounded."
        ),
        skills=["research"],
        max_steps=12,
        planning=False,
        builtins=True,
        description="Retrieval-grounded answering agent (uses builtins for retrieval helpers)",
    ),
}


def catalog() -> dict[str, Any]:
    return {
        "kind": "agent_patterns",
        "patterns": {
            name: {
                "description": p.description,
                "tools": list(p.tools),
                "skills": list(p.skills),
                "max_steps": p.max_steps,
                "planning": p.planning,
            }
            for name, p in sorted(_PATTERNS.items())
        },
    }


def get_pattern(name: str) -> AgentPattern:
    if name not in _PATTERNS:
        raise ConfigurationError(
            f"unknown agent pattern {name!r}",
            code="agents.pattern_unknown",
            context={"available": sorted(_PATTERNS)},
        )
    return _PATTERNS[name]


def apply_pattern(builder: AgentBuilder, name: str) -> AgentBuilder:
    """Apply a named pattern onto an existing builder (fluent)."""
    pattern = get_pattern(name)
    builder.system(pattern.system)
    if pattern.tools:
        builder.tools(list(pattern.tools))
    if pattern.skills:
        builder.skills(pattern.skills)
    builder.budget(max_steps=pattern.max_steps)
    builder.planning(pattern.planning)
    builder.builtins(pattern.builtins)
    builder.parallel_tools(pattern.parallel_tools)
    if pattern.permissions:
        builder.permissions(*pattern.permissions)
    builder.meta(pattern=pattern.name)
    return builder


def pattern_builder(name: str, agent_name: str | None = None, **options: Any) -> AgentBuilder:
    """Start an :class:`AgentBuilder` preloaded with a pattern."""
    pattern = get_pattern(name)
    builder = AgentBuilder(agent_name or pattern.name, **options)
    return apply_pattern(builder, name)


def research_agent(**options: Any) -> AgentBuilder:
    return pattern_builder("research", **options)


def coder_agent(**options: Any) -> AgentBuilder:
    return pattern_builder("coder", **options)


def critic_agent(**options: Any) -> AgentBuilder:
    return pattern_builder("critic", **options)


def planner_agent(**options: Any) -> AgentBuilder:
    return pattern_builder("planner", **options)


def rag_agent(**options: Any) -> AgentBuilder:
    return pattern_builder("rag", **options)

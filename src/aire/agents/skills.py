"""Installable skill packs: tools + prompts + evals bundled for agents."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from aire.core.errors import ConfigurationError, NotFoundError
from aire.tools.tool import Tool


class Skill(BaseModel):
    """A named capability pack agents can load."""

    name: str
    description: str = ""
    prompts: dict[str, str] = Field(default_factory=dict)
    tool_names: list[str] = Field(default_factory=list)
    evals: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SkillRegistry:
    """Process-local registry of skills (+ filesystem load)."""

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}
        self._tools: dict[str, list[Tool]] = {}

    def register(
        self,
        skill: Skill,
        *,
        tools: list[Tool] | None = None,
        replace: bool = False,
    ) -> Skill:
        if skill.name in self._skills and not replace:
            raise ConfigurationError(
                f"skill {skill.name!r} already registered", code="skill.duplicate"
            )
        self._skills[skill.name] = skill
        self._tools[skill.name] = list(tools or [])
        return skill

    def get(self, name: str) -> Skill:
        if name not in self._skills:
            raise NotFoundError("skill", name, context={"available": sorted(self._skills)})
        return self._skills[name]

    def tools_for(self, name: str) -> list[Tool]:
        self.get(name)
        return list(self._tools.get(name, []))

    def names(self) -> list[str]:
        return sorted(self._skills)

    def load_dir(self, path: str | Path) -> list[Skill]:
        """Load ``*/skill.json`` packs from a directory."""
        root = Path(path)
        loaded: list[Skill] = []
        if not root.is_dir():
            raise ConfigurationError(f"skills dir not found: {root}", code="skill.dir")
        for skill_json in sorted(root.glob("*/skill.json")):
            payload = json.loads(skill_json.read_text())
            skill = Skill.model_validate(payload)
            self.register(skill, replace=True)
            loaded.append(skill)
        return loaded

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "skills",
            "skills": [
                {"name": s.name, "description": s.description, "tools": s.tool_names}
                for s in self._skills.values()
            ],
        }


_default_skills: SkillRegistry | None = None


def default_skills() -> SkillRegistry:
    global _default_skills
    if _default_skills is None:
        _default_skills = SkillRegistry()
        _register_builtins(_default_skills)
    return _default_skills


def skill(
    name: str, *, description: str = ""
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator: register a function-backed skill prompt helper."""

    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        default_skills().register(
            Skill(
                name=name,
                description=description or (fn.__doc__ or ""),
                prompts={"main": fn.__doc__ or ""},
            ),
            replace=True,
        )
        return fn

    return deco


def _register_builtins(reg: SkillRegistry) -> None:
    reg.register(
        Skill(
            name="research",
            description="Search and summarize sources",
            prompts={
                "main": "Research the topic thoroughly. Cite sources. Be concise.",
            },
            tool_names=["web_search", "fetch_url"],
        ),
        replace=True,
    )
    reg.register(
        Skill(
            name="code",
            description="Write and explain code",
            prompts={"main": "Write correct, minimal code. Explain briefly."},
            tool_names=["python_eval"],
        ),
        replace=True,
    )

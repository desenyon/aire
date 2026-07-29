"""Installable skill packs: tools + prompts + evals bundled for agents."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from aire.core.errors import ConfigurationError, NotFoundError
from aire.tools.tool import Tool

if TYPE_CHECKING:
    from aire.agents.agent import Agent


# Map skill tool aliases → builtin tool names when packs reference legacy names.
_TOOL_ALIASES: dict[str, str] = {
    "fetch_url": "http_get",
    "web_search": "web_search",
    "search": "web_search",
    "python_eval": "calculator",  # no arbitrary exec; arithmetic only
}


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
        """Load ``*/skill.json`` packs from a directory.

        Also picks up sibling ``tools.py`` modules (optional) and inline
        ``tools`` definitions from skill.json when present.
        """
        root = Path(path)
        loaded: list[Skill] = []
        if not root.is_dir():
            raise ConfigurationError(f"skills dir not found: {root}", code="skill.dir")
        for skill_json in sorted(root.glob("*/skill.json")):
            payload = json.loads(skill_json.read_text())
            skill = Skill.model_validate(
                {k: v for k, v in payload.items() if k != "tools"}
            )
            pack_tools = _load_skill_tools(skill_json.parent, payload)
            self.register(skill, tools=pack_tools or None, replace=True)
            loaded.append(skill)
        return loaded

    def resolve_tools(self, name: str, *, builtins: bool = True) -> list[Tool]:
        """Return registered skill tools plus builtins matching ``tool_names``."""
        skill = self.get(name)
        found: dict[str, Tool] = {t.spec.name: t for t in self.tools_for(name)}
        if builtins:
            from aire.tools.builtins import builtin_tools

            by_name = {t.spec.name: t for t in builtin_tools()}
            for wanted in skill.tool_names:
                key = _TOOL_ALIASES.get(wanted, wanted)
                if key in by_name and key not in found:
                    found[key] = by_name[key]
        return list(found.values())

    def apply(self, agent: Agent, name: str, *, builtins: bool = True) -> Agent:
        """Bind skill tools + system prompt onto an existing agent (mutates in place)."""
        skill = self.get(name)
        for tool in self.resolve_tools(name, builtins=builtins):
            if not agent.registry.has(tool.spec.name):
                agent.registry.register(tool)
        prompt = skill.prompts.get("main") or skill.description
        if prompt:
            existing = agent.config.system_prompt or ""
            block = f"[skill:{skill.name}] {prompt}"
            if block not in existing:
                agent.config.system_prompt = f"{existing}\n{block}".strip() if existing else block
        skills = list(getattr(agent, "_skills", []))
        if skill.name not in skills:
            skills.append(skill.name)
        agent._skills = skills
        return agent

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "skills",
            "skills": [
                {"name": s.name, "description": s.description, "tools": s.tool_names}
                for s in self._skills.values()
            ],
            "methods": ["register", "get", "tools_for", "resolve_tools", "apply", "load_dir"],
        }


_default_skills: SkillRegistry | None = None


def default_skills() -> SkillRegistry:
    global _default_skills
    if _default_skills is None:
        _default_skills = SkillRegistry()
        _register_builtins(_default_skills)
    return _default_skills


def apply_skill(agent: Agent, name: str, *, builtins: bool = True) -> Agent:
    """Apply a named skill from the default registry onto ``agent``."""
    return default_skills().apply(agent, name, builtins=builtins)


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


def _load_skill_tools(skill_dir: Path, payload: dict[str, Any]) -> list[Tool]:  # noqa: C901
    """Load tools from sibling tools.py and/or inline skill.json ``tools`` entries."""
    found: list[Tool] = []
    tools_py = skill_dir / "tools.py"
    if tools_py.is_file():
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            f"aire_skill_{skill_dir.name}_tools", tools_py
        )
        if spec is not None and spec.loader is not None:
            module = importlib.util.module_from_spec(spec)
            loaded_ok = False
            try:
                spec.loader.exec_module(module)
                loaded_ok = True
            except Exception:
                loaded_ok = False
            if loaded_ok:
                for attr in ("TOOLS", "tools", "get_tools"):
                    value = getattr(module, attr, None)
                    if callable(value) and attr == "get_tools":
                        try:
                            value = value()
                        except Exception:
                            value = None
                    if isinstance(value, list):
                        found.extend(t for t in value if isinstance(t, Tool))
                        break

    inline = payload.get("tools")
    if isinstance(inline, list):
        for item in inline:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if not name:
                continue
            # Light support: register a stub that returns the description/body text.
            body = str(item.get("body") or item.get("description") or "")
            desc = str(item.get("description") or f"Skill-defined tool {name}")

            def _make(n: str, b: str, d: str) -> Tool:
                def _fn() -> str:
                    return b

                _fn.__doc__ = d
                return Tool(_fn, name=n)

            found.append(_make(str(name), body, desc))
    return found


def _register_builtins(reg: SkillRegistry) -> None:
    reg.register(
        Skill(
            name="research",
            description="Search and summarize sources via web_search + http_get",
            prompts={
                "main": "Research the topic thoroughly. Cite sources. Be concise.",
            },
            tool_names=["web_search", "http_get", "fetch_url"],
        ),
        replace=True,
    )
    reg.register(
        Skill(
            name="code",
            description="Write and explain code using calculator + filesystem builtins",
            prompts={"main": "Write correct, minimal code. Explain briefly."},
            tool_names=["calculator", "read_file", "list_files"],
        ),
        replace=True,
    )

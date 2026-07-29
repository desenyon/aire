"""Curated recipes: one-call scaffolds for common aire patterns."""

from __future__ import annotations

from typing import Any

from aire.core.errors import ConfigurationError


def recipe(name: str, **options: Any) -> Any:
    """Build a ready-made stack: ``rag`` | ``agent`` | ``finetune`` | ``gateway``.

    Pass ``execute=True`` to run setup steps where defined (e.g. RAG indexing).
    """
    execute = bool(options.pop("execute", False))
    key = name.strip().lower()
    if key == "rag":
        return rag_recipe(execute=execute, **options)
    if key == "agent":
        return agent_recipe(**options)
    if key in {"finetune", "fine_tune", "lora"}:
        return finetune_recipe(**options)
    if key == "gateway":
        return gateway_recipe(**options)
    raise ConfigurationError(
        f"unknown recipe {name!r}",
        code="recipes.unknown",
        context={"available": sorted(RECIPES)},
    )


def rag_recipe(
    *,
    store: str = "local:default",
    documents: str | None = None,
    execute: bool = False,
    **options: Any,
) -> dict[str, Any]:
    """Knowledge pipeline + optional document path.

    When ``execute=True`` and ``documents`` is set, indexes immediately.
    """
    from aire.ai import AI
    from aire.models.base import run_sync

    knowledge = AI.rag.create(**options)
    indexed = False
    index_report = None
    if execute and documents:
        index_report = run_sync(knowledge.ingest(documents))
        indexed = True
    return {
        "kind": "rag",
        "knowledge": knowledge,
        "documents": documents,
        "store": store,
        "execute": execute,
        "indexed": indexed,
        "index_report": (
            index_report.model_dump(mode="json")
            if index_report is not None and hasattr(index_report, "model_dump")
            else index_report
        ),
        "next": "knowledge.index(documents) then knowledge.ask(question)"
        if not indexed
        else "knowledge.ask(question)",
    }


def agent_recipe(
    *,
    model: str = "mock:echo",
    builtins: bool = True,
    skills: list[str] | None = None,
    **options: Any,
) -> dict[str, Any]:
    from aire.agents.skills import apply_skill, default_skills
    from aire.ai import AI

    agent = AI.agents.create_sync(model, builtins=builtins, **options)
    loaded_skills = []
    if skills:
        for name in skills:
            apply_skill(agent, name, builtins=True)
            loaded_skills.append(default_skills().get(name))
    return {
        "kind": "agent",
        "agent": agent,
        "skills": loaded_skills,
        "model": model,
        "next": "agent.run(prompt)",
    }


def finetune_recipe(
    *,
    model_name: str = "gpt2",
    backend: str = "lora",
    **options: Any,
) -> dict[str, Any]:
    from aire.training.hpo import SearchSpace
    from aire.training.hpo import describe as hpo_describe
    from aire.training.lm_trainer import LMTrainer
    from aire.training.lora import create_lora

    if backend == "lora":
        options.setdefault("dry_run", False)
        trainer: Any = create_lora(model_name, **options)
        next_step = (
            "trainer.fit(dataset, dry_run=True) for CI; "
            "trainer.fit(dataset) for real PEFT (requires aire[peft])"
        )
    elif backend == "lm":
        trainer = LMTrainer(**options)
        next_step = "await trainer.fit(dataset)"
    else:
        raise ConfigurationError(
            f"unknown finetune backend {backend!r}",
            code="recipes.finetune_backend",
            context={"available": ["lora", "lm"]},
        )
    return {
        "kind": "finetune",
        "backend": backend,
        "trainer": trainer,
        "hpo": hpo_describe(),
        "search_space": SearchSpace(
            discrete={"epochs": [1, 2, 3]},
            log_continuous={"learning_rate": (1e-5, 1e-3)},
        ),
        "next": next_step,
        "describe": trainer.describe() if hasattr(trainer, "describe") else {},
    }


def gateway_recipe(**options: Any) -> dict[str, Any]:
    from aire.ai import AI

    models = options.pop("models", None) or ["mock:echo"]
    app = AI.gateway.create(models=models, **options)
    return {
        "kind": "gateway",
        "app": app,
        "models": models,
        "next": "uvicorn.run(app) or AI.gateway.serve()",
    }


RECIPES = {
    "rag": rag_recipe,
    "agent": agent_recipe,
    "finetune": finetune_recipe,
    "gateway": gateway_recipe,
}


def describe() -> dict[str, Any]:
    return {
        "kind": "recipes",
        "available": sorted(RECIPES),
        "usage": 'AI.recipe("rag"|"agent"|"finetune"|"gateway")',
    }

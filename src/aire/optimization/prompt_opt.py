"""Evaluation-guided prompt optimization loop (offline-capable)."""

from __future__ import annotations

import random
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, Field

from aire.models.base import Model


class PromptCandidate(BaseModel):
    prompt: str
    score: float = 0.0
    generation: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class PromptOptResult(BaseModel):
    best_prompt: str
    best_score: float
    history: list[PromptCandidate] = Field(default_factory=list)
    generations: int = 0

    def describe(self) -> dict[str, Any]:
        return {
            "best_prompt": self.best_prompt,
            "best_score": self.best_score,
            "candidates": len(self.history),
            "generations": self.generations,
        }


EvalFn = Callable[[str], float | Awaitable[float]]


_MUTATORS = [
    lambda p: p + " Be concise.",
    lambda p: p + " Cite evidence.",
    lambda p: "Think step by step.\n" + p,
    lambda p: p.replace("Answer", "Respond carefully").replace("answer", "respond carefully"),
    lambda p: p + " If unsure, say you don't know.",
]


async def optimize_prompt(
    seed_prompt: str,
    evaluate: EvalFn,
    *,
    generations: int = 3,
    population: int = 4,
    seed: int = 0,
    mutator: Callable[[str], str] | None = None,
    model: Model | None = None,
) -> PromptOptResult:
    """Hill-climb / mutate prompts; keep the highest-scoring variant.

    ``evaluate(prompt) -> score`` should return a higher-is-better metric
    (e.g. eval accuracy). Optional ``model`` can propose mutations via generation.
    """
    import inspect

    rng = random.Random(seed)
    history: list[PromptCandidate] = []

    async def score(prompt: str, gen: int) -> PromptCandidate:
        value = evaluate(prompt)
        if inspect.isawaitable(value):
            value = await value
        cand = PromptCandidate(prompt=prompt, score=float(value), generation=gen)
        history.append(cand)
        return cand

    best = await score(seed_prompt, 0)
    current = [best]

    for gen in range(1, generations + 1):
        next_gen: list[PromptCandidate] = []
        parents = sorted(current, key=lambda c: c.score, reverse=True)[: max(1, population // 2)]
        while len(next_gen) < population:
            parent = rng.choice(parents)
            child_prompt = await _mutate(parent.prompt, rng, mutator=mutator, model=model)
            next_gen.append(await score(child_prompt, gen))
        current = next_gen
        gen_best = max(current, key=lambda c: c.score)
        if gen_best.score > best.score:
            best = gen_best

    return PromptOptResult(
        best_prompt=best.prompt,
        best_score=best.score,
        history=history,
        generations=generations,
    )


async def _mutate(
    prompt: str,
    rng: random.Random,
    *,
    mutator: Callable[[str], str] | None,
    model: Model | None,
) -> str:
    if mutator is not None:
        return mutator(prompt)
    if model is not None and rng.random() < 0.5:
        try:
            suggestion = await model.ask(
                "Rewrite this system prompt to improve task accuracy. "
                f"Return only the new prompt.\n\n{prompt}"
            )
            text = str(suggestion).strip()
            if text:
                return text
        except Exception:
            pass
    return str(rng.choice(_MUTATORS)(prompt))


def describe() -> dict[str, Any]:
    return {
        "kind": "prompt_optimization",
        "strategies": ["mutation_hillclimb", "model_rewrite"],
        "offline": True,
    }

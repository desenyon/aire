"""Model routing: pick the best candidate model for each request.

Routing scores candidates from their normalized :class:`ModelInfo` metadata —
cost, latency, context capacity and capabilities — plus optional historical
performance fed back via :meth:`ModelRouter.record`.
"""

from __future__ import annotations

import time
from typing import Literal

from pydantic import BaseModel, Field

from aire.core.errors import ConfigurationError, ContextLengthError, NotFoundError
from aire.core.types import Capability, HealthStatus, Manifest
from aire.models.base import Model, estimate_tokens
from aire.models.types import GenerationRequest, GenerationResult, ModelInfo

Objective = Literal[
    "lowest_cost",
    "lowest_latency",
    "highest_quality",
    "quality_under_budget",
    "balanced",
]

# Relative quality priors by provider (tunable; history overrides over time).
_QUALITY_PRIOR: dict[str, float] = {
    "openai": 0.9,
    "anthropic": 0.9,
    "huggingface": 0.7,
    "ollama": 0.6,
    "mock": 0.1,
    "callable": 0.5,
    "local": 0.5,
}


class RouteDecision(BaseModel):
    """Why a model was chosen — fully inspectable by agents."""

    chosen: str
    scores: dict[str, float] = Field(default_factory=dict)
    objective: str
    reason: str = ""


class RoutingStats(BaseModel):
    calls: int = 0
    successes: int = 0
    total_latency_ms: float = 0.0
    total_cost_usd: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.successes / self.calls if self.calls else 1.0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.calls if self.calls else 0.0


class ModelRouter(Model):
    """A Model that delegates to the best-scoring candidate per request."""

    def __init__(
        self,
        candidates: list[Model],
        *,
        objective: Objective = "balanced",
        cost_limit_usd: float | None = None,
        latency_target_ms: float | None = None,
        fallback: bool = True,
    ) -> None:
        if not candidates:
            raise ConfigurationError("router requires at least one candidate model")
        self.candidates = candidates
        self.objective = objective
        self.cost_limit_usd = cost_limit_usd
        self.latency_target_ms = latency_target_ms
        self.fallback = fallback
        self.history: dict[str, RoutingStats] = {m.info.ref: RoutingStats() for m in candidates}

    @property
    def info(self) -> ModelInfo:
        first = self.candidates[0].info
        return ModelInfo(
            ref="router:" + "+".join(m.info.ref for m in self.candidates),
            provider="router",
            capabilities=list(first.capabilities),
            context_window=max((m.info.context_window or 0) for m in self.candidates) or None,
        )

    # -- routing ---------------------------------------------------------------------

    def score(self, model: Model, request: GenerationRequest) -> float:
        """Score a candidate in [0, 1]; higher is better for the objective."""
        info = model.info
        stats = self.history.get(info.ref, RoutingStats())
        estimated_tokens = sum(estimate_tokens(m.text_content) for m in request.messages)
        if info.context_window and estimated_tokens > info.context_window:
            return -1.0  # cannot serve this request
        cost_rate = (info.cost.input_per_million or 0.0) + (info.cost.output_per_million or 0.0)
        cost_score = 1.0 / (1.0 + cost_rate)
        latency_ms = stats.avg_latency_ms or info.latency_ms_p50 or 500.0
        latency_score = 1.0 / (1.0 + latency_ms / 1000.0)
        quality = _QUALITY_PRIOR.get(info.provider, 0.5) * (0.5 + 0.5 * stats.success_rate)
        if request.tools and not info.supports(Capability.TOOL_CALLING):
            return -1.0
        if request.response_format and not info.supports(Capability.STRUCTURED_OUTPUT):
            return -1.0
        objective = self.objective
        if objective == "lowest_cost":
            return cost_score
        if objective == "lowest_latency":
            return latency_score
        if objective == "highest_quality":
            return quality
        if objective == "quality_under_budget":
            if self.cost_limit_usd is not None:
                estimated_cost = cost_rate * max(estimated_tokens, 1) / 1_000_000
                if estimated_cost > self.cost_limit_usd:
                    return -1.0
            return quality
        return 0.4 * quality + 0.3 * cost_score + 0.3 * latency_score

    def route(self, request: GenerationRequest) -> RouteDecision:
        """Decide which candidate would serve this request, with full reasoning."""
        scores = {m.info.ref: round(self.score(m, request), 4) for m in self.candidates}
        eligible = {ref: s for ref, s in scores.items() if s >= 0}
        if not eligible:
            raise NotFoundError(
                "router candidate",
                f"objective={self.objective}",
                context={"scores": scores, "objective": self.objective},
            )
        chosen = max(eligible, key=lambda ref: eligible[ref])
        return RouteDecision(
            chosen=chosen,
            scores=scores,
            objective=self.objective,
            reason=f"best score for objective '{self.objective}'",
        )

    # -- model interface --------------------------------------------------------------

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        decision = self.route(request)
        order = sorted(
            (m for m in self.candidates if decision.scores.get(m.info.ref, -1) >= 0),
            key=lambda m: decision.scores[m.info.ref],
            reverse=True,
        )
        last_error: BaseException | None = None
        for model in order if self.fallback else order[:1]:
            started = time.perf_counter()
            try:
                result = await model.generate(request)
            except Exception as exc:
                self._record(model.info.ref, False, (time.perf_counter() - started) * 1000.0, 0.0)
                last_error = exc
                continue
            self._record(
                model.info.ref,
                True,
                (time.perf_counter() - started) * 1000.0,
                result.usage.cost_usd,
            )
            return result
        assert last_error is not None
        raise last_error

    def _record(self, ref: str, success: bool, latency_ms: float, cost: float) -> None:
        stats = self.history.setdefault(ref, RoutingStats())
        stats.calls += 1
        stats.successes += int(success)
        stats.total_latency_ms += latency_ms
        stats.total_cost_usd += cost

    async def health(self) -> HealthStatus:
        unhealthy = []
        for model in self.candidates:
            status = await model.health()
            if not status.ok:
                unhealthy.append(f"{model.info.ref}: {status.detail}")
        if len(unhealthy) == len(self.candidates):
            return HealthStatus.unhealthy("all candidates unhealthy: " + "; ".join(unhealthy))
        return HealthStatus.healthy(
            f"{len(self.candidates) - len(unhealthy)}/{len(self.candidates)} candidates healthy"
        )

    def describe(self) -> Manifest:
        base = super().describe()
        return base.model_copy(
            update={
                "extra": {
                    **base.extra,
                    "objective": self.objective,
                    "candidates": [m.info.ref for m in self.candidates],
                    "history": {k: v.model_dump() for k, v in self.history.items()},
                }
            }
        )


def estimate_context_tokens(request: GenerationRequest) -> int:
    """Public helper: rough token count of a request's messages."""
    total = sum(estimate_tokens(m.text_content) for m in request.messages)
    return total


def assert_fits(model: Model, request: GenerationRequest) -> None:
    """Raise ContextLengthError when the request exceeds the model's window."""
    window = model.info.context_window
    if window is not None and estimate_context_tokens(request) > window:
        raise ContextLengthError(
            f"request exceeds context window of {model.info.ref} ({window})",
            context={"model": model.info.ref, "window": window},
        )

"""Cost-optimization policies for model routing."""

from __future__ import annotations

import time

from pydantic import BaseModel, Field


def _today() -> str:
    return time.strftime("%Y-%m-%d", time.gmtime())


class CostPolicy(BaseModel):
    """Budget and preference knobs applied by :class:`ModelRouter`.

    Policies are inspectable by agents (``policy.model_dump()``) and compose
    with routing objectives — they filter/penalize candidates rather than
    replacing ``objective=``. Daily counters auto-reset when the UTC date changes.
    """

    # Hard caps (None = unlimited)
    max_cost_per_request_usd: float | None = None
    daily_budget_usd: float | None = None
    # Soft preferences
    prefer_cheaper_within: float = 0.05
    """If a cheaper candidate scores within this margin of the top score, pick it."""
    escalate_on_failure: bool = True
    """Keep fallback chain when the primary fails (router.fallback already does this)."""
    # Runtime accounting (mutated via record_spend)
    spent_today_usd: float = 0.0
    requests_today: int = 0
    spend_day: str = Field(default_factory=_today)

    def _ensure_day(self) -> None:
        today = _today()
        if self.spend_day != today:
            self.reset_daily()

    def remaining_budget(self) -> float | None:
        self._ensure_day()
        if self.daily_budget_usd is None:
            return None
        return max(0.0, self.daily_budget_usd - self.spent_today_usd)

    def allows_estimated_cost(self, estimated_usd: float) -> bool:
        self._ensure_day()
        max_per = self.max_cost_per_request_usd
        if max_per is not None and estimated_usd > max_per:
            return False
        remaining = self.remaining_budget()
        return not (remaining is not None and estimated_usd > remaining)

    def record_spend(self, cost_usd: float) -> None:
        self._ensure_day()
        self.spent_today_usd += max(0.0, cost_usd)
        self.requests_today += 1

    def reset_daily(self) -> None:
        self.spent_today_usd = 0.0
        self.requests_today = 0
        self.spend_day = _today()


class CostPolicyState(BaseModel):
    """Read-only snapshot of policy spend for manifests / gateway headers."""

    spent_today_usd: float = 0.0
    requests_today: int = 0
    remaining_budget_usd: float | None = None
    max_cost_per_request_usd: float | None = None
    daily_budget_usd: float | None = None

    @classmethod
    def from_policy(cls, policy: CostPolicy) -> CostPolicyState:
        policy._ensure_day()
        return cls(
            spent_today_usd=policy.spent_today_usd,
            requests_today=policy.requests_today,
            remaining_budget_usd=policy.remaining_budget(),
            max_cost_per_request_usd=policy.max_cost_per_request_usd,
            daily_budget_usd=policy.daily_budget_usd,
        )


def estimate_request_cost_usd(cost_rate_per_million: float, tokens: int) -> float:
    return cost_rate_per_million * max(tokens, 1) / 1_000_000.0


__all__ = [
    "CostPolicy",
    "CostPolicyState",
    "estimate_request_cost_usd",
]

"""Optimization: caching, routing, and cost/latency controls."""

from aire.optimization.cache import CachedModel, SemanticCachedModel, cache_key
from aire.optimization.cost_policy import CostPolicy, CostPolicyState
from aire.optimization.router import ModelRouter, RouteDecision, RoutingStats, assert_fits

__all__ = [
    "CachedModel",
    "CostPolicy",
    "CostPolicyState",
    "ModelRouter",
    "RouteDecision",
    "RoutingStats",
    "SemanticCachedModel",
    "assert_fits",
    "cache_key",
]

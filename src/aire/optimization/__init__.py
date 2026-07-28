"""Optimization: caching, routing, and cost/latency controls."""

from aire.optimization.cache import CachedModel, SemanticCachedModel, cache_key
from aire.optimization.router import ModelRouter, RouteDecision, RoutingStats, assert_fits

__all__ = [
    "CachedModel",
    "ModelRouter",
    "RouteDecision",
    "RoutingStats",
    "SemanticCachedModel",
    "assert_fits",
    "cache_key",
]

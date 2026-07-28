"""Lightweight in-process metrics: counters and latency summaries."""

from __future__ import annotations

import threading
from typing import Any


class Metrics:
    """Thread-safe counters, gauges and latency observations."""

    def __init__(self) -> None:
        self._counters: dict[str, float] = {}
        self._gauges: dict[str, float] = {}
        self._latencies: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def increment(self, name: str, value: float = 1.0, **labels: Any) -> None:
        key = self._key(name, labels)
        with self._lock:
            self._counters[key] = self._counters.get(key, 0.0) + value

    def gauge(self, name: str, value: float, **labels: Any) -> None:
        with self._lock:
            self._gauges[self._key(name, labels)] = value

    def observe_latency(self, name: str, ms: float, **labels: Any) -> None:
        key = self._key(name, labels)
        with self._lock:
            self._latencies.setdefault(key, []).append(ms)

    def record_tokens(self, input_tokens: int, output_tokens: int, *, model: str) -> None:
        self.increment("aire.tokens.input", input_tokens, model=model)
        self.increment("aire.tokens.output", output_tokens, model=model)

    def record_cost(self, usd: float, *, model: str) -> None:
        self.increment("aire.cost.usd", usd, model=model)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            latencies = {}
            for key, values in self._latencies.items():
                ordered = sorted(values)
                n = len(ordered)
                latencies[key] = {
                    "count": n,
                    "p50": ordered[n // 2] if n else 0.0,
                    "p95": ordered[int(n * 0.95)] if n else 0.0,
                    "max": ordered[-1] if n else 0.0,
                }
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "latencies": latencies,
            }

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._latencies.clear()

    @staticmethod
    def _key(name: str, labels: dict[str, Any]) -> str:
        if not labels:
            return name
        suffix = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{suffix}}}"

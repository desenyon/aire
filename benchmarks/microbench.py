#!/usr/bin/env python3
"""Offline microbenchmarks for aire (no network, no GPU required).

Writes a JSON report suitable for CI artifacts / release notes.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from aire._version import __version__
from aire.evaluation.metrics import _bleu4
from aire.models.base import run_sync
from aire.rag.store import LocalVectorStore, tokenize
from aire.rag.types import Chunk
from aire.safety.guardrails import GuardrailChain


def _timeit(fn: Callable[[], Any], *, rounds: int = 50, warmup: int = 5) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    samples: list[float] = []
    for _ in range(rounds):
        started = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - started) * 1000.0)
    return {
        "rounds": float(rounds),
        "mean_ms": statistics.fmean(samples),
        "stdev_ms": statistics.pstdev(samples) if len(samples) > 1 else 0.0,
        "p50_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=None, help="Write JSON report path")
    parser.add_argument("--rounds", type=int, default=50)
    args = parser.parse_args()

    text = ("aire builds agents, rag pipelines, and gateways. " * 40).strip()
    ref = tokenize("the cat sat on the mat and watched the birds")
    hyp = tokenize("the cat sat on the mat watching birds")
    chain = GuardrailChain()
    store = LocalVectorStore()
    chunks = [
        Chunk(
            document_id="d",
            text=f"fact {i} about cats and dogs",
            index=i,
            embedding=[float(i % 7)] * 8,
        )
        for i in range(64)
    ]
    run_sync(store.upsert(chunks))

    results: dict[str, Any] = {
        "aire_version": __version__,
        "kind": "offline_microbench",
        "benches": {
            "tokenize": _timeit(lambda: tokenize(text), rounds=args.rounds),
            "bleu4": _timeit(lambda: _bleu4(hyp, ref), rounds=args.rounds),
            "guardrail_chain": _timeit(
                lambda: chain.apply("hello world, contact me at a@b.co", stage="input"),
                rounds=args.rounds,
            ),
            "local_vector_search": _timeit(
                lambda: run_sync(store.search([0.0] * 8, k=5)),
                rounds=max(10, args.rounds // 2),
            ),
        },
    }

    print(json.dumps(results, indent=2, sort_keys=True))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

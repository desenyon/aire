"""Model routing and caching — pick and reuse models by objective.

Run:  python examples/model_router/main.py

Routes generation requests across candidate models by a cost/latency/quality
objective, with an exact-match cache in front to skip repeat calls.
"""

from __future__ import annotations

import asyncio

from aire import AI


async def main() -> None:
    cheap = await AI.models.use("mock:echo")
    capable = await AI.models.use("callable:uppercase")

    router = AI.models.router([cheap, capable], objective="balanced")
    cached = AI.models.cache(router, max_entries=128)

    first = await cached.ask("route this request")
    second = await cached.ask("route this request")  # served from cache

    print("first answer: ", first)
    print("second answer:", second)
    print("cache hits:", cached.hits, "misses:", cached.misses)

    from aire.models.types import GenerationRequest

    decision = router.route(GenerationRequest.of("route this request"))
    print("routing decision:", decision.chosen, "-", decision.reason)
    print("scores:", decision.scores)


AI.models.register_callable("uppercase", lambda prompt: prompt.upper())
asyncio.run(main())

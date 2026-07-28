"""Graph workflow — branching, fan-in, streaming events, offline.

Run:  python examples/workflows/main.py

A three-stage research pipeline: search -> analyze -> verify, with a
conditional edge that skips verification for trivial inputs. Events stream
as nodes transition, and a JSON checkpoint is written after every node.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from aire import AI


async def main() -> None:
    wf = AI.workflows.create("research_pipeline")

    async def search(query: str, ctx: dict) -> list[str]:
        return [f"result about {query}", f"another note on {query}"]

    async def analyze(results: list[str], ctx: dict) -> dict:
        return {"findings": len(results), "trivial": len(results) < 1}

    async def verify(analysis: dict, ctx: dict) -> dict:
        return {**analysis, "verified": True}

    wf.add("search", search)
    wf.add("analyze", analyze)
    wf.add("verify", verify, retries=1)
    wf.connect("search", "analyze")
    wf.connect("analyze", "verify", when=lambda out: not out["trivial"])

    checkpoint = Path(tempfile.mkdtemp(prefix="aire-wf-")) / "checkpoint.json"
    wf.checkpoint_path = checkpoint  # persist state after every node

    result = await wf.run("aire plugins")
    print("\nworkflow ok:", result.ok)
    print("outputs:", result.outputs)
    print("checkpoint written:", checkpoint.exists())

    # Streaming variant: watch transitions live.
    async for event in wf.run_stream("aire plugins"):
        print(f"event: {event.kind:<20} node={event.node or '-'}")


asyncio.run(main())

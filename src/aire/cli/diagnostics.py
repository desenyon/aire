"""Offline-first installed-package diagnostics with explicit evidence states."""

from __future__ import annotations

import asyncio
import importlib.metadata
import importlib.util
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aire._version import __version__
from aire.core.config import Settings
from aire.core.errors import AireError
from aire.models.base import run_sync

if TYPE_CHECKING:
    from aire.core.runtime import Runtime
    from aire.models.base import EmbeddingModel, Model


class Diagnostics:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def add(self, name: str, ok: bool | None, detail: str = "", *, required: bool = True) -> None:
        self.checks.append(
            {
                "check": name,
                "status": "skipped" if ok is None else "passed" if ok else "failed",
                "ok": ok,
                "detail": detail,
                "required": required,
                "skipped": ok is None,
            }
        )

    async def probe(self, name: str, operation: Callable[[], Awaitable[str]]) -> None:
        try:
            self.add(name, True, await operation())
        except Exception as exc:
            # Provider/config errors can echo credentials or complete HTTP bodies.
            code = exc.code if isinstance(exc, AireError) else type(exc).__name__
            self.add(name, False, f"check failed ({code})")


async def _rag_smoke(runtime: Runtime, model: Model, embedder: EmbeddingModel) -> str:
    from aire.rag.pipeline import Knowledge
    from aire.rag.types import Document

    knowledge = Knowledge(runtime, embedder=embedder, model=model, guardrails=False)
    await knowledge.ingest(
        [
            Document(
                text="Aire diagnostics use a local fixture.",
                metadata={"source": "doctor-fixture"},
            )
        ]
    )
    answer = await knowledge.ask("What do Aire diagnostics use?")
    if not answer.text or (answer.retrieved, [c.source for c in answer.citations]) != (
        1,
        ["doctor-fixture"],
    ):
        raise RuntimeError("RAG mismatch")
    return "retrieved=1; source=doctor-fixture"


async def _smoke(checks: Diagnostics) -> None:
    from aire.core.runtime import Runtime
    from aire.models.builtin import EchoModel, HashingEmbedder
    from aire.tools.tool import Tool

    # Never resolve the user's configured provider or exporter for offline smoke.
    async with Runtime(Settings()) as runtime:
        model, embedder = EchoModel(), HashingEmbedder()

        async def generation() -> str:
            if await model.ask("doctor ping") != "doctor ping":
                raise RuntimeError("generation mismatch")
            return "mock:echo roundtrip"

        async def embedding() -> str:
            vector = await embedder.embed_one("doctor ping")
            if len(vector) != embedder.dimension or not any(vector):
                raise RuntimeError("embedding mismatch")
            return f"local:hashing dimension={len(vector)}"

        async def tool() -> str:
            def add(a: int, b: int) -> int:
                return a + b

            result = await Tool(add).execute({"a": 2, "b": 3})
            if not result.ok or result.output != 5:
                raise RuntimeError("tool mismatch")
            return "2 + 3 = 5"

        async def rag() -> str:
            return await _rag_smoke(runtime, model, embedder)

        for name, operation in (
            ("generation", generation),
            ("embedding", embedding),
            ("tool", tool),
            ("rag", rag),
        ):
            await checks.probe(f"smoke:{name}", operation)


async def _live(checks: Diagnostics, settings: Settings) -> None:
    import httpx

    from aire.core.runtime import Runtime
    from aire.models.registry import ModelRegistry

    async with Runtime(settings) as runtime:
        registry = ModelRegistry(runtime)
        for provider, variable, reference in (
            ("openai", "OPENAI_API_KEY", "openai:gpt-4o-mini"),
            ("anthropic", "ANTHROPIC_API_KEY", "anthropic:claude-sonnet-4-5"),
        ):
            if not settings.credential(provider).resolve_key(variable):
                checks.add(f"live:{reference}", None, "no credentials", required=False)
                continue

            async def health(ref: str = reference) -> str:
                # Bound the complete probe, including retries and connection time.
                async with asyncio.timeout(10):
                    status = await (await registry.use(ref)).health()
                if not status.ok:
                    raise RuntimeError("provider health failed")
                return "provider health passed"

            await checks.probe(f"live:{reference}", health)

        async def ollama() -> str:
            # Keep this probe local; do not infer a model-quality guarantee.
            response = await asyncio.to_thread(
                httpx.get, "http://127.0.0.1:11434/api/tags", timeout=2.0
            )
            response.raise_for_status()
            models = response.json().get("models", [])
            return f"models={len(models)}"

        await checks.probe("live:ollama", ollama)


def run_diagnostics(
    *, live: bool = False, smoke: bool = False, config: Path | None = None
) -> dict[str, Any]:
    checks = Diagnostics()
    checks.add("python>=3.11", sys.version_info >= (3, 11), sys.version.split()[0])
    for module, package in (
        ("pydantic", "pydantic"),
        ("httpx", "httpx"),
        ("yaml", "PyYAML"),
        ("typer", "typer"),
    ):
        present = importlib.util.find_spec(module) is not None
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = "metadata unavailable"
        checks.add(f"dependency:{module}", present, version)
    fastapi = importlib.util.find_spec("fastapi") is not None
    checks.add(
        "dependency:fastapi(optional)",
        True if fastapi else None,
        "needed for aire deploy/serve",
        required=False,
    )
    try:
        settings = Settings.load(config)
    except AireError as exc:
        settings = None
        checks.add("config:loaded", False, f"invalid configuration ({exc.code})")
    if settings is not None:
        checks.add("config:loaded", True, "configuration validated")
        for provider, variable in (
            ("openai", "OPENAI_API_KEY"),
            ("anthropic", "ANTHROPIC_API_KEY"),
            ("huggingface", "HF_TOKEN"),
        ):
            present = bool(settings.credential(provider).resolve_key(variable))
            checks.add(
                f"credentials:{provider}",
                True if present else None,
                "configured" if present else "not configured",
                required=False,
            )
    checks.add("provider:mock", True, "mock:echo available (not a quality assessment)")
    checks.add("embedder:local", True, "local:hashing available")

    async def probes() -> None:
        if smoke or live:
            await _smoke(checks)
        if live and settings is not None:
            await _live(checks, settings)
        elif live:
            checks.add("live:providers", None, "configuration invalid")

    run_sync(probes())
    return {
        "schema_version": 1,
        "version": __version__,
        "python": sys.version.split()[0],
        "checks": checks.checks,
        "healthy": not any(
            item["status"] == "failed" and item["required"] for item in checks.checks
        ),
        "live": live,
        "smoke": smoke or live,
        "skipped": sum(item["status"] == "skipped" for item in checks.checks),
    }

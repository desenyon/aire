"""Generate a production FastAPI application around any aire target.

The generated app includes health/readiness endpoints, a manifest endpoint
(machine-readable self-description), optional bearer auth, optional in-memory
rate limiting, and a task-specific invocation endpoint:

- ``Agent``    → POST /v1/run       {"input": "..."}
- ``Knowledge``→ POST /v1/ask       {"question": "..."}
- ``Model``    → POST /v1/generate  {"prompt": "..."}
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from aire.core.errors import AireError, ConfigurationError
from aire.models.base import Model

if TYPE_CHECKING:
    from fastapi import Depends, FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse
else:
    try:  # optional dependency: aire[serve]
        from fastapi import Depends, FastAPI, HTTPException, Request
        from fastapi.responses import JSONResponse
    except ImportError:  # pragma: no cover - exercised only without the extra
        Depends = FastAPI = HTTPException = Request = JSONResponse = None

if TYPE_CHECKING:
    from aire.agents.agent import Agent
    from aire.observability.metrics import Metrics
    from aire.rag.pipeline import Knowledge


def create_app(
    target: Agent | Knowledge | Model,
    *,
    title: str = "aire service",
    auth_token: str | None = None,
    rate_limit_per_minute: int | None = None,
    metrics: Metrics | None = None,
) -> Any:
    """Build a FastAPI app serving ``target``. Requires ``pip install aire[serve]``."""
    if FastAPI is None:
        raise ConfigurationError(
            "fastapi is required for deployment: pip install 'aire[serve]'",
            code="deploy.fastapi_missing",
        )

    app = FastAPI(title=title, version="0.1.0")
    guard = _make_guard(auth_token=auth_token, rate_limit_per_minute=rate_limit_per_minute)

    @app.exception_handler(AireError)
    async def _aire_error_handler(request: Request, exc: AireError) -> Any:
        return JSONResponse(status_code=422, content=exc.to_dict())

    _register_meta_routes(app, target, metrics)
    if isinstance(target, Model):
        _register_model_routes(app, target, guard, metrics)
    elif _is_knowledge(target):
        _register_knowledge_routes(app, target, guard, metrics)
    else:  # Agent
        _register_agent_routes(app, target, guard, metrics)
    return app


def _make_guard(*, auth_token: str | None, rate_limit_per_minute: int | None) -> Any:
    buckets: dict[str, list[float]] = defaultdict(list)

    async def _guard(request: Request) -> None:
        if auth_token is not None:
            header = request.headers.get("authorization", "")
            if header != f"Bearer {auth_token}":
                raise HTTPException(status_code=401, detail="invalid or missing bearer token")
        if rate_limit_per_minute is not None:
            key = request.client.host if request.client else "unknown"
            now = time.time()
            window = [t for t in buckets[key] if now - t < 60.0]
            if len(window) >= rate_limit_per_minute:
                raise HTTPException(status_code=429, detail="rate limit exceeded")
            window.append(now)
            buckets[key] = window

    return _guard


def _register_meta_routes(app: Any, target: Any, metrics: Metrics | None) -> None:
    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok"}

    @app.get("/ready")
    async def ready() -> dict[str, Any]:
        return {"status": "ready", "target": _target_name(target)}

    @app.get("/manifest")
    async def manifest() -> dict[str, Any]:
        describe = getattr(target, "describe", None)
        return describe() if callable(describe) else {"target": _target_name(target)}

    @app.get("/metrics")
    async def metrics_endpoint() -> dict[str, Any]:
        return metrics.snapshot() if metrics is not None else {"counters": {}, "latencies": {}}


def _register_model_routes(app: Any, target: Model, guard: Any, metrics: Metrics | None) -> None:
    @app.post("/v1/generate", dependencies=[Depends(guard)])
    async def generate(body: dict[str, Any]) -> dict[str, Any]:
        from aire.models.types import GenerationRequest

        prompt = body.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise HTTPException(status_code=400, detail="'prompt' (string) is required")
        request = GenerationRequest.of(
            prompt,
            temperature=body.get("temperature"),
            max_tokens=body.get("max_tokens"),
        )
        started = time.perf_counter()
        result = await target.generate(request)
        if metrics is not None:
            metrics.observe_latency("generate", (time.perf_counter() - started) * 1000.0)
            metrics.record_tokens(
                result.usage.input_tokens, result.usage.output_tokens, model=result.model
            )
            metrics.record_cost(result.usage.cost_usd, model=result.model)
        return {
            "text": result.text,
            "model": result.model,
            "usage": result.usage.model_dump(),
            "finish_reason": result.finish_reason,
        }


def _register_knowledge_routes(app: Any, target: Any, guard: Any, metrics: Metrics | None) -> None:
    @app.post("/v1/ask", dependencies=[Depends(guard)])
    async def ask(body: dict[str, Any]) -> dict[str, Any]:
        question = body.get("question")
        if not isinstance(question, str) or not question:
            raise HTTPException(status_code=400, detail="'question' (string) is required")
        started = time.perf_counter()
        answer = await target.ask(question, k=int(body.get("k", 5)))
        if metrics is not None:
            metrics.observe_latency("ask", (time.perf_counter() - started) * 1000.0)
            metrics.record_cost(answer.usage.cost_usd, model=answer.model)
        return {
            "answer": answer.text,
            "citations": [c.model_dump(mode="json") for c in answer.citations],
            "model": answer.model,
            "usage": answer.usage.model_dump(),
        }


def _register_agent_routes(app: Any, target: Any, guard: Any, metrics: Metrics | None) -> None:
    @app.post("/v1/run", dependencies=[Depends(guard)])
    async def run(body: dict[str, Any]) -> dict[str, Any]:
        input_text = body.get("input")
        if not isinstance(input_text, str) or not input_text:
            raise HTTPException(status_code=400, detail="'input' (string) is required")
        started = time.perf_counter()
        result = await target.run(input_text)
        if metrics is not None:
            metrics.observe_latency("run", (time.perf_counter() - started) * 1000.0)
            metrics.record_cost(result.usage.cost_usd, model=_target_name(target))
        return {
            "output": result.output,
            "status": str(result.status),
            "steps": len(result.steps),
            "usage": result.usage.model_dump(),
            "run_id": result.run_id,
        }


def _is_knowledge(target: Any) -> bool:
    from aire.rag.pipeline import Knowledge

    return isinstance(target, Knowledge)


def _target_name(target: Any) -> str:
    if isinstance(target, Model):
        return target.info.ref
    return getattr(target, "name", type(target).__name__)

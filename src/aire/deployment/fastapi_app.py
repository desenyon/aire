"""Generate a production FastAPI application around any aire target.

The generated app includes health/readiness endpoints, a manifest endpoint
(machine-readable self-description), optional bearer auth, optional in-memory
rate limiting, and a task-specific invocation endpoint:

- ``Agent``    → POST /v1/run       {"input": "..."}
- ``Knowledge``→ POST /v1/ask       {"question": "..."}
- ``Model``    → POST /v1/generate  {"prompt": "..."}
- ``Team``     → POST /v1/team/run  {"task": "..."}
"""

from __future__ import annotations

import asyncio
import inspect
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, ValidationError

from aire._version import __version__
from aire.core.errors import AireError, ConfigurationError
from aire.models.base import Model

if TYPE_CHECKING:
    from fastapi import Depends, FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse
else:
    try:  # optional dependency: aire-ai[serve]
        from fastapi import Depends, FastAPI, HTTPException, Request
        from fastapi.responses import JSONResponse
    except ImportError:  # pragma: no cover - exercised only without the extra
        Depends = FastAPI = HTTPException = Request = JSONResponse = None

if TYPE_CHECKING:
    from aire.agents.agent import Agent
    from aire.agents.team import Team
    from aire.observability.metrics import Metrics
    from aire.rag.pipeline import Knowledge


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    k: int = Field(default=5, ge=1, le=1000)


def create_app(
    target: Agent | Knowledge | Model | Team,
    *,
    title: str = "aire service",
    auth_token: str | None = None,
    rate_limit_per_minute: int | None = None,
    metrics: Metrics | None = None,
    close_target: bool = False,
    shared_agent_session: bool = False,
) -> Any:
    """Build a FastAPI app serving ``target``. Requires ``pip install aire-ai[serve]``."""
    if FastAPI is None:
        raise ConfigurationError(
            "fastapi is required for deployment: pip install 'aire-ai[serve]'",
            code="deploy.fastapi_missing",
        )

    @asynccontextmanager
    async def lifespan(app: Any) -> Any:
        yield
        if close_target:
            resource = getattr(target, "runtime", None) or target
            close = getattr(resource, "aclose", None)
            if close is not None:
                await close()

    app = FastAPI(title=title, version=__version__, lifespan=lifespan)

    @app.exception_handler(ValidationError)
    async def invalid_request(request: Request, exc: ValidationError) -> Any:
        return JSONResponse(status_code=422, content={"detail": "invalid generation parameters"})

    guard = _make_guard(auth_token=auth_token, rate_limit_per_minute=rate_limit_per_minute)

    @app.exception_handler(AireError)
    async def _aire_error_handler(request: Request, exc: AireError) -> Any:
        return JSONResponse(status_code=422, content=exc.to_dict())

    _register_meta_routes(app, target, metrics, guard)
    if _is_team(target):
        _register_team_routes(app, target, guard, metrics, shared_agent_session)
    elif isinstance(target, Model):
        _register_model_routes(app, target, guard, metrics)
    elif _is_knowledge(target):
        _register_knowledge_routes(app, target, guard, metrics)
    else:  # Agent
        _register_agent_routes(app, target, guard, metrics, shared_agent_session)
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


def _register_meta_routes(app: Any, target: Any, metrics: Metrics | None, guard: Any) -> None:
    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok"}

    @app.get("/ready")
    async def ready() -> Any:
        available = _runtime_present(target)
        runtime = getattr(target, "runtime", None)
        if runtime is not None and getattr(runtime, "_closed", False):
            available = False
        components = [target, getattr(target, "store", None), getattr(target, "model", None)]
        for component in components:
            probe = getattr(component, "health", None)
            if callable(probe):
                try:
                    result = probe()
                    if inspect.isawaitable(result):
                        result = await asyncio.wait_for(result, timeout=5)
                    available = available and bool(result.ok)
                except Exception:
                    available = False
        return JSONResponse(
            status_code=200 if available else 503,
            content={
                "status": "ready" if available else "degraded",
                "target": _target_name(target),
            },
        )

    @app.get("/manifest", dependencies=[Depends(guard)])
    async def manifest() -> dict[str, Any]:
        describe = getattr(target, "describe", None)
        return describe() if callable(describe) else {"target": _target_name(target)}

    @app.get("/metrics", dependencies=[Depends(guard)])
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
    async def ask(body: AskRequest) -> dict[str, Any]:
        question = body.question
        started = time.perf_counter()
        answer = await target.ask(question, k=body.k)
        if metrics is not None:
            metrics.observe_latency("ask", (time.perf_counter() - started) * 1000.0)
            metrics.record_cost(answer.usage.cost_usd, model=answer.model)
        return {
            "answer": answer.text,
            "citations": [c.model_dump(mode="json") for c in answer.citations],
            "model": answer.model,
            "usage": answer.usage.model_dump(),
        }


def _isolated_agent(target: Any) -> Any:
    from aire.agents.agent import Agent

    return Agent(
        target.model,
        tools=target.registry,
        config=target.config.model_copy(deep=True),
        runtime=target.runtime,
        approver=target.approver,
        policy=target.policy,
        name=target.name,
    )


def _register_agent_routes(
    app: Any, target: Any, guard: Any, metrics: Metrics | None, shared: bool
) -> None:
    lock = asyncio.Lock()

    @app.post("/v1/run", dependencies=[Depends(guard)])
    async def run(body: dict[str, Any]) -> dict[str, Any]:
        input_text = body.get("input")
        if not isinstance(input_text, str) or not input_text:
            raise HTTPException(status_code=400, detail="'input' (string) is required")
        started = time.perf_counter()
        if shared:
            async with lock:
                result = await target.run(input_text)
        else:
            result = await _isolated_agent(target).run(input_text)
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


def _register_team_routes(
    app: Any, target: Any, guard: Any, metrics: Metrics | None, shared: bool
) -> None:
    from aire.agents.team import Team

    lock = asyncio.Lock()

    @app.post("/v1/team/run", dependencies=[Depends(guard)])
    async def team_run(body: dict[str, Any]) -> dict[str, Any]:
        task = body.get("task") or body.get("input")
        if not isinstance(task, str) or not task:
            raise HTTPException(status_code=400, detail="'task' (string) is required")
        started = time.perf_counter()
        if shared:
            async with lock:
                result = await target.run(task)
        else:
            isolated = Team(
                {name: _isolated_agent(agent) for name, agent in target.members.items()},
                target.supervisor,
                max_rounds=target.max_rounds,
                prompt_template=target.prompt_template,
            )
            result = await isolated.run(task)
        if metrics is not None:
            metrics.observe_latency("team.run", (time.perf_counter() - started) * 1000.0)
            metrics.record_cost(result.usage.cost_usd, model=_target_name(target))
        return {
            "answer": result.answer,
            "rounds": result.rounds,
            "delegations": [d.model_dump(mode="json") for d in result.delegations],
            "usage": result.usage.model_dump(),
        }


def _is_knowledge(target: Any) -> bool:
    from aire.rag.pipeline import Knowledge

    return isinstance(target, Knowledge)


def _is_team(target: Any) -> bool:
    from aire.agents.team import Team

    return isinstance(target, Team)


def _runtime_present(target: Any) -> bool:
    """True when the target (or nested agent/knowledge) exposes a runtime."""
    if getattr(target, "runtime", None) is not None:
        return True
    if isinstance(target, Model):
        return True  # models are usable without a shared Runtime
    if _is_team(target):
        members = getattr(target, "members", {}) or {}
        return any(getattr(m, "runtime", None) is not None for m in members.values())
    return False


def _target_name(target: Any) -> str:
    if isinstance(target, Model):
        return target.info.ref
    if _is_team(target):
        return getattr(target, "name", None) or "team"
    return getattr(target, "name", type(target).__name__)

"""Model gateway: an OpenAI-compatible server in front of every provider.

One process exposes ``/v1/chat/completions``, ``/v1/embeddings`` and
``/v1/models`` and routes each request to any aire model reference — OpenAI,
Anthropic, Ollama, LM Studio, llama.cpp, vLLM, Groq, or a custom plugin — with
ordered fallbacks, round-robin or objective-based routing, bearer auth, rate
limiting, tracing and per-model metrics. Existing OpenAI clients point at the
gateway unchanged::

    app = create_gateway(
        models=["ollama:llama3.2"],
        aliases={"smart": ["anthropic:claude-sonnet-4-5", "openai:gpt-4o-mini"]},
        routing="first",          # or "round_robin", or objective="lowest_cost"
        auth_token="sk-internal",
    )

The request ``model`` field accepts an exposed alias, an exposed ref, or (when
``allow_direct_refs``) any ``provider:name`` reference directly.
"""

from __future__ import annotations

import contextlib
import json
import time
import uuid
from collections import defaultdict
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, Literal

from aire._version import __version__
from aire.core.errors import (
    AireError,
    AuthenticationError,
    ConfigurationError,
    ContextLengthError,
    NotFoundError,
    PermissionDeniedError,
    ProviderError,
    RateLimitError,
    SafetyError,
)
from aire.core.runtime import Runtime
from aire.deployment.fastapi_app import _make_guard
from aire.models.base import EmbeddingModel, Model
from aire.models.registry import ModelRegistry
from aire.models.types import (
    EmbeddingRequest,
    GenerationChunk,
    GenerationRequest,
    GenerationResult,
    StructuredOutputSpec,
    ToolDefinition,
)

if TYPE_CHECKING:
    from fastapi import Depends, FastAPI, Request
    from fastapi.responses import JSONResponse, Response, StreamingResponse
else:
    try:  # optional dependency: aire[serve]
        from fastapi import Depends, FastAPI, Request
        from fastapi.responses import JSONResponse, Response, StreamingResponse
    except ImportError:  # pragma: no cover - exercised only without the extra
        Depends = FastAPI = Request = JSONResponse = Response = StreamingResponse = None

if TYPE_CHECKING:
    from aire.observability.metrics import Metrics
    from aire.optimization.router import Objective
    from aire.safety.guardrails import Guardrail, GuardrailChain

RoutingMode = Literal["first", "round_robin"]


class Gateway:
    """Resolves public model names to candidate model refs and routes requests.

    Production guards: per-candidate circuit breakers (open after
    ``failure_threshold`` consecutive failures, half-open after
    ``cooldown_seconds``) and daily cost budgets keyed by alias or ref.
    """

    def __init__(
        self,
        runtime: Runtime,
        *,
        chat_routes: dict[str, list[str]] | None = None,
        embedding_routes: dict[str, list[str]] | None = None,
        routing: RoutingMode = "first",
        objective: Objective | None = None,
        fallback: bool = True,
        allow_direct_refs: bool = True,
        budgets: dict[str, float] | None = None,
        circuit_breaker: bool = True,
        failure_threshold: int = 3,
        cooldown_seconds: float = 30.0,
        semantic_cache: bool = False,
        semantic_threshold: float = 0.95,
        semantic_embedder: str | EmbeddingModel | None = None,
        semantic_cache_redis_url: str | None = None,
    ) -> None:
        if routing not in ("first", "round_robin"):
            raise ConfigurationError(
                f"unknown gateway routing mode: {routing!r}",
                code="config.gateway_routing",
                context={"routing": routing},
            )
        self._runtime = runtime
        self._models = ModelRegistry(runtime)
        self.chat_routes = chat_routes or {}
        self.embedding_routes = embedding_routes or {}
        self.routing = routing
        self.objective = objective
        self.fallback = fallback
        self.allow_direct_refs = allow_direct_refs
        self.budgets = budgets or {}
        self.circuit_breaker = circuit_breaker
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.semantic_cache = semantic_cache
        self.semantic_threshold = semantic_threshold
        self._semantic_embedder_spec = semantic_embedder
        self._semantic_embedder: EmbeddingModel | None = (
            semantic_embedder if isinstance(semantic_embedder, EmbeddingModel) else None
        )
        self._semantic_entries: list[tuple[str, list[float], str, GenerationResult]] = []
        self._semantic_redis_url = semantic_cache_redis_url
        self._semantic_redis_backend: Any = None
        self.semantic_hits = 0
        self.semantic_misses = 0
        self._round_robin: dict[str, int] = defaultdict(int)
        self._routers: dict[str, Model] = {}
        self._circuits: dict[str, dict[str, Any]] = {}
        self._spend: dict[str, tuple[str, float]] = {}

    # -- route resolution ----------------------------------------------------------

    def _refs_for(self, public: str, routes: dict[str, list[str]], kind: str) -> list[str]:
        refs = routes.get(public)
        if refs:
            return list(refs)
        if self.allow_direct_refs and ":" in public:
            return [public]
        raise NotFoundError(
            f"gateway {kind} model",
            public,
            context={
                "available": sorted(routes),
                "hint": "use an exposed alias, or pass a 'provider:name' ref",
            },
        )

    def _ordered(self, public: str, refs: list[str]) -> list[str]:
        ordered = refs
        if self.routing == "round_robin" and len(refs) > 1:
            start = self._round_robin[public] % len(refs)
            self._round_robin[public] += 1
            ordered = refs[start:] + refs[:start]
        available = [ref for ref in ordered if self._available(public, ref)]
        if not available:
            raise RateLimitError(
                "gateway",
                f"all candidates for {public!r} are unavailable "
                "(circuit open or daily budget exhausted)",
                status=429,
            )
        return available

    # -- circuit breakers & budgets -----------------------------------------------------

    def _available(self, public: str, ref: str) -> bool:
        if self.circuit_breaker:
            circuit = self._circuits.get(ref)
            if (
                circuit is not None
                and circuit["opened_at"] is not None
                and time.monotonic() - circuit["opened_at"] < self.cooldown_seconds
            ):
                return False
        return not (self._over_budget(public) or self._over_budget(ref))

    def _over_budget(self, key: str) -> bool:
        limit = self.budgets.get(key)
        if limit is None:
            return False
        day, spent = self._spend.get(key, ("", 0.0))
        return day == _today() and spent >= limit

    def _record_success(self, public: str, ref: str, cost_usd: float) -> None:
        self._circuits[ref] = {"failures": 0, "opened_at": None}
        for key in {public, ref}:
            day, spent = self._spend.get(key, (_today(), 0.0))
            self._spend[key] = (_today(), spent + cost_usd if day == _today() else cost_usd)

    def _record_failure(self, ref: str) -> None:
        if not self.circuit_breaker:
            return
        circuit = self._circuits.setdefault(ref, {"failures": 0, "opened_at": None})
        circuit["failures"] += 1
        if circuit["failures"] >= self.failure_threshold:
            circuit["opened_at"] = time.monotonic()

    async def _resolve_chain(self, public: str, refs: list[str]) -> list[tuple[str, Model]]:
        chain: list[tuple[str, Model]] = []
        last_error: Exception | None = None
        for ref in self._ordered(public, refs):
            try:
                chain.append((ref, await self._models.use(ref)))
            except Exception as exc:  # broken candidate: skip, fall back
                last_error = exc
        if not chain:
            raise (last_error or NotFoundError("gateway chat model", public))
        return chain

    async def _chat_chain(self, public: str) -> list[tuple[str, Model]]:
        refs = self._refs_for(public, self.chat_routes, "chat")
        if self.objective is None:
            return await self._resolve_chain(public, refs)
        if public not in self._routers:
            from aire.optimization.router import ModelRouter

            chain = await self._resolve_chain(public, refs)
            self._routers[public] = ModelRouter(
                [model for _, model in chain], objective=self.objective, fallback=self.fallback
            )
        return [(self._routers[public].info.ref, self._routers[public])]

    async def _embedder_for(self, public: str) -> tuple[str, EmbeddingModel]:
        refs = self._refs_for(public, self.embedding_routes, "embedding")
        last_error: Exception | None = None
        for ref in self._ordered(public, refs):
            try:
                return ref, await self._models.embedder(ref)
            except Exception as exc:
                last_error = exc
        raise (last_error or NotFoundError("gateway embedding model", public))

    # -- invocation ------------------------------------------------------------------

    async def generate(
        self, public: str, request: GenerationRequest
    ) -> tuple[str, GenerationResult]:
        """Generate with fallback through the route's candidate chain."""
        cached = await self._semantic_lookup(public, request)
        if cached is not None:
            return cached
        chain = await self._chat_chain(public)
        last_error: Exception | None = None
        for ref, model in chain if self.fallback else chain[:1]:
            try:
                result = await model.generate(request)
            except Exception as exc:
                self._record_failure(ref)
                last_error = exc
                continue
            self._record_success(public, ref, result.usage.cost_usd)
            await self._semantic_store(public, request, result)
            return ref, result
        assert last_error is not None
        if isinstance(last_error, AireError):
            raise last_error
        raise ProviderError(
            "gateway",
            f"all candidates for {public!r} failed; last error: "
            f"{type(last_error).__name__}: {last_error}",
        ) from last_error

    async def _ensure_semantic_embedder(self) -> EmbeddingModel | None:
        if not self.semantic_cache:
            return None
        if self._semantic_embedder is not None:
            return self._semantic_embedder
        spec = self._semantic_embedder_spec
        if isinstance(spec, str) or spec is None:
            self._semantic_embedder = await self._models.embedder(spec)
        return self._semantic_embedder

    async def _get_semantic_redis(self) -> Any | None:
        if not self._semantic_redis_url:
            return None
        if self._semantic_redis_backend is not None:
            return self._semantic_redis_backend
        from aire.optimization.redis_cache import RedisCacheBackend

        self._semantic_redis_backend = RedisCacheBackend(
            self._semantic_redis_url, prefix="aire:gateway:sem:"
        )
        return self._semantic_redis_backend

    async def _semantic_lookup(
        self, public: str, request: GenerationRequest
    ) -> tuple[str, GenerationResult] | None:
        embedder = await self._ensure_semantic_embedder()
        if embedder is None:
            return None
        from aire.optimization.cache import _params_signature
        from aire.rag.store import cosine_similarity

        prompt = "\n".join(m.text_content for m in request.messages)
        signature = f"{public}|{_params_signature(request)}"
        vector = await embedder.embed_one(prompt)

        redis = await self._get_semantic_redis()
        if redis is not None:
            import json

            raw = await redis.aget(signature)
            if raw:
                try:
                    payload = json.loads(raw)
                    for item in payload:
                        if cosine_similarity(vector, item["vector"]) >= self.semantic_threshold:
                            self.semantic_hits += 1
                            return item["ref"], GenerationResult.model_validate(item["result"])
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    pass
            self.semantic_misses += 1
            return None

        for cached_sig, cached_vec, cached_ref, result in self._semantic_entries:
            if cached_sig != signature:
                continue
            if cosine_similarity(vector, cached_vec) >= self.semantic_threshold:
                self.semantic_hits += 1
                return cached_ref, result.model_copy(deep=True)
        self.semantic_misses += 1
        return None

    async def _semantic_store(
        self, public: str, request: GenerationRequest, result: GenerationResult
    ) -> None:
        embedder = await self._ensure_semantic_embedder()
        if embedder is None:
            return
        from aire.optimization.cache import _params_signature

        prompt = "\n".join(m.text_content for m in request.messages)
        signature = f"{public}|{_params_signature(request)}"
        vector = await embedder.embed_one(prompt)

        redis = await self._get_semantic_redis()
        if redis is not None:
            import json

            existing: list[dict[str, Any]] = []
            raw = await redis.aget(signature)
            if raw:
                with contextlib.suppress(json.JSONDecodeError, TypeError):
                    existing = list(json.loads(raw))
            existing.append(
                {
                    "vector": vector,
                    "ref": public,
                    "result": result.model_dump(mode="json"),
                }
            )
            existing = existing[-32:]
            await redis.aset(signature, json.dumps(existing), ttl_seconds=3600)
            return

        if len(self._semantic_entries) >= 1024:
            self._semantic_entries.pop(0)
        self._semantic_entries.append((signature, vector, public, result.model_copy(deep=True)))

    async def stream(
        self, public: str, request: GenerationRequest
    ) -> AsyncIterator[tuple[str, GenerationChunk]]:
        """Stream chunks; semantic-cache hits replay as a single chunk stream."""
        cached = await self._semantic_lookup(public, request)
        if cached is not None:
            ref, result = cached
            yield ref, GenerationChunk(
                text=result.text, finish_reason="stop", usage=result.usage
            )
            return

        chain = await self._chat_chain(public)
        last_error: Exception | None = None
        for ref, model in chain if self.fallback else chain[:1]:
            started = False
            pieces: list[str] = []
            cost_usd = 0.0
            try:
                async for chunk in model.stream(request):
                    started = True
                    if chunk.text:
                        pieces.append(chunk.text)
                    if chunk.usage is not None:
                        cost_usd += float(chunk.usage.cost_usd or 0.0)
                    yield ref, chunk
            except Exception as exc:
                self._record_failure(ref)
                last_error = exc
                if started:
                    raise
                continue
            self._record_success(public, ref, cost_usd)
            if pieces and self.semantic_cache:
                await self._semantic_store(
                    public,
                    request,
                    GenerationResult.text_result("".join(pieces), model=ref),
                )
            return
        assert last_error is not None
        raise last_error

    # -- introspection -----------------------------------------------------------------

    def spend_today(self) -> dict[str, float]:
        """Per-alias / per-ref USD spent today (UTC day)."""
        return {
            key: round(spent, 6)
            for key, (day, spent) in sorted(self._spend.items())
            if day == _today()
        }

    def describe(self) -> dict[str, Any]:
        """Machine-readable gateway manifest — for agents and operators."""
        return {
            "kind": "gateway",
            "aire_version": __version__,
            "routing": self.routing,
            "objective": self.objective,
            "fallback": self.fallback,
            "allow_direct_refs": self.allow_direct_refs,
            "circuit_breaker": {
                "enabled": self.circuit_breaker,
                "failure_threshold": self.failure_threshold,
                "cooldown_seconds": self.cooldown_seconds,
                "circuits": {
                    ref: {
                        "failures": state["failures"],
                        "open": state["opened_at"] is not None,
                    }
                    for ref, state in sorted(self._circuits.items())
                },
            },
            "budgets": dict(self.budgets),
            "spend_today": self.spend_today(),
            "chat_models": {k: list(v) for k, v in sorted(self.chat_routes.items())},
            "embedding_models": {k: list(v) for k, v in sorted(self.embedding_routes.items())},
            "semantic_cache": {
                "enabled": self.semantic_cache,
                "threshold": self.semantic_threshold,
                "hits": self.semantic_hits,
                "misses": self.semantic_misses,
                "entries": len(self._semantic_entries),
                "redis": bool(self._semantic_redis_url),
            },
            "endpoints": [
                "/health",
                "/v1/health",
                "/v1/chat/completions",
                "/v1/messages",
                "/v1/embeddings",
                "/v1/images/generations",
                "/v1/models",
                "/v1/providers",
                "/v1/metrics",
                "/v1/gateway/manifest",
                "/v1/gateway/spend",
            ],
        }


def create_gateway(
    runtime: Runtime | None = None,
    *,
    models: list[str] | None = None,
    aliases: dict[str, str | list[str]] | None = None,
    embeddings: dict[str, str | list[str]] | None = None,
    routing: RoutingMode = "first",
    objective: Objective | None = None,
    fallback: bool = True,
    allow_direct_refs: bool = True,
    budgets: dict[str, float] | None = None,
    circuit_breaker: bool = True,
    failure_threshold: int = 3,
    cooldown_seconds: float = 30.0,
    request_log: str | None = None,
    auth_token: str | None = None,
    rate_limit_per_minute: int | None = None,
    metrics: Metrics | None = None,
    title: str = "aire gateway",
    semantic_cache: bool = False,
    semantic_threshold: float = 0.95,
    semantic_embedder: str | EmbeddingModel | None = None,
    semantic_cache_redis_url: str | None = None,
    guardrails: GuardrailChain | list[Guardrail] | bool | None = None,
) -> Any:
    """Build an OpenAI-compatible gateway app. Requires ``pip install aire[serve]``."""
    if FastAPI is None:
        raise ConfigurationError(
            "fastapi is required for the gateway: pip install 'aire[serve]'",
            code="deploy.fastapi_missing",
        )
    if runtime is None:
        from aire.ai import default_runtime

        runtime = default_runtime()

    import os

    from aire.safety.guardrails import resolve_guardrails

    redis_url = semantic_cache_redis_url or os.environ.get("AIRE_REDIS_URL")
    safety_chain = resolve_guardrails(guardrails, safety=runtime.settings.safety)
    chat_routes = _normalize_routes(models=models, aliases=aliases)
    gateway = Gateway(
        runtime,
        chat_routes=chat_routes,
        embedding_routes=_normalize_routes(aliases=embeddings),
        routing=routing,
        objective=objective,
        fallback=fallback,
        allow_direct_refs=allow_direct_refs,
        budgets=budgets,
        circuit_breaker=circuit_breaker,
        failure_threshold=failure_threshold,
        cooldown_seconds=cooldown_seconds,
        semantic_cache=semantic_cache,
        semantic_threshold=semantic_threshold,
        semantic_embedder=semantic_embedder,
        semantic_cache_redis_url=redis_url if semantic_cache else None,
    )
    guard = _make_guard(auth_token=auth_token, rate_limit_per_minute=rate_limit_per_minute)
    log = _make_request_logger(request_log)
    app = FastAPI(title=title, version=__version__)

    @app.exception_handler(AireError)
    async def _aire_error_handler(request: Request, exc: AireError) -> Any:
        return JSONResponse(status_code=_error_status(exc), content=_error_body(exc))

    _register_meta_routes(app, gateway, guard)
    _register_chat_route(app, gateway, guard, runtime, metrics, log, safety_chain)
    _register_anthropic_route(app, gateway, guard, runtime, metrics, log, safety_chain)
    _register_embeddings_route(app, gateway, guard, runtime, metrics, log)
    _register_images_route(app, gateway, guard, runtime, metrics, log)
    _register_providers_route(app, gateway, guard)
    if metrics is not None:
        _register_metrics_route(app, metrics, guard)
    return app


def _register_meta_routes(app: Any, gateway: Gateway, guard: Any) -> None:
    @app.get("/health")
    async def health() -> dict[str, Any]:
        return _health_payload(gateway)

    @app.get("/v1/health")
    async def health_v1() -> dict[str, Any]:
        return _health_payload(gateway)

    @app.get("/v1/models", dependencies=[Depends(guard)])
    async def list_models() -> dict[str, Any]:
        data = [
            _model_card(public, refs, kind="chat")
            for public, refs in sorted(gateway.chat_routes.items())
        ]
        data += [
            _model_card(public, refs, kind="embedding")
            for public, refs in sorted(gateway.embedding_routes.items())
        ]
        return {"object": "list", "data": data}

    @app.get("/v1/gateway/manifest", dependencies=[Depends(guard)])
    async def manifest() -> dict[str, Any]:
        return gateway.describe()

    @app.get("/v1/gateway/spend", dependencies=[Depends(guard)])
    async def spend() -> dict[str, Any]:
        today = _today()
        return {
            "day": today,
            "spend_usd": gateway.spend_today(),
            "budgets_usd": dict(gateway.budgets),
            "remaining_usd": {
                key: round(limit - gateway.spend_today().get(key, 0.0), 6)
                for key, limit in gateway.budgets.items()
            },
        }


def _health_payload(gateway: Gateway) -> dict[str, Any]:
    open_circuits = sum(
        1
        for state in gateway._circuits.values()
        if state.get("opened_at") is not None
        and time.monotonic() - state["opened_at"] < gateway.cooldown_seconds
    )
    return {
        "status": "ok",
        "aire_version": __version__,
        "chat_models": len(gateway.chat_routes),
        "embedding_models": len(gateway.embedding_routes),
        "open_circuits": open_circuits,
        "spend_keys_today": len(gateway.spend_today()),
    }


def _register_chat_route(
    app: Any,
    gateway: Gateway,
    guard: Any,
    runtime: Runtime,
    metrics: Metrics | None,
    log: Any,
    safety_chain: Any | None = None,
) -> None:
    @app.post("/v1/chat/completions", dependencies=[Depends(guard)])
    async def chat_completions(body: dict[str, Any]) -> Any:
        public = _require_model_name(body)
        request = await _guard_generation_request(
            _build_generation_request(body), safety_chain
        )
        if body.get("stream"):
            return StreamingResponse(
                _sse_stream(gateway, public, request, metrics),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        started = time.perf_counter()
        with _maybe_span(runtime, "gateway.chat", {"model": public}):
            resolved, result = await gateway.generate(public, request)
        result = await _guard_generation_result(result, safety_chain)
        _record_chat_metrics(metrics, started, result, public=public)
        _log_request(log, "chat.completions", public, resolved, result, started)
        return JSONResponse(
            content=_chat_completion_body(public, resolved, result),
            headers=_aire_headers(resolved, result),
        )


def _register_anthropic_route(
    app: Any,
    gateway: Gateway,
    guard: Any,
    runtime: Runtime,
    metrics: Metrics | None,
    log: Any,
    safety_chain: Any | None = None,
) -> None:
    @app.post("/v1/messages", dependencies=[Depends(guard)])
    async def anthropic_messages(body: dict[str, Any]) -> Any:
        public = _require_model_name(body)
        request = await _guard_generation_request(
            _build_anthropic_request(body), safety_chain
        )
        if body.get("stream"):
            return StreamingResponse(
                _anthropic_sse_stream(gateway, public, request, metrics),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    "X-Aire-Streaming": "anthropic-sse",
                },
            )
        started = time.perf_counter()
        with _maybe_span(runtime, "gateway.messages", {"model": public}):
            resolved, result = await gateway.generate(public, request)
        result = await _guard_generation_result(result, safety_chain)
        _record_chat_metrics(metrics, started, result, public=public)
        _log_request(log, "messages", public, resolved, result, started)
        return JSONResponse(
            content=_anthropic_body(public, resolved, result),
            headers=_aire_headers(resolved, result),
        )


def _register_embeddings_route(
    app: Any,
    gateway: Gateway,
    guard: Any,
    runtime: Runtime,
    metrics: Metrics | None,
    log: Any,
) -> None:
    @app.post("/v1/embeddings", dependencies=[Depends(guard)])
    async def create_embeddings(body: dict[str, Any]) -> dict[str, Any]:
        public = _require_model_name(body)
        inputs = _embedding_inputs(body.get("input"))
        started = time.perf_counter()
        with _maybe_span(runtime, "gateway.embeddings", {"model": public}):
            resolved, embedder = await gateway._embedder_for(public)
            result = await embedder.embed(EmbeddingRequest(inputs=inputs))
        if metrics is not None:
            metrics.observe_latency("gateway.embeddings", (time.perf_counter() - started) * 1000.0)
            metrics.record_tokens(result.usage.input_tokens, 0, model=resolved)
        return {
            "object": "list",
            "data": [
                {"object": "embedding", "index": i, "embedding": vector}
                for i, vector in enumerate(result.vectors)
            ],
            "model": public,
            "usage": {
                "prompt_tokens": result.usage.input_tokens,
                "total_tokens": result.usage.input_tokens,
            },
            "aire": {"resolved_model": resolved},
        }


def _register_images_route(
    app: Any,
    gateway: Gateway,
    guard: Any,
    runtime: Runtime,
    metrics: Metrics | None,
    log: Any,
) -> None:
    @app.post("/v1/images/generations", dependencies=[Depends(guard)])
    async def create_image(body: dict[str, Any]) -> Any:
        from aire.core.types import Capability
        from aire.vision.pipelines import ImageGenerationPipeline

        public = _require_model_name(body)
        prompt = body.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ConfigurationError(
                "'prompt' (string) is required",
                code="gateway.bad_request",
            )
        size = str(body.get("size") or "1024x1024")
        n = int(body.get("n") or 1)
        # Resolve the first candidate to inspect image-generation capability.
        chain = await gateway._chat_chain(public)
        resolved, model = chain[0]
        if not model.info.supports(Capability.IMAGE_GENERATION):
            return JSONResponse(
                status_code=501,
                content={
                    "error": {
                        "message": (
                            "aire does not synthesize images via chat scrape. "
                            f"Model {resolved!r} lacks the image-generation capability. "
                            "Use a provider/model that advertises Capability.IMAGE_GENERATION."
                        ),
                        "type": "not_implemented_error",
                        "code": "gateway.images_not_supported",
                    }
                },
            )
        started = time.perf_counter()
        images = []
        pipeline = ImageGenerationPipeline(model)
        with _maybe_span(runtime, "gateway.images", {"model": public}):
            for _ in range(max(1, min(n, 4))):
                img = await pipeline.generate(prompt, size=size)
                if img.stub or (not img.uri and not img.b64):
                    return JSONResponse(
                        status_code=501,
                        content={
                            "error": {
                                "message": (
                                    "ImageGenerationPipeline returned a stub result "
                                    "(no real image URI/bytes). aire will not pretend "
                                    "chat text is an image."
                                ),
                                "type": "not_implemented_error",
                                "code": "gateway.images_stub",
                            }
                        },
                    )
                images.append(
                    {
                        "url": img.uri,
                        "b64_json": img.b64,
                        "revised_prompt": prompt,
                    }
                )
                if metrics is not None:
                    metrics.record_tokens(0, 0, model=resolved)
        if metrics is not None:
            metrics.observe_latency("gateway.images", (time.perf_counter() - started) * 1000.0)
        return {
            "created": int(time.time()),
            "data": images,
            "aire": {"resolved_model": resolved, "size": size},
        }


def _register_providers_route(app: Any, gateway: Gateway, guard: Any) -> None:
    @app.get("/v1/providers", dependencies=[Depends(guard)])
    async def list_providers() -> dict[str, Any]:
        from aire.integrations.openai_compat import describe_endpoints

        return {
            "object": "list",
            "gateway": gateway.describe(),
            "openai_compatible": describe_endpoints(),
        }


def _register_metrics_route(app: Any, metrics: Metrics, guard: Any) -> None:
    @app.get("/v1/metrics", dependencies=[Depends(guard)])
    async def prometheus_metrics() -> Any:
        from aire.observability.analytics import Analytics

        body = Analytics(metrics).prometheus()
        return Response(content=body, media_type="text/plain; version=0.0.4")


def _embedding_inputs(raw_input: Any) -> list[str]:
    if isinstance(raw_input, str):
        return [raw_input]
    if isinstance(raw_input, list) and all(isinstance(t, str) for t in raw_input):
        return list(raw_input)
    raise ConfigurationError(
        "'input' must be a string or list of strings",
        code="gateway.bad_request",
    )


# -- payload conversion --------------------------------------------------------------


def _normalize_routes(
    *,
    models: list[str] | None = None,
    aliases: dict[str, str | list[str]] | None = None,
) -> dict[str, list[str]]:
    routes: dict[str, list[str]] = {ref: [ref] for ref in models or []}
    for public, refs in (aliases or {}).items():
        routes[public] = [refs] if isinstance(refs, str) else list(refs)
    return routes


def _require_model_name(body: dict[str, Any]) -> str:
    public = body.get("model")
    if not isinstance(public, str) or not public:
        raise ConfigurationError(
            "'model' (string) is required",
            code="gateway.bad_request",
            context={"hint": "pass an exposed alias or a 'provider:name' ref"},
        )
    return public


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        return "".join(parts)
    return ""


def _build_generation_request(body: dict[str, Any]) -> GenerationRequest:
    from aire.core.content import Message, TextContent

    messages = []
    for raw in body.get("messages") or []:
        if not isinstance(raw, dict):
            continue
        messages.append(
            Message(
                role=raw.get("role", "user"),
                content=[TextContent(text=_content_text(raw.get("content")))],
                name=raw.get("name"),
                tool_call_id=raw.get("tool_call_id"),
            )
        )
    tools = [
        ToolDefinition(
            name=(t.get("function") or {}).get("name", ""),
            description=(t.get("function") or {}).get("description", ""),
            parameters=(t.get("function") or {}).get("parameters")
            or {"type": "object", "properties": {}},
        )
        for t in body.get("tools") or []
        if isinstance(t, dict) and (t.get("function") or {}).get("name")
    ]
    stop_raw = body.get("stop")
    stop = [stop_raw] if isinstance(stop_raw, str) else stop_raw or None
    return GenerationRequest(
        messages=messages,
        temperature=body.get("temperature"),
        max_tokens=body.get("max_tokens"),
        stop=stop,
        tools=tools or None,
        tool_choice=body.get("tool_choice") if isinstance(body.get("tool_choice"), str) else None,
        response_format=_response_format(body.get("response_format")),
        seed=body.get("seed"),
    )


def _response_format(raw: Any) -> StructuredOutputSpec | None:
    if not isinstance(raw, dict):
        return None
    if raw.get("type") == "json_schema" and isinstance(raw.get("json_schema"), dict):
        schema = raw["json_schema"]
        return StructuredOutputSpec(
            name=schema.get("name", "output"),
            schema=schema.get("schema") or {"type": "object"},
            strict=bool(schema.get("strict", True)),
        )
    if raw.get("type") == "json_object":
        return StructuredOutputSpec(name="json", schema={"type": "object"}, strict=False)
    return None


def _tool_calls_payload(result: GenerationResult) -> list[dict[str, Any]]:
    return [
        {
            "id": tc.id or f"call_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
        }
        for tc in result.tool_calls
    ]


def _usage_payload(result: GenerationResult) -> dict[str, int]:
    return {
        "prompt_tokens": result.usage.input_tokens,
        "completion_tokens": result.usage.output_tokens,
        "total_tokens": result.usage.input_tokens + result.usage.output_tokens,
    }


def _chat_completion_body(public: str, resolved: str, result: GenerationResult) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": result.text or None}
    tool_calls = _tool_calls_payload(result)
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": public,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": result.finish_reason,
            }
        ],
        "usage": _usage_payload(result),
        "aire": {"resolved_model": resolved},
    }


# -- streaming ------------------------------------------------------------------------


def _sse(payload: dict[str, Any] | str) -> str:
    data = payload if isinstance(payload, str) else json.dumps(payload, default=str)
    return f"data: {data}\n\n"


async def _sse_stream(
    gateway: Gateway, public: str, request: GenerationRequest, metrics: Metrics | None
) -> AsyncIterator[str]:
    stream_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    resolved = "unknown"

    def _chunk(delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
        return {
            "id": stream_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": public,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }

    started = time.perf_counter()
    emitted_role = False
    try:
        async for _resolved, chunk in gateway.stream(public, request):
            if not emitted_role:
                yield _sse(_chunk({"role": "assistant"}))
                yield _sse({"aire": {"resolved_model": resolved}})
                emitted_role = True
            delta: dict[str, Any] = {}
            if chunk.text:
                delta["content"] = chunk.text
            if chunk.tool_calls:
                delta["tool_calls"] = [
                    {
                        "index": i,
                        "id": tc.id or f"call_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                    }
                    for i, tc in enumerate(chunk.tool_calls)
                ]
            if delta or chunk.finish_reason:
                yield _sse(_chunk(delta, chunk.finish_reason))
    except Exception as exc:
        error = exc if isinstance(exc, AireError) else ProviderError("gateway", str(exc))
        yield _sse(_error_body(error))
    finally:
        if metrics is not None:
            metrics.observe_latency("gateway.chat.stream", (time.perf_counter() - started) * 1000.0)
    yield _sse("[DONE]")


# -- errors, metrics, tracing -----------------------------------------------------------


def _error_status(exc: AireError) -> int:
    if isinstance(exc, NotFoundError):
        return 404
    if isinstance(exc, RateLimitError):
        return 429
    if isinstance(exc, AuthenticationError):
        return 502
    if isinstance(exc, PermissionDeniedError):
        return 403
    if isinstance(exc, SafetyError):
        return 400
    if isinstance(exc, (ConfigurationError, ContextLengthError)):
        return 400
    if isinstance(exc, ProviderError):
        return exc.status or 502
    return 500


async def _guard_generation_request(
    request: GenerationRequest, chain: Any | None
) -> GenerationRequest:
    if chain is None:
        return request
    from aire.core.content import Message, TextContent

    messages: list[Message] = []
    for message in request.messages:
        text = message.text_content
        scrubbed, _ = await chain.aapply(text, stage="input")
        if scrubbed == text:
            messages.append(message)
            continue
        messages.append(
            Message(
                role=message.role,
                content=[TextContent(text=scrubbed)],
                name=message.name,
                tool_call_id=message.tool_call_id,
            )
        )
    return request.with_messages(messages)


async def _guard_generation_result(
    result: GenerationResult, chain: Any | None
) -> GenerationResult:
    if chain is None:
        return result
    from aire.core.content import TextContent

    scrubbed, _ = await chain.aapply(result.text, stage="output")
    if scrubbed == result.text:
        return result
    return result.model_copy(update={"content": [TextContent(text=scrubbed)]})


def _error_body(exc: AireError) -> dict[str, Any]:
    type_by_status = {
        400: "invalid_request_error",
        403: "permission_error",
        404: "not_found_error",
        429: "rate_limit_error",
    }
    status = _error_status(exc)
    return {
        "error": {
            "message": exc.message,
            "type": type_by_status.get(status, "server_error"),
            "code": exc.code,
        }
    }


def _aire_headers(resolved: str, result: GenerationResult) -> dict[str, str]:
    return {
        "X-Aire-Resolved-Model": resolved,
        "X-Aire-Cost-Usd": f"{result.usage.cost_usd:.8f}",
        "X-Aire-Input-Tokens": str(result.usage.input_tokens),
        "X-Aire-Output-Tokens": str(result.usage.output_tokens),
    }


def _model_card(public: str, refs: list[str], *, kind: str) -> dict[str, Any]:
    return {
        "id": public,
        "object": "model",
        "created": 0,
        "owned_by": "aire-gateway",
        "aire_kind": kind,
        "aire_refs": list(refs),
    }


def _record_chat_metrics(
    metrics: Metrics | None, started: float, result: GenerationResult, *, public: str
) -> None:
    if metrics is None:
        return
    metrics.observe_latency("gateway.chat", (time.perf_counter() - started) * 1000.0)
    metrics.record_tokens(result.usage.input_tokens, result.usage.output_tokens, model=public)
    metrics.record_cost(result.usage.cost_usd, model=public)


def _maybe_span(runtime: Runtime, name: str, attributes: dict[str, Any]) -> Any:
    import contextlib

    if runtime.tracer is None:
        return contextlib.nullcontext()
    return runtime.tracer.span(name, attributes=attributes)


# -- anthropic-compatible endpoint --------------------------------------------------------


def _build_anthropic_request(body: dict[str, Any]) -> GenerationRequest:  # noqa: C901
    """Translate an Anthropic /v1/messages body into a GenerationRequest.

    Supports text and image content blocks (base64 / url), tools, tool_choice,
    and multi-turn tool_use / tool_result messages.
    """
    from aire.core.content import Message, StructuredContent, TextContent

    messages: list[Message] = []
    system = _content_text(body.get("system"))
    if system:
        messages.append(Message(role="system", content=[TextContent(text=system)]))
    for raw in body.get("messages") or []:
        if not isinstance(raw, dict):
            continue
        role = raw.get("role", "user")
        content = raw.get("content")
        # Expand Anthropic tool_result user turns into aire tool messages.
        if isinstance(content, list) and any(
            isinstance(p, dict) and p.get("type") == "tool_result" for p in content
        ):
            for part in content:
                if not isinstance(part, dict) or part.get("type") != "tool_result":
                    continue
                tool_use_id = str(part.get("tool_use_id") or "")
                messages.append(
                    Message(
                        role="tool",
                        tool_call_id=tool_use_id or None,
                        content=[TextContent(text=_content_text(part.get("content")))],
                    )
                )
            continue
        if role not in ("user", "assistant"):
            role = "user"
        blocks = _anthropic_content_blocks(content)
        if not blocks:
            blocks = [TextContent(text=_content_text(content))]
        # Keep tool_use as structured blocks on assistant turns.
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "tool_use":
                    blocks.append(
                        StructuredContent(
                            data={
                                "type": "tool_use",
                                "id": part.get("id"),
                                "name": part.get("name"),
                                "input": part.get("input") or {},
                            }
                        )
                    )
        messages.append(Message(role=role, content=blocks))
    tools: list[ToolDefinition] = []
    for raw_tool in body.get("tools") or []:
        if not isinstance(raw_tool, dict):
            continue
        name = raw_tool.get("name")
        if not isinstance(name, str) or not name:
            continue
        tools.append(
            ToolDefinition(
                name=name,
                description=str(raw_tool.get("description") or ""),
                parameters=raw_tool.get("input_schema")
                or raw_tool.get("parameters")
                or {"type": "object", "properties": {}},
            )
        )
    tool_choice_str = _map_anthropic_tool_choice(body.get("tool_choice"))
    stop_raw = body.get("stop_sequences")
    return GenerationRequest(
        messages=messages,
        temperature=body.get("temperature"),
        max_tokens=body.get("max_tokens") or 1024,
        stop=stop_raw if isinstance(stop_raw, list) else None,
        tools=tools or None,
        tool_choice=tool_choice_str,
    )


def _map_anthropic_tool_choice(tool_choice: Any) -> str | None:
    if isinstance(tool_choice, str):
        return tool_choice
    if not isinstance(tool_choice, dict):
        return None
    kind = tool_choice.get("type")
    if kind == "auto":
        return "auto"
    if kind == "any":
        return "required"
    if kind == "none":
        return "none"
    if kind == "tool":
        name = tool_choice.get("name")
        return str(name) if name else "required"
    return str(kind) if kind else None


def _anthropic_content_blocks(content: Any) -> list[Any]:  # noqa: C901
    """Parse Anthropic content blocks into aire TextContent / ImageContent."""
    import base64
    import binascii

    from aire.core.content import ImageContent, TextContent

    if isinstance(content, str):
        return [TextContent(text=content)] if content else []
    if not isinstance(content, list):
        return []
    blocks: list[Any] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        kind = part.get("type")
        if kind == "text":
            text = str(part.get("text") or "")
            if text:
                blocks.append(TextContent(text=text))
        elif kind == "image":
            source = part.get("source") or {}
            if not isinstance(source, dict):
                continue
            src_type = source.get("type")
            if src_type == "base64":
                data = source.get("data")
                media = str(source.get("media_type") or "image/png")
                if isinstance(data, str) and data:
                    try:
                        raw = base64.b64decode(data, validate=True)
                    except (binascii.Error, ValueError):
                        continue
                    blocks.append(ImageContent(data=raw, media_type=media))
            elif src_type == "url":
                url = source.get("url")
                if isinstance(url, str) and url:
                    blocks.append(ImageContent.from_uri(url))
        elif kind == "tool_use":
            # Handled by _build_anthropic_request via StructuredContent.
            continue
        elif kind == "tool_result":
            # Handled by _build_anthropic_request as role=tool messages.
            continue
    return blocks


async def _anthropic_sse_stream(
    gateway: Gateway, public: str, request: GenerationRequest, metrics: Metrics | None
) -> AsyncIterator[str]:
    """Anthropic-compatible SSE stream for ``/v1/messages?stream``."""
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    started = time.perf_counter()
    resolved = "unknown"
    output_tokens = 0
    block_index = 0
    stop_reason = "end_turn"
    try:
        yield _sse(
            {
                "type": "message_start",
                "message": {
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": public,
                },
            }
        )
        yield _sse(
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            }
        )
        async for resolved_ref, chunk in gateway.stream(public, request):
            resolved = resolved_ref
            if chunk.usage is not None:
                output_tokens = max(output_tokens, chunk.usage.output_tokens)
            if chunk.text:
                yield _sse(
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": chunk.text},
                    }
                )
            for tc in chunk.tool_calls:
                stop_reason = "tool_use"
                block_index += 1
                yield _sse(
                    {
                        "type": "content_block_start",
                        "index": block_index,
                        "content_block": {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": {},
                        },
                    }
                )
                yield _sse(
                    {
                        "type": "content_block_delta",
                        "index": block_index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": json.dumps(tc.arguments),
                        },
                    }
                )
                yield _sse({"type": "content_block_stop", "index": block_index})
            if chunk.finish_reason == "tool_calls":
                stop_reason = "tool_use"
            elif chunk.finish_reason == "length":
                stop_reason = "max_tokens"
        yield _sse({"type": "content_block_stop", "index": 0})
        yield _sse(
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason},
                "usage": {"output_tokens": output_tokens},
            }
        )
        yield _sse({"type": "message_stop"})
        yield _sse({"aire": {"resolved_model": resolved}})
    except Exception as exc:
        error = exc if isinstance(exc, AireError) else ProviderError("gateway", str(exc))
        yield _sse({"type": "error", "error": _error_body(error)["error"]})
    finally:
        if metrics is not None:
            metrics.observe_latency(
                "gateway.messages.stream", (time.perf_counter() - started) * 1000.0
            )


_ANTHROPIC_STOP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "refusal",
    "error": "end_turn",
}


def _anthropic_body(public: str, resolved: str, result: GenerationResult) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    if result.text:
        content.append({"type": "text", "text": result.text})
    for tc in result.tool_calls:
        content.append(
            {
                "type": "tool_use",
                "id": tc.id,
                "name": tc.name,
                "input": tc.arguments,
            }
        )
    if not content:
        content.append({"type": "text", "text": ""})
    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": public,
        "stop_reason": _ANTHROPIC_STOP.get(result.finish_reason, "end_turn"),
        "usage": {
            "input_tokens": result.usage.input_tokens,
            "output_tokens": result.usage.output_tokens,
        },
        "aire": {"resolved_model": resolved},
    }


# -- request logging -----------------------------------------------------------------------


def _today() -> str:
    return time.strftime("%Y-%m-%d", time.gmtime())


def _make_request_logger(path: str | None) -> Any:
    if not path:
        return None
    from pathlib import Path

    target = Path(path)

    def _log(entry: dict[str, Any]) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")

    return _log


def _log_request(
    log: Any,
    endpoint: str,
    public: str,
    resolved: str,
    result: GenerationResult,
    started: float,
) -> None:
    if log is None:
        return
    log(
        {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "endpoint": endpoint,
            "model": public,
            "resolved": resolved,
            "input_tokens": result.usage.input_tokens,
            "output_tokens": result.usage.output_tokens,
            "cost_usd": result.usage.cost_usd,
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
            "finish_reason": str(result.finish_reason),
        }
    )

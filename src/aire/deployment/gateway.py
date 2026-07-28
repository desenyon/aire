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
    from fastapi.responses import JSONResponse, StreamingResponse
else:
    try:  # optional dependency: aire[serve]
        from fastapi import Depends, FastAPI, Request
        from fastapi.responses import JSONResponse, StreamingResponse
    except ImportError:  # pragma: no cover - exercised only without the extra
        Depends = FastAPI = Request = JSONResponse = StreamingResponse = None

if TYPE_CHECKING:
    from aire.observability.metrics import Metrics
    from aire.optimization.router import Objective

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
            yield ref, GenerationChunk(text=result.text, finish_reason="stop")
            return

        chain = await self._chat_chain(public)
        last_error: Exception | None = None
        for ref, model in chain if self.fallback else chain[:1]:
            started = False
            pieces: list[str] = []
            try:
                async for chunk in model.stream(request):
                    started = True
                    if chunk.text:
                        pieces.append(chunk.text)
                    yield ref, chunk
            except Exception as exc:
                self._record_failure(ref)
                last_error = exc
                if started:
                    raise
                continue
            self._record_success(public, ref, 0.0)
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
            },
            "endpoints": [
                "/health",
                "/v1/health",
                "/v1/chat/completions",
                "/v1/messages",
                "/v1/embeddings",
                "/v1/models",
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
    )
    guard = _make_guard(auth_token=auth_token, rate_limit_per_minute=rate_limit_per_minute)
    log = _make_request_logger(request_log)
    app = FastAPI(title=title, version=__version__)

    @app.exception_handler(AireError)
    async def _aire_error_handler(request: Request, exc: AireError) -> Any:
        return JSONResponse(status_code=_error_status(exc), content=_error_body(exc))

    _register_meta_routes(app, gateway, guard)
    _register_chat_route(app, gateway, guard, runtime, metrics, log)
    _register_anthropic_route(app, gateway, guard, runtime, metrics, log)
    _register_embeddings_route(app, gateway, guard, runtime, metrics, log)
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
) -> None:
    @app.post("/v1/chat/completions", dependencies=[Depends(guard)])
    async def chat_completions(body: dict[str, Any]) -> Any:
        public = _require_model_name(body)
        request = _build_generation_request(body)
        if body.get("stream"):
            return StreamingResponse(
                _sse_stream(gateway, public, request, metrics),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        started = time.perf_counter()
        with _maybe_span(runtime, "gateway.chat", {"model": public}):
            resolved, result = await gateway.generate(public, request)
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
) -> None:
    @app.post("/v1/messages", dependencies=[Depends(guard)])
    async def anthropic_messages(body: dict[str, Any]) -> Any:
        public = _require_model_name(body)
        request = _build_anthropic_request(body)
        started = time.perf_counter()
        with _maybe_span(runtime, "gateway.messages", {"model": public}):
            resolved, result = await gateway.generate(public, request)
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
        async for resolved, chunk in gateway.stream(public, request):
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
    if isinstance(exc, (ConfigurationError, ContextLengthError)):
        return 400
    if isinstance(exc, ProviderError):
        return exc.status or 502
    return 500


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


def _build_anthropic_request(body: dict[str, Any]) -> GenerationRequest:
    """Translate an Anthropic /v1/messages body into a GenerationRequest."""
    from aire.core.content import Message, TextContent

    messages: list[Message] = []
    system = _content_text(body.get("system"))
    if system:
        messages.append(Message(role="system", content=[TextContent(text=system)]))
    for raw in body.get("messages") or []:
        if not isinstance(raw, dict):
            continue
        role = raw.get("role", "user")
        if role not in ("user", "assistant"):
            role = "user"
        messages.append(
            Message(role=role, content=[TextContent(text=_content_text(raw.get("content")))])
        )
    stop_raw = body.get("stop_sequences")
    return GenerationRequest(
        messages=messages,
        temperature=body.get("temperature"),
        max_tokens=body.get("max_tokens") or 1024,
        stop=stop_raw if isinstance(stop_raw, list) else None,
    )


_ANTHROPIC_STOP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "refusal",
    "error": "end_turn",
}


def _anthropic_body(public: str, resolved: str, result: GenerationResult) -> dict[str, Any]:
    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": result.text}],
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

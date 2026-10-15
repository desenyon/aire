# Gateway

OpenAI-compatible HTTP gateway over aire model refs. Requires `aire-ai[serve]` (FastAPI + uvicorn).

## Create (do not have to bind)

```python
from aire import AI

app = AI.gateway.create(models=["mock:echo"], aliases={"echo": "mock:echo"})
# inspect routes / OpenAPI without listening:
print(app.title)
```

Serve:

```bash
aire gateway -m mock:echo --port 4000
# or
AI.gateway.serve(models=["mock:echo"], port=4000)
```

## Behavior

- Chat/completions and embeddings map to registered `provider:name` refs / aliases
- Routing: `first` | `round_robin`; optional objectives (`lowest_cost`, `balanced`, …)
- Optional auth bearer, rate limit, budgets, circuit breaker (from `aire.yaml` `gateway:` or kwargs)
- Semantic cache optional via `create_gateway(semantic_cache=True)`

## Images → 501 without capability

`POST /v1/images/generations` does **not** scrape chat for fake images. If the resolved model does not advertise `Capability.IMAGE_GENERATION`, the gateway returns **HTTP 501** with code `gateway.images_not_supported` (or stub-related 501 when generation is unimplemented). Prefer real image-capable providers when you need this route.

## Endpoints catalog

```python
AI.gateway.endpoints()   # known local + hosted OpenAI-compat endpoints
AI.gateway.describe()
```

### Output checks while streaming

When guardrails are enabled, the gateway buffers a completion and checks the
entire output before emitting text. This prevents a restricted value split across
provider chunks from escaping. Both OpenAI and Anthropic SSE routes use this rule;
a blocked output or interrupted provider stream emits an error without partial
text. Buffering increases time to first text. `guardrails=False` enables native
incremental streaming when the application explicitly chooses that behavior.

Semantic caching compares only eligible text requests within the same exact
conversation history, system instructions, generation parameters and request
metadata. Requests containing tools or media bypass semantic caching. Put caller
or tenant identifiers in request metadata when using caches across identities.

`/health` is liveness; `/ready` probes exposed chat-model routes and returns 503
when no healthy candidate is available or the runtime is closed. Gateway shutdown
closes its own semantic Redis client. Use `close_runtime=True` only when the app
owns the supplied runtime; externally managed runtimes remain open by default.
OpenAI image URLs and input-audio blocks are preserved. Unsupported content blocks
and malformed generation requests return a structured 400 instead of being dropped.

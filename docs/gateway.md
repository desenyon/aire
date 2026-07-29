# Gateway

OpenAI-compatible HTTP gateway over aire model refs. Requires `aire[serve]` (FastAPI + uvicorn).

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

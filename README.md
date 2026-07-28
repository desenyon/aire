# aire

**The agent-first AI creation library.** One consistent interface from idea to deployed AI system.

```python
from aire import AI

assistant = (
    AI.project("knowledge_assistant")
    .documents("./docs")
    .model("openai:gpt-4o-mini")  # or mock:echo, ollama:llama3.2, anthropic:claude-sonnet-4-5
    .vector_store("local")
    .citations(True)
)

assistant.index()

answer = assistant.ask("What does the documentation say about authentication?")
print(answer.text)
print(answer.citations)

assistant.evaluate("./evals.jsonl", metrics=["accuracy", "groundedness"])
app = assistant.deploy()  # production FastAPI app
```

Works **fully offline** out of the box (`mock:echo` model + `local:hashing` embedder) — every subsystem is testable with zero credentials. Swap providers by changing one string.

## Why agent-first

- **Everything is discoverable**: every component emits a machine-readable `.describe()` manifest (schemas, capabilities, permissions) so agents can introspect the library at runtime.
- **Everything is a tool**: `@AI.tool` turns any function into a self-describing, permissioned, auditable tool with JSON-schema contracts.
- **Deterministic agent runtime**: agents run as explicit state machines with budgets, permission checks, and full step traces — never unbounded recursion.
- **Structured everything**: Pydantic models for all requests/responses; structured errors with stable codes (`error.code`, `error.retryable`).
- **Provider independence**: `provider:name` references (`openai:gpt-4o-mini`, `anthropic:claude-sonnet-4-5`, `ollama:llama3.2`) resolved through a registry. Core never imports vendor SDKs.

## Install

```bash
pip install aire                 # core (zero heavy deps)
pip install "aire[serve]"        # FastAPI deployment
pip install "aire[all]"          # everything
```

Requires Python 3.11+.

## The five-minute tour

```bash
pip install aire
aire init my-project && cd my-project
aire doctor                      # diagnose environment
aire run "hello, aire"           # one-shot generation
python app.py                    # full RAG pipeline, offline
```

## Capabilities

| Namespace | What you get |
|---|---|
| `AI.models` | Unified model interface, `provider:name` registry, streaming, structured output, tool calling, model router, caching |
| `AI.data` | Load (files/dirs/URLs/JSONL/CSV), validate, dedupe, split, sample, lineage, quality reports, chunkers |
| `AI.rag` | Knowledge pipelines: ingest → chunk → embed → hybrid retrieve → rerank → grounded answer with citations |
| `AI.agents` | Deterministic state-machine agents: tools, memory, budgets, permissions, human approval, full traces |
| `AI.workflows` | Graph engine: conditional branches, parallel execution, retries, checkpoints, streaming events |
| `AI.evaluate` | Metrics (exact, semantic, groundedness, latency, cost, model-judge), regression suites, rich reports |
| `AI.observe` | OpenTelemetry-style tracing, metrics, event history |
| `AI.deploy` | FastAPI app factory (health/ready/metrics/manifest), Dockerfile + artifact generation |
| `AI.safety` | Guardrails (PII, injection, secrets), risk-classified side effects, redaction, approval policies |
| `AI.training` | Framework-independent trainer contract + function trainer (PyTorch/JAX via plugins) |
| `AI.synthetic` | Synthetic QA/eval data generation |

## Providers

`openai` (any OpenAI-compatible endpoint), `anthropic`, `huggingface`, `ollama`, `mock` (offline), `callable` (any Python function). Vector stores: `local`, `qdrant`, `chroma`. All reached over plain HTTP — no vendor SDKs. Write your own in ~30 lines: see [PLUGIN_SPEC.md](docs/PLUGIN_SPEC.md).

## CLI

```bash
aire init | run | evaluate | serve | inspect | plugins | doctor
```

## Documentation

- [docs/PRODUCT_SPEC.md](docs/PRODUCT_SPEC.md) — product principles and scope
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — layers, contracts, decisions
- [docs/PUBLIC_API.md](docs/PUBLIC_API.md) — stability guarantees
- [docs/PLUGIN_SPEC.md](docs/PLUGIN_SPEC.md) — writing providers and plugins
- [docs/SECURITY_MODEL.md](docs/SECURITY_MODEL.md) — threat model and controls
- [docs/ROADMAP.md](docs/ROADMAP.md) — phases and version plan
- [examples/](examples/) — runnable projects (all offline-capable)

## Development

```bash
pip install -e ".[dev]"
ruff check . && ruff format --check .
mypy src
pytest
```

## License

Apache-2.0 — see [LICENSE](LICENSE).

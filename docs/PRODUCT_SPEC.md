# aire — Product Specification

Version: 0.1.0 (first working release)

## Mission

`aire` is an agent-first AI creation library: a single, consistent framework that
takes a developer from an idea to a deployed AI system without stitching together
vendor SDKs, ad-hoc scripts and infrastructure glue.

The library is **agent-first**: every component is discoverable, self-describing
and composable, so both humans *and* coding agents can build on it safely.
Machine-readable manifests (`.describe()`), structured errors, strict typing and
declarative configuration are the mechanisms that make this true.

## Product principle

One consistent interface from idea to deployment:

```python
from aire import AI

assistant = (
    AI.project("knowledge_assistant")
    .documents("./docs")
    .model("openai:gpt-4o-mini")
    .vector_store("local:default")
    .citations(True)
)

assistant.index()
answer = assistant.ask("What does the documentation say about authentication?")
print(answer.text, answer.citations)
assistant.evaluate("./evaluation_questions.jsonl")
assistant.deploy()   # FastAPI app
```

Three levels of abstraction are supported for every subsystem:

1. **Declarative** — `AI.from_config("aire.yaml")` / `AI.project(...)`.
2. **Composable** — `AI.models.use(...)`, `AI.rag.knowledge(...)`, `AI.agents.create(...)`.
3. **Low level** — direct provider and protocol use (`GenerationRequest`, `Model.generate`).

## Subsystems (version 0.1)

| Subsystem | Module | Status in 0.1 |
|---|---|---|
| Core runtime | `aire.core` | complete |
| Models & embeddings | `aire.models` | complete |
| Provider integrations | `aire.integrations` | openai, anthropic, ollama, huggingface, mock, qdrant, chroma |
| Data | `aire.data` | loaders, dataset ops, chunkers |
| RAG | `aire.rag` | local store, hybrid retrieval, reranking, citations |
| Tools | `aire.tools` | decorator, registry, builtin tools, permissions |
| Agents | `aire.agents` | deterministic state machine, memory, budgets, approval |
| Workflows | `aire.workflows` | graph engine: branches, parallel, retries, checkpoints, streaming |
| Evaluation | `aire.evaluation` | metrics registry, model judges, reports |
| Observability | `aire.observability` | tracing (OTel-shaped), metrics, events |
| Safety | `aire.safety` | guardrails, side-effect risk levels, redaction, approval policy |
| Optimization | `aire.optimization` | exact/semantic caching, model router |
| Multimodal | `aire.multimodal`, `aire.vision`, `aire.audio` | normalized content + conversion pipelines |
| Synthetic data | `aire.synthetic` | QA generation, dataset augmentation |
| Training | `aire.training` | framework-agnostic trainer loop |
| Deployment | `aire.deployment` | FastAPI app factory, artifact generation |
| CLI | `aire.cli` | init, run, evaluate, serve, inspect, plugins, doctor, version |

Explicitly **out of scope for 0.1** (interfaces reserved, see `docs/ROADMAP.md`):
distributed training, reinforcement learning environments, hosted vector DB
writes beyond qdrant/chroma, and the local inspection web UI.

## Users

- **Application developers** building agents/RAG/workflow systems who want
  provider independence and production plumbing for free.
- **Coding agents** consuming aire as a toolkit: manifests, structured errors
  and typed contracts let an agent discover and compose capabilities without
  reading source code.
- **Platform teams** extending aire with internal providers via the plugin API.

## Non-functional requirements

- Python ≥ 3.11, strict typing (`mypy --strict` clean), Pydantic v2 models.
- Async-first for all network and inference paths (`async`/`await`, async iterators).
- Core never imports vendor SDKs; providers are reached over `httpx` or plugins.
- Optional dependencies are lazily imported and gated behind extras.
- Structured errors (`AireError` hierarchy) with machine-readable `code`,
  `context`, `retryable` and causal chaining — never bare `Exception` escapes.
- Backward-compatible public API from 1.0 onward (see `docs/PUBLIC_API.md`).
- Every public component emits a manifest via `.describe()`.

## Success criteria (from the development plan)

1. Build common AI systems with less code.
2. Change model providers without rewriting applications.
3. Combine AI capabilities through consistent interfaces.
4. Inspect and evaluate every execution.
5. Deploy systems without constructing infrastructure from scratch.
6. Extend the library without modifying its core.
7. Simple abstractions without losing access to advanced controls.
8. Reliable AI applications, not isolated demonstrations.

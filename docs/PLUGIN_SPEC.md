# aire — Plugin Specification

Plugins extend aire with new model providers, embedders, vector stores, tools
and more — without modifying the core.

## Contract

A plugin is any importable module (or `entry_points` distribution) exposing:

```python
def register(runtime: aire.core.runtime.Runtime) -> aire.core.plugins.PluginInfo:
    ...
```

`PluginInfo` (Pydantic) declares:

| Field | Meaning |
|---|---|
| `name` | unique plugin name |
| `version` | plugin version string |
| `providers` | model provider names this plugin registers |
| `description` | human/agent-readable summary |

## Discovery

1. **Entry points** — declare the group in your `pyproject.toml`:

   ```toml
   [project.entry-points."aire.providers"]
   myprovider = "my_package.aire_plugin:MyPlugin"
   ```

   Entry-point objects may be a class with `register(runtime)`, a module, or a
   function. `PluginManager.discover()` loads them lazily.

2. **Programmatic** — `runtime.plugins.register_module("my_package.aire_plugin")`
   or register factories directly:

   ```python
   runtime.model_providers.register("myprovider", my_model_factory)
   runtime.embedders.register("myprovider", my_embedder_factory)
   runtime.vector_stores.register("mystore", my_store_factory)
   ```

## Factory signatures

Factories are called by the registries as:

```python
model_factory(name: str, *, runtime: Runtime, **options) -> Model
embedder_factory(name: str, *, runtime: Runtime, **options) -> EmbeddingModel
store_factory(name: str, *, runtime: Runtime, **options) -> VectorStore
```

Returning the wrong interface type raises `ProviderError` at resolution time.

## Interface requirements

- **Models** implement `aire.models.base.Model`: `generate`, `stream`, `info`,
  `health`, `describe()`. Use `GenerationRequest`/`GenerationResult` — never
  expose vendor payload types.
- **Embedders** implement `aire.models.base.EmbeddingModel`: `embed`,
  `embed_one`, `dimensions`, `health`.
- **Vector stores** implement `aire.rag.store.VectorStore`: `upsert`, `search`,
  `delete`, `count`, `health`, `describe()` (see `tests/contract` for the exact
  behavioral contract every store must satisfy).

## Rules

1. Raise `AireError` subclasses (`ProviderError`, `RateLimitError`, …) —
   wrap vendor exceptions; never leak them.
2. Register HTTP clients and connections with
   `runtime.resources.track_resource(...)` so shutdown is clean.
3. Read credentials from `runtime.settings.providers` or environment variables —
   never hard-code secrets.
4. Keep heavy imports function-local so `import aire` stays fast.
5. Emit a `Manifest` from `.describe()` so agents can discover capabilities.
6. Ship contract tests: run your provider against `tests/contract` equivalents.

## Builtin example

`aire.integrations.mock` is the reference plugin: zero-dependency, offline,
registered through the same entry-point machinery as third-party plugins.

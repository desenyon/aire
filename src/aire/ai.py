"""The AI facade: one consistent entry point to every subsystem.

Three levels of abstraction:

1. Declarative — ``AI.project(...)`` fluent builder / ``AI.from_config("aire.yaml")``
2. Composable  — ``AI.models.use(...)``, ``AI.rag.create(...)``, ``AI.agents.create(...)``
3. Low level   — provider adapters and protocol implementations directly
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from aire.core.config import Settings
from aire.core.runtime import Runtime
from aire.models.base import EmbeddingModel, Model, run_sync
from aire.models.registry import ModelRegistry, register_callable
from aire.tools.tool import Tool
from aire.tools.tool import tool as tool_decorator
from aire.tools.types import RetryPolicy, SideEffect

if TYPE_CHECKING:
    from aire.agents.agent import Agent
    from aire.agents.types import AgentConfig
    from aire.data.dataset import Dataset
    from aire.evaluation.runner import Evaluator
    from aire.evaluation.types import EvalReport
    from aire.knowledge_assistant import Assistant
    from aire.observability.metrics import Metrics
    from aire.observability.tracing import Tracer
    from aire.optimization.router import ModelRouter, Objective
    from aire.rag.pipeline import Knowledge
    from aire.rag.store import VectorStore
    from aire.safety.guardrails import Guardrail
    from aire.tools.registry import ToolRegistry
    from aire.workflows.graph import Workflow

_default_runtime: Runtime | None = None


def default_runtime() -> Runtime:
    """The process-wide default runtime (created lazily)."""
    global _default_runtime
    if _default_runtime is None:
        _default_runtime = _build_runtime()
    return _default_runtime


def _build_runtime(settings: Settings | None = None) -> Runtime:
    runtime = Runtime(settings)
    from aire.rag.store import register as register_local_store

    register_local_store(runtime)
    return runtime


class _Namespace:
    """Base for facade namespaces; binds to the default runtime lazily."""

    def __init__(self, runtime: Runtime | None = None) -> None:
        self._bound = runtime

    def _rt(self) -> Runtime:
        return self._bound or default_runtime()


class _ModelsNamespace(_Namespace):
    def _registry(self) -> ModelRegistry:
        return ModelRegistry(self._rt())

    async def use(self, spec: str, **options: Any) -> Model:
        """Resolve ``"provider:name"`` to a live model."""
        return await self._registry().use(spec, **options)

    def use_sync(self, spec: str, **options: Any) -> Model:
        return run_sync(self.use(spec, **options))

    async def embedder(self, spec: str | None = None, **options: Any) -> EmbeddingModel:
        return await self._registry().embedder(spec, **options)

    def embedder_sync(self, spec: str | None = None, **options: Any) -> EmbeddingModel:
        return run_sync(self.embedder(spec, **options))

    def register_callable(self, name: str, fn: Any) -> None:
        """Expose a Python function as ``callable:<name>``."""
        register_callable(name, fn)

    def router(
        self,
        candidates: list[str | Model],
        *,
        objective: Objective = "balanced",
        **kwargs: Any,
    ) -> ModelRouter:
        """Build a model router over candidate models or specs."""
        from aire.optimization.router import ModelRouter

        resolved = [self.use_sync(c) if isinstance(c, str) else c for c in candidates]
        return ModelRouter(resolved, objective=objective, **kwargs)

    def cache(self, model: Model, **kwargs: Any) -> Any:
        from aire.optimization.cache import CachedModel

        return CachedModel(model, **kwargs)

    def providers(self) -> list[str]:
        return self._rt().model_providers.names()

    def describe(self) -> dict[str, Any]:
        return self._registry().describe()


class _DataNamespace(_Namespace):
    def load(self, source: Any, **options: Any) -> Dataset:
        from aire.data.loaders import load

        return load(source, **options)

    def chunker(self, name: str = "recursive", **options: Any) -> Any:
        from aire.data.chunking import get_chunker

        return get_chunker(name, **options)

    def describe(self) -> dict[str, Any]:
        return {
            "loaders": ["jsonl", "json", "csv", "text", "directory", "url", "memory"],
            "chunkers": ["fixed", "sentence", "recursive", "semantic"],
        }


class _RagNamespace(_Namespace):
    def create(self, **options: Any) -> Knowledge:
        """Create a Knowledge pipeline (store/embedder/chunker/reranker options)."""
        from aire.rag.pipeline import Knowledge

        return Knowledge(self._rt(), **options)

    def vector_store(self, spec: str = "local:default", **options: Any) -> VectorStore:
        """Resolve a vector store by ``provider:name``."""
        from aire.core.types import Ref

        ref = Ref.parse(spec)
        runtime = self._rt()
        if not runtime.vector_stores.has(ref.provider):
            from aire.models.registry import _maybe_hint_integration

            _maybe_hint_integration(ref.provider, runtime)
        store = runtime.vector_stores.create(
            ref.provider, name=ref.name, runtime=runtime, **options
        )
        return cast("VectorStore", store)

    def describe(self) -> dict[str, Any]:
        return {"stores": self._rt().vector_stores.names()}


class _AgentsNamespace(_Namespace):
    async def create(
        self,
        model: str | Model | None = None,
        *,
        tools: list[Tool | str] | None = None,
        memory: str | None = None,
        config: AgentConfig | None = None,
        approver: Any = None,
        name: str = "agent",
        builtins: bool = False,
    ) -> Agent:
        """Create an agent; tool name strings resolve against the runtime registry."""
        from aire.agents.agent import Agent
        from aire.tools.builtins import builtin_tools

        runtime = self._rt()
        if isinstance(model, Model):
            resolved_model = model
        else:
            resolved_model = await ModelRegistry(runtime).use(model or runtime.settings.model.ref)
        resolved_tools: list[Tool] = []
        if builtins:
            resolved_tools.extend(builtin_tools())
        for t in tools or []:
            if isinstance(t, Tool):
                resolved_tools.append(t)
            elif runtime.tools.has(t):
                resolved_tools.append(runtime.tools.create(t))
            else:
                from aire.core.errors import NotFoundError

                raise NotFoundError("tool", t)
        return Agent(
            resolved_model,
            tools=resolved_tools,
            memory=memory,
            config=config,
            runtime=runtime,
            approver=approver,
            name=name,
        )

    def create_sync(self, model: str | Model | None = None, **kwargs: Any) -> Agent:
        return run_sync(self.create(model, **kwargs))

    def describe(self) -> dict[str, Any]:
        return {"kind": "agents", "memory": ["buffer", "jsonl:<path>"]}


class _ObserveNamespace(_Namespace):
    def __init__(self, runtime: Runtime | None = None) -> None:
        super().__init__(runtime)
        self._metrics: Metrics | None = None

    def tracer(self) -> Tracer:
        """The runtime tracer (created and attached on first access)."""
        runtime = self._rt()
        if runtime.tracer is None:
            from aire.observability.tracing import JsonlExporter, Tracer

            obs = runtime.settings.observability
            exporter: Any = None
            if obs.exporter == "jsonl" and obs.trace_file:
                exporter = JsonlExporter(obs.trace_file)
            runtime.tracer = Tracer(exporter=exporter, mask_fields=obs.mask_fields)
        return runtime.tracer

    @property
    def metrics(self) -> Metrics:
        if self._metrics is None:
            from aire.observability.metrics import Metrics

            self._metrics = Metrics()
        return self._metrics

    def traces(self) -> list[dict[str, Any]]:
        tracer = self._rt().tracer
        if tracer is None:
            return []
        return [r.model_dump(mode="json") for r in tracer.records()]

    def events(self, pattern: str | None = None) -> list[dict[str, Any]]:
        events = self._rt().events.history
        if pattern:
            events = [e for e in events if e.matches(pattern)]
        return [{"topic": e.topic, "data": e.data, "timestamp": e.timestamp} for e in events]

    def describe(self) -> dict[str, Any]:
        runtime = self._rt()
        return {
            "tracer": runtime.tracer.describe() if runtime.tracer else None,
            "events": len(runtime.events.history),
        }


class _DeployNamespace(_Namespace):
    def api(self, target: Any, **options: Any) -> Any:
        """Wrap a target in a production FastAPI app (requires aire[serve])."""
        from aire.deployment.fastapi_app import create_app

        return create_app(target, **options)

    def artifacts(self, directory: str | Path, **options: Any) -> Any:
        from aire.deployment.artifacts import generate_artifacts

        return generate_artifacts(directory, **options)

    def describe(self) -> dict[str, Any]:
        from aire.deployment.artifacts import describe

        return describe()


class _WorkflowNamespace(_Namespace):
    def create(self, name: str = "workflow", **options: Any) -> Workflow:
        from aire.workflows.graph import Workflow

        return Workflow(name, **options)


class _GatewayNamespace(_Namespace):
    def create(self, **options: Any) -> Any:
        """Build an OpenAI-compatible gateway app (requires aire[serve]).

        Unset options fall back to the ``gateway:`` section of aire.yaml.
        """
        from aire.deployment.gateway import create_gateway

        config = self._rt().settings.gateway
        defaults: dict[str, Any] = {
            "models": config.models or None,
            "aliases": config.aliases or None,
            "embeddings": config.embeddings or None,
            "routing": config.routing,
            "objective": config.objective,
            "auth_token": config.auth_token,
            "rate_limit_per_minute": config.rate_limit_per_minute,
        }
        merged = {**defaults, **{k: v for k, v in options.items() if v is not None}}
        return create_gateway(self._rt(), **merged)

    def serve(self, host: str = "127.0.0.1", port: int = 4000, **options: Any) -> None:
        """Build the gateway and serve it with uvicorn."""
        try:
            import uvicorn
        except ImportError as exc:
            from aire.core.errors import ConfigurationError

            raise ConfigurationError(
                "uvicorn required: pip install 'aire[serve]'",
                code="deploy.uvicorn_missing",
                cause=exc,
            ) from exc
        uvicorn.run(self.create(**options), host=host, port=port)

    def endpoints(self) -> dict[str, Any]:
        """Catalog of known OpenAI-compatible endpoints (local + hosted)."""
        from aire.integrations.openai_compat import describe_endpoints

        return describe_endpoints()

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "gateway",
            "routing_modes": ["first", "round_robin"],
            "objectives": [
                "lowest_cost",
                "lowest_latency",
                "highest_quality",
                "quality_under_budget",
                "balanced",
            ],
            "endpoints": self.endpoints(),
        }


class _TrainingNamespace(_Namespace):
    def create(self, step: Any, **options: Any) -> Any:
        from aire.training.trainer import FunctionTrainer, TrainingConfig

        config = options.pop("config", None) or TrainingConfig(**options)
        return FunctionTrainer(step, config)


class _SafetyNamespace(_Namespace):
    def guardrails(self, *names: str) -> Any:
        from aire.safety.guardrails import (
            GuardrailChain,
            InjectionGuardrail,
            PIIGuardrail,
            SecretGuardrail,
        )

        mapping: dict[str, type[Guardrail]] = {
            "pii": PIIGuardrail,
            "injection": InjectionGuardrail,
            "secret": SecretGuardrail,
        }
        rails = [mapping[n]() for n in names] if names else None
        return GuardrailChain(rails)

    def redact(self, text: str, **options: Any) -> str:
        from aire.safety.redaction import redact

        return redact(text, **options)


class AI:
    """Unified entry point to the aire library.

    Namespaces operate on the process-wide default runtime; use
    :meth:`AI.configure` to replace it (e.g. with a specific config file).
    """

    models = _ModelsNamespace()
    data = _DataNamespace()
    rag = _RagNamespace()
    agents = _AgentsNamespace()
    observe = _ObserveNamespace()
    deploy = _DeployNamespace()
    gateway = _GatewayNamespace()
    workflows = _WorkflowNamespace()
    training = _TrainingNamespace()
    safety = _SafetyNamespace()

    # -- runtime -----------------------------------------------------------------

    @classmethod
    def runtime(cls) -> Runtime:
        return default_runtime()

    @classmethod
    def configure(cls, settings: Settings | None = None, **overrides: Any) -> Runtime:
        """Rebuild the default runtime from explicit settings or overrides."""
        global _default_runtime
        settings = settings or Settings.load(overrides=overrides or None)
        _default_runtime = _build_runtime(settings)
        return _default_runtime

    # -- projects (Level 1: declarative) ---------------------------------------------

    @classmethod
    def project(cls, name: str, *, config: str | Path | None = None) -> Assistant:
        """Start a fluent project builder (documents → index → ask → deploy)."""
        from aire.knowledge_assistant import Assistant

        runtime = default_runtime() if config is None else _build_runtime(Settings.load(config))
        return Assistant(name, runtime)

    @classmethod
    def from_config(cls, path: str | Path) -> Assistant:
        """Load a project from a configuration file."""
        settings = Settings.load(path)
        return cls.project(settings.project, config=path)

    # -- workflows ------------------------------------------------------------------------

    @classmethod
    def workflow(cls, name: str = "workflow", **options: Any) -> Workflow:
        return cls.workflows.create(name, **options)

    # -- tools --------------------------------------------------------------------------

    @classmethod
    def tool(
        cls,
        fn: Any = None,
        *,
        register: bool = True,
        name: str | None = None,
        description: str | None = None,
        permissions: list[str] | None = None,
        timeout_seconds: float = 30.0,
        retry: RetryPolicy | None = None,
        side_effect: SideEffect | str = SideEffect.READ_ONLY,
        **kwargs: Any,
    ) -> Any:
        """``@AI.tool`` decorator: create (and optionally register) a tool."""
        decorator = tool_decorator(
            fn,
            name=name,
            description=description,
            permissions=permissions,
            timeout_seconds=timeout_seconds,
            retry=retry,
            side_effect=side_effect,
            **kwargs,
        )
        if not register:
            return decorator

        def _register(t: Tool) -> Tool:
            default_runtime().tools.register(t.name, lambda: t, replace=True)
            return t

        if isinstance(decorator, Tool):
            return _register(decorator)

        def _wrapped(f: Any) -> Tool:
            return _register(decorator(f))

        return _wrapped

    @classmethod
    def tools(cls) -> ToolRegistry:
        """The runtime tool registry as a ToolRegistry view."""
        from aire.tools.registry import ToolRegistry

        registry = ToolRegistry()
        runtime = default_runtime()
        for tool_name in runtime.tools.names():
            registry.register(runtime.tools.create(tool_name))
        return registry

    # -- evaluation ----------------------------------------------------------------------

    @classmethod
    def evaluator(cls, *, judge: Model | None = None, name: str = "evaluation") -> Evaluator:
        from aire.evaluation.runner import Evaluator

        return Evaluator(judge=judge, name=name)

    @classmethod
    def evaluate(
        cls,
        target: Any,
        dataset: Any,
        *,
        metrics: list[str] | None = None,
        judge: Model | None = None,
        **kwargs: Any,
    ) -> EvalReport:
        """Evaluate any target (agent, knowledge, model, callable) synchronously."""
        return cls.evaluator(judge=judge).run_sync(target, dataset, metrics=metrics, **kwargs)

    # -- synthetic -----------------------------------------------------------------------

    @classmethod
    def synthetic(cls, model: str | Model | None = None) -> Any:
        from aire.synthetic.generator import SyntheticGenerator

        resolved = (
            model
            if isinstance(model, Model)
            else cls.models.use_sync(model or default_runtime().settings.model.ref)
        )
        return SyntheticGenerator(resolved)

    # -- introspection ---------------------------------------------------------------------

    @classmethod
    def describe(cls) -> dict[str, Any]:
        """Machine-readable description of the whole library surface — for agents."""
        from aire._version import __version__

        return {
            "library": "aire",
            "version": __version__,
            "runtime": default_runtime().describe(),
            "namespaces": [
                "models",
                "data",
                "rag",
                "agents",
                "workflows",
                "training",
                "observe",
                "deploy",
                "gateway",
                "safety",
                "synthetic",
            ],
        }

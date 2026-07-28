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
        """Build a model router over candidate models or specs.

        Pass ``policy=CostPolicy(...)`` for daily/per-request budget guards.
        """
        from aire.optimization.router import ModelRouter

        resolved = [self.use_sync(c) if isinstance(c, str) else c for c in candidates]
        return ModelRouter(resolved, objective=objective, **kwargs)

    def cost_policy(self, **options: Any) -> Any:
        """Create a :class:`~aire.optimization.cost_policy.CostPolicy` for routers."""
        from aire.optimization.cost_policy import CostPolicy

        return CostPolicy(**options)

    def cache(self, model: Model, **kwargs: Any) -> Any:
        backend = kwargs.pop("backend", "memory")
        if backend == "redis":
            from aire.optimization.redis_cache import RedisCachedModel

            return RedisCachedModel(model, **kwargs)
        if kwargs.get("embedder") is not None or kwargs.pop("semantic", False):
            from aire.optimization.cache import SemanticCachedModel

            embedder = kwargs.pop("embedder", None)
            if embedder is None:
                embedder = self.embedder_sync()
            return SemanticCachedModel(model, embedder, **kwargs)
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
            "loaders": [
                "jsonl",
                "json",
                "csv",
                "html",
                "parquet",
                "excel",
                "text",
                "directory",
                "url",
                "memory",
            ],
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
        skills: list[str] | None = None,
        session: Any = None,
    ) -> Agent:
        """Create an agent; tool name strings resolve against the runtime registry."""
        from aire.agents.agent import Agent
        from aire.agents.skills import apply_skill
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
        agent = Agent(
            resolved_model,
            tools=resolved_tools,
            memory=memory,
            config=config,
            runtime=runtime,
            approver=approver,
            name=name,
            session=session,
        )
        for skill_name in skills or []:
            apply_skill(agent, skill_name, builtins=True)
        return agent

    def create_sync(self, model: str | Model | None = None, **kwargs: Any) -> Agent:
        return run_sync(self.create(model, **kwargs))

    def team(
        self,
        members: dict[str, Agent] | list[Agent],
        supervisor: str | Model | None = None,
        **options: Any,
    ) -> Any:
        """Create a supervisor-routed Team of agents.

        ``supervisor`` is any model ref (defaults to the configured model) that
        decides which member handles each subtask.
        """
        from aire.agents.team import Team

        runtime = self._rt()
        if not isinstance(supervisor, Model):
            supervisor = run_sync(
                ModelRegistry(runtime).use(supervisor or runtime.settings.model.ref)
            )
        return Team(members, supervisor, **options)

    def swarm(self, agents: list[Agent], goal: str, **options: Any) -> Any:
        from aire.agents.topologies import swarm

        return run_sync(swarm(agents, goal, **options))

    def debate(self, agents: list[Agent], goal: str, **options: Any) -> Any:
        from aire.agents.topologies import debate

        return run_sync(debate(agents, goal, **options))

    def topologies(self) -> Any:
        from aire.agents import topologies

        return topologies

    def approver(self, kind: str = "rule", **options: Any) -> Any:
        """Build an approval policy: "rule" (side-effect thresholds) or
        "interactive" (human-in-the-loop stdin prompts)."""
        from aire.agents.approvals import InteractiveApprover, RuleApprover
        from aire.core.errors import ConfigurationError
        from aire.workflows.hitl import NodeInteractiveApprover

        if kind == "rule":
            return RuleApprover(**options)
        if kind == "interactive":
            return InteractiveApprover(**options)
        if kind in {"workflow", "node"}:
            return NodeInteractiveApprover(**options)
        raise ConfigurationError(
            f"unknown approver kind {kind!r}",
            code="agents.approver_unknown",
            context={"available": ["rule", "interactive", "workflow"]},
        )

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "agents",
            "memory": ["buffer", "jsonl:<path>", "long-term (AI.memory)"],
            "composition": ["agent.as_tool()", "AI.agents.team(...)", "topologies"],
            "approvers": ["rule", "interactive", "workflow"],
            "topologies": ["swarm", "debate", "auction", "blackboard"],
        }


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
            elif obs.exporter == "otlp" and obs.otlp_endpoint:
                from aire.observability.otlp import OTLPExporter

                exporter = OTLPExporter(obs.otlp_endpoint, service_name=obs.service_name)
            runtime.tracer = Tracer(exporter=exporter, mask_fields=obs.mask_fields)
        return runtime.tracer

    @property
    def metrics(self) -> Metrics:
        if self._metrics is None:
            from aire.observability.metrics import Metrics

            self._metrics = Metrics()
        return self._metrics

    def traces(
        self,
        *,
        name: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        tracer = self._rt().tracer
        if tracer is None:
            return []
        rows = [r.model_dump(mode="json") for r in tracer.records()]
        if name:
            rows = [r for r in rows if name in str(r.get("name", ""))]
        if limit is not None:
            rows = rows[-limit:]
        return rows

    def costs(self) -> dict[str, Any]:
        """Aggregate cost counters from the in-process metrics registry."""
        snap = self.metrics.snapshot()
        by_model: dict[str, float] = {}
        total = 0.0
        for key, value in snap.get("counters", {}).items():
            if not str(key).startswith("aire.cost.usd"):
                continue
            total += float(value)
            if "model=" in str(key):
                model = str(key).split("model=", 1)[1].rstrip("}")
                by_model[model] = by_model.get(model, 0.0) + float(value)
        return {"total_usd": total, "by_model": by_model, "snapshot": snap}

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
            "otlp": "aire.observability.otlp.OTLPExporter",
            "otel_sdk": "aire.observability.otel_sdk.SdkBridgeExporter",
        }

    def otel_exporter(self, endpoint: str | None = None, **options: Any) -> Any:
        from aire.observability.otel_sdk import create_exporter

        return create_exporter(endpoint, **options)


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

    def hitl_node(
        self,
        workflow: Workflow,
        name: str,
        fn: Any,
        **options: Any,
    ) -> Workflow:
        """Add a node that requires human approval before running."""
        from aire.workflows.hitl import hitl_node

        return hitl_node(workflow, name, fn, **options)

    def interactive_approver(self, **options: Any) -> Any:
        from aire.workflows.hitl import NodeInteractiveApprover

        return NodeInteractiveApprover(**options)

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "workflows",
            "features": ["graph", "checkpoints", "hitl", "parallel", "retries"],
            "hitl": ["hitl_node", "interactive_approver"],
        }


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
            "budgets": config.budgets or None,
            "circuit_breaker": config.circuit_breaker,
            "failure_threshold": config.failure_threshold,
            "cooldown_seconds": config.cooldown_seconds,
            "request_log": config.request_log,
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
            "semantic_cache": "optional via create_gateway(semantic_cache=True)",
            "endpoints": self.endpoints(),
        }


class _GraphNamespace(_Namespace):
    def create(self, **options: Any) -> Any:
        """Create a KnowledgeGraph (GraphRAG pipeline: ingest → triples → cited answers)."""
        from aire.graph.pipeline import KnowledgeGraph

        return KnowledgeGraph(self._rt(), **options)

    def store(self, spec: str = "sqlite:memory", **options: Any) -> Any:
        """Resolve a graph store by ``provider:name`` (embedded sqlite by default)."""
        from aire.core.types import Ref

        ref = Ref.parse(spec)
        runtime = self._rt()
        if not runtime.graph_stores.has(ref.provider):
            from aire.graph.store import register

            register(runtime)
            if ref.provider == "neo4j":
                from aire.graph.neo4j_store import register as register_neo4j

                register_neo4j(runtime)
        return runtime.graph_stores.create(ref.provider, name=ref.name, runtime=runtime, **options)

    def communities(self, entities: Any, relations: Any, **options: Any) -> Any:
        from aire.graph.community import detect_communities

        return detect_communities(entities, relations, **options)

    def describe(self) -> dict[str, Any]:
        runtime = self._rt()
        return {
            "kind": "graph",
            "stores": runtime.graph_stores.names() or ["sqlite", "neo4j"],
            "extractors": ["lexical", "model"],
            "communities": ["label_propagation"],
        }


class _MemoryNamespace(_Namespace):
    def create(
        self,
        *,
        path: str | Path | None = None,
        embedder: Any = None,
        window: int = 200,
        **options: Any,
    ) -> Any:
        """Create long-term memory (episodic + semantic, optional persistence)."""
        from aire.memory.store import LongTermMemory

        return LongTermMemory(embedder=embedder, path=path, window=window, **options)

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "memory",
            "types": ["episodic", "semantic", "procedural"],
            "agent_usage": "AI.agents.create(memory=AI.memory.create(...))",
        }


class _McpNamespace(_Namespace):
    def server(self, tools: list[Tool] | None = None, **options: Any) -> Any:
        """Build an MCP server exposing the given tools (or builtins + registered)."""
        from aire.mcp.server import MCPServer, default_server

        if tools is None:
            return default_server()
        return MCPServer(tools, **options)

    async def connect(self, command: list[str], **options: Any) -> Any:
        """Connect to an MCP server subprocess; returns a connected MCPClient."""
        from aire.mcp.client import MCPClient

        return await MCPClient(command, **options).connect()

    def connect_sync(self, command: list[str], **options: Any) -> Any:
        return run_sync(self.connect(command, **options))

    def describe(self) -> dict[str, Any]:
        from aire.mcp.knowledge import builtin_prompts, builtin_resources
        from aire.mcp.protocol import PROTOCOL_VERSION

        return {
            "kind": "mcp",
            "protocol": PROTOCOL_VERSION,
            "transport": "stdio (newline-delimited JSON-RPC 2.0)",
            "cli": "aire mcp-serve",
            "resources": [r.uri for r in builtin_resources()],
            "prompts": [p.name for p in builtin_prompts()],
        }


class _MLNamespace(_Namespace):
    def create(self, spec: str = "simple:centroid", **options: Any) -> Any:
        """Create an estimator from a ``backend:name`` ref.

        Backends: ``simple:*``, ``sklearn:*``, ``torch:*``, ``keras:*``,
        ``xgboost:*``, ``lightgbm:*``, ``catboost:*``. Options flow to the estimator.
        """
        from aire.ml.factory import create_estimator

        return create_estimator(spec, runtime=self._rt(), **options)

    def pipeline(
        self,
        steps: list[tuple[str, Any]],
        *,
        target: str = "label",
    ) -> Any:
        """Build a :class:`~aire.ml.pipeline.Pipeline` of transforms → estimator."""
        from aire.ml.pipeline import Pipeline

        return Pipeline(steps=steps, target=target)

    def transform(self, spec: str = "native:standard_scaler", **options: Any) -> Any:
        """Create a :class:`~aire.ml.transform.Transform` (``native:*`` / ``sklearn:*``)."""
        from aire.ml.transform import create_transform

        return create_transform(spec, **options)

    def column_transformer(
        self,
        transformers: list[tuple[str, Any, list[Any]]],
        *,
        remainder: str = "drop",
    ) -> Any:
        """Column-wise transform composition (sklearn-style)."""
        from aire.ml.compose import ColumnTransformer

        return ColumnTransformer(transformers, remainder=remainder)

    def feature_union(self, transformer_list: list[tuple[str, Any]]) -> Any:
        """Concatenate outputs of multiple transforms."""
        from aire.ml.compose import FeatureUnion

        return FeatureUnion(transformer_list)

    def scorers(self) -> dict[str, str]:
        """Named scorers available for CV / search (name → direction)."""
        from aire.ml.scoring import scorers

        return scorers()

    async def fit(self, spec: str, dataset: Any, *, target: str = "label", **options: Any) -> Any:
        """Create + fit an estimator in one call; returns the fitted estimator."""
        estimator = self.create(spec, **options)
        await estimator.fit(dataset, target=target)
        return estimator

    def fit_sync(self, spec: str, dataset: Any, *, target: str = "label", **options: Any) -> Any:
        return run_sync(self.fit(spec, dataset, target=target, **options))

    async def train(
        self,
        spec: str,
        dataset: Any,
        *,
        target: str = "label",
        transforms: list[str | Any] | None = None,
        **options: Any,
    ) -> Any:
        """Fit an estimator, optionally behind a transform pipeline.

        ``transforms`` are prepended as named steps before the final estimator.
        """
        if transforms:
            steps: list[tuple[str, Any]] = [
                (f"t{i}", t) for i, t in enumerate(transforms)
            ]
            steps.append(("estimator", spec if not options else self.create(spec, **options)))
            pipe = self.pipeline(steps, target=target)
            await pipe.fit(dataset, target=target)
            return pipe
        return await self.fit(spec, dataset, target=target, **options)

    def train_sync(self, spec: str, dataset: Any, **options: Any) -> Any:
        return run_sync(self.train(spec, dataset, **options))

    def catalog(self) -> dict[str, list[str]]:
        """Catalog of creatable estimators by backend (agent-readable)."""
        from aire.ml.native import NATIVE_ESTIMATORS
        from aire.ml.pandas_bridge import available_backends
        from aire.ml.sklearn_adapter import _SKLEARN_NAMES

        backends = available_backends()
        out: dict[str, list[str]] = {
            "simple": [f"simple:{n}" for n in sorted(NATIVE_ESTIMATORS)],
        }
        if backends.get("sklearn"):
            out["sklearn"] = [f"sklearn:{n}" for n in sorted(_SKLEARN_NAMES)]
        out["torch"] = ["torch:mlp"]
        out["keras"] = ["keras:mlp"]
        out["xgboost"] = ["xgboost:classifier", "xgboost:regressor"]
        out["lightgbm"] = ["lightgbm:classifier", "lightgbm:regressor"]
        out["catboost"] = ["catboost:classifier", "catboost:regressor"]
        return out

    def transforms_catalog(self) -> dict[str, list[str]]:
        """Catalog of creatable transforms by provider."""
        from aire.ml.pandas_bridge import available_backends
        from aire.ml.sklearn_adapter import _SKLEARN_TRANSFORMS
        from aire.ml.transform import NATIVE_TRANSFORMS

        out: dict[str, list[str]] = {
            "native": [
                f"native:{n}"
                for n in sorted(
                    [*NATIVE_TRANSFORMS, "column_transformer", "feature_union"]
                )
            ],
        }
        if available_backends().get("sklearn"):
            out["sklearn"] = [f"sklearn:{n}" for n in sorted(_SKLEARN_TRANSFORMS)]
        return out

    def backends(self) -> dict[str, bool]:
        """Which ML backends are importable right now (native always true)."""
        from aire.ml.pandas_bridge import available_backends

        return available_backends()

    async def cross_validate(
        self,
        spec: str,
        dataset: Any,
        *,
        k: int = 5,
        target: str = "label",
        seed: int = 0,
        scoring: str | None = None,
        stratified: bool = False,
        **options: Any,
    ) -> Any:
        """K-fold CV for an estimator ref; returns :class:`~aire.ml.metrics.CVReport`."""
        from aire.ml.metrics import cross_validate as _cv

        def factory() -> Any:
            return self.create(spec, **options)

        return await _cv(
            factory,
            dataset,
            k=k,
            target=target,
            seed=seed,
            scoring=scoring,
            stratified=stratified,
        )

    def cross_validate_sync(self, spec: str, dataset: Any, **options: Any) -> Any:
        return run_sync(self.cross_validate(spec, dataset, **options))

    async def grid_search(
        self,
        spec: str,
        dataset: Any,
        param_grid: dict[str, list[Any]],
        *,
        k: int = 3,
        target: str = "label",
        scoring: str | None = None,
        direction: str = "maximize",
        seed: int = 0,
        **fixed: Any,
    ) -> Any:
        """Grid search over ``param_grid`` with inner k-fold CV."""
        from aire.ml.metrics import grid_search as _gs

        provider_name = spec

        def factory(**params: Any) -> Any:
            merged = {**fixed, **params}
            return self.create(provider_name, **merged)

        return await _gs(
            factory,
            dataset,
            param_grid,
            k=k,
            target=target,
            scoring=scoring,
            direction=direction,
            seed=seed,
        )

    def grid_search_sync(
        self, spec: str, dataset: Any, param_grid: dict[str, list[Any]], **options: Any
    ) -> Any:
        return run_sync(self.grid_search(spec, dataset, param_grid, **options))

    async def random_search(
        self,
        spec: str,
        dataset: Any,
        param_distributions: dict[str, list[Any]],
        *,
        n_iter: int = 10,
        k: int = 3,
        target: str = "label",
        scoring: str | None = None,
        direction: str = "maximize",
        seed: int = 0,
        **fixed: Any,
    ) -> Any:
        """Random search over discrete ``param_distributions`` with inner k-fold CV."""
        from aire.ml.metrics import random_search as _rs

        provider_name = spec

        def factory(**params: Any) -> Any:
            merged = {**fixed, **params}
            return self.create(provider_name, **merged)

        return await _rs(
            factory,
            dataset,
            param_distributions,
            n_iter=n_iter,
            k=k,
            target=target,
            scoring=scoring,
            direction=direction,
            seed=seed,
        )

    def random_search_sync(
        self,
        spec: str,
        dataset: Any,
        param_distributions: dict[str, list[Any]],
        **options: Any,
    ) -> Any:
        return run_sync(self.random_search(spec, dataset, param_distributions, **options))

    def to_frame(self, dataset: Any, **options: Any) -> Any:
        """Dataset → pandas DataFrame (requires pandas)."""
        from aire.ml.pandas_bridge import dataset_to_frame

        return dataset_to_frame(dataset, **options)

    def from_frame(self, frame: Any, **options: Any) -> Any:
        """pandas DataFrame → Dataset (requires pandas)."""
        from aire.ml.pandas_bridge import frame_to_dataset

        return frame_to_dataset(frame, **options)

    def to_polars(self, dataset: Any, **options: Any) -> Any:
        """Dataset → polars DataFrame (requires polars)."""
        from aire.ml.polars_bridge import dataset_to_frame

        return dataset_to_frame(dataset, **options)

    def from_polars(self, frame: Any, **options: Any) -> Any:
        """polars DataFrame → Dataset (requires polars)."""
        from aire.ml.polars_bridge import frame_to_dataset

        return frame_to_dataset(frame, **options)

    def describe(self) -> dict[str, Any]:
        from aire.ml.pandas_bridge import available_backends
        from aire.ml.scoring import scorers

        return {
            "kind": "ml",
            "contract": (
                "Pipeline(transforms→estimator) | Estimator: "
                "fit → predict → evaluate → cross_validate → "
                "grid_search/random_search → save/load"
            ),
            "backends": available_backends(),
            "estimators": self.catalog(),
            "transforms": self.transforms_catalog(),
            "scorers": scorers(),
            "tasks": ["classification", "regression", "clustering", "multi_label"],
            "feature_convention": "record.metadata['features'] → numeric metadata → text-derived",
            "metrics": [
                "accuracy",
                "precision/recall/f1 (macro+micro+per-class)",
                "balanced_accuracy",
                "roc_auc",
                "log_loss",
                "mae",
                "rmse",
                "r2",
                "confusion_matrix",
                "permutation_importance",
            ],
            "selection": [
                "cross_validate (stratified=)",
                "grid_search",
                "random_search",
            ],
            "orchestration": [
                "AI.ml.pipeline",
                "AI.ml.transform",
                "AI.ml.column_transformer",
                "AI.ml.feature_union",
                "AI.ml.train",
                "callbacks (EarlyStopping, History, keras zoo)",
                "torch: amp/compile/grad_clip/DataLoader/val_split",
                "keras: compile(metrics)/callbacks/validation_split",
            ],
            "arch": "AI.ml.arch — composable attention/ffn/norm/residual blocks",
            "optim": "AI.ml.optim — sgd/adam/adamw/rmsprop/adagrad",
            "loss": "AI.ml.loss — cross_entropy/mse/huber/moe_load_balance/...",
        }

    @property
    def arch(self) -> Any:
        return _ArchNamespace()

    @property
    def optim(self) -> Any:
        return _OptimNamespace()

    @property
    def loss(self) -> Any:
        return _LossNamespace()


class _ArchNamespace(_Namespace):
    """Composable neural architecture blocks — assemble any stack from parts."""

    def attention(self, kind: str, **options: Any) -> Any:
        from aire.ml.arch import attention as _attention

        return _attention(kind, **options)

    def ffn(self, kind: str, **options: Any) -> Any:
        from aire.ml.arch import ffn as _ffn

        return _ffn(kind, **options)

    def norm(self, kind: str, **options: Any) -> Any:
        from aire.ml.arch import norm as _norm

        return _norm(kind, **options)

    def residual(self, kind: str, **options: Any) -> Any:
        from aire.ml.arch import residual as _residual

        return _residual(kind, **options)

    def block(self, **options: Any) -> Any:
        from aire.ml.arch import block as _block

        return _block(**options)

    def compose(self, layers: list[Any], **options: Any) -> Any:
        from aire.ml.arch import compose as _compose

        return _compose(layers, **options)

    def create(self, name: str, **options: Any) -> Any:
        from aire.ml.arch import create as _create

        return _create(name, **options)

    def available(self) -> dict[str, list[str]]:
        from aire.ml.arch import available

        return available()

    def register_attention(self, name: str, factory: Any | None = None, **options: Any) -> Any:
        from aire.ml.arch import register_attention

        return register_attention(name, factory, **options)

    def register_ffn(self, name: str, factory: Any | None = None, **options: Any) -> Any:
        from aire.ml.arch import register_ffn

        return register_ffn(name, factory, **options)

    def register_architecture(self, name: str, factory: Any | None = None, **options: Any) -> Any:
        from aire.ml.arch import register_architecture

        return register_architecture(name, factory, **options)

    def describe(self) -> dict[str, Any]:
        from aire.ml.arch import describe

        return describe()


class _OptimNamespace(_Namespace):
    def create(self, name: str, params: Any, **options: Any) -> Any:
        from aire.ml import optim

        return optim.create(name, params, **options)

    def available(self) -> list[str]:
        from aire.ml import optim

        return optim.names()

    def describe(self) -> dict[str, Any]:
        from aire.ml import optim

        return optim.describe()


class _LossNamespace(_Namespace):
    def create(self, name: str, **options: Any) -> Any:
        from aire.ml import loss

        return loss.create(name, **options)

    def available(self) -> list[str]:
        from aire.ml import loss

        return loss.names()

    def describe(self) -> dict[str, Any]:
        from aire.ml import loss

        return loss.describe()


class _TrainingNamespace(_Namespace):
    def create(self, step: Any, **options: Any) -> Any:
        from aire.training.trainer import FunctionTrainer, TrainingConfig

        config = options.pop("config", None) or TrainingConfig(**options)
        return FunctionTrainer(step, config)

    def lora(self, model_name: str = "gpt2", **options: Any) -> Any:
        from aire.training.lora import create_lora

        return create_lora(model_name, **options)

    def lm(self, architecture: Any | None = None, **options: Any) -> Any:
        from aire.training.lm_trainer import create_lm_trainer

        return create_lm_trainer(architecture, **options)

    def quantize(self, model_name: str = "gpt2", **options: Any) -> Any:
        from aire.training.quantize import create_quantizer

        return create_quantizer(model_name, **options)

    def distill(self, **options: Any) -> Any:
        from aire.training.distill import create_distiller

        return create_distiller(**options)

    def distill_trainer(self, **options: Any) -> Any:
        from aire.training.distill import DistillTrainer

        return DistillTrainer(**options)

    async def hpo(
        self,
        objective: Any,
        space: Any,
        **options: Any,
    ) -> Any:
        from aire.training.hpo import random_search

        return await random_search(objective, space, **options)

    def hpo_sync(self, objective: Any, space: Any, **options: Any) -> Any:
        return run_sync(self.hpo(objective, space, **options))

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "training",
            "trainers": ["function", "lora", "lm", "quantize", "distill"],
            "hpo": ["random", "optuna (optional)"],
        }


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

    def policy(self, rules: list[Any] | None = None, **options: Any) -> Any:
        from aire.safety.policy import PolicyEngine, PolicyRule, default_engine

        if rules is None and not options:
            return default_engine()
        parsed = [
            r if isinstance(r, PolicyRule) else PolicyRule.model_validate(r) for r in (rules or [])
        ]
        return PolicyEngine(parsed, **options)

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "safety",
            "guardrails": ["pii", "injection", "secret"],
            "policy": ["ApprovalPolicy", "PolicyEngine"],
        }


class _SkillsNamespace(_Namespace):
    def registry(self) -> Any:
        from aire.agents.skills import default_skills

        return default_skills()

    def get(self, name: str) -> Any:
        return self.registry().get(name)

    def load(self, path: str | Any) -> Any:
        return self.registry().load_dir(path)

    def register(self, skill: Any, **options: Any) -> Any:
        return self.registry().register(skill, **options)

    def describe(self) -> dict[str, Any]:
        return dict(self.registry().describe())


class _ScheduleNamespace(_Namespace):
    def create(self) -> Any:
        from aire.schedule import Scheduler

        return Scheduler()

    def every(self, interval_seconds: float, workflow: Any, **options: Any) -> Any:
        sched = self.create()
        return sched.every(interval_seconds, workflow, **options)

    def describe(self) -> dict[str, Any]:
        from aire.schedule import describe

        return describe()


class _WorkersNamespace(_Namespace):
    def create(self, kind: str = "in_process", **options: Any) -> Any:
        from aire.workers import create_worker

        return create_worker(kind, **options)

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "workers",
            "backends": ["in_process", "file"],
        }


class _RecipesNamespace(_Namespace):
    def create(self, name: str, **options: Any) -> Any:
        from aire.recipes import recipe

        return recipe(name, **options)

    def describe(self) -> dict[str, Any]:
        from aire.recipes import describe

        return describe()


class _ProjectNamespace(_Namespace):
    def lock(self, project: str, **refs: Any) -> Any:
        from aire.project.lock import create_lock

        return create_lock(project, **refs)

    def load_lock(self, path: str | Any | None = None) -> Any:
        from aire.project.lock import load_lock

        return load_lock(path)

    def write_lock(self, lock: Any, path: str | Any | None = None) -> Any:
        from aire.project.lock import write_lock

        return write_lock(lock, path)

    def apply(self, lock: Any, settings: Any | None = None) -> Any:
        """Apply lock pins to settings (defaults to current runtime settings)."""
        from aire.project.lock import ProjectLock, apply_lock, load_lock

        loaded = lock if isinstance(lock, ProjectLock) else load_lock(lock)
        base = settings or self._rt().settings
        return apply_lock(base, loaded)

    def describe(self) -> dict[str, Any]:
        from aire.project.lock import describe

        return describe()


class _VisionNamespace(_Namespace):
    def pipeline(self, model: str | Model | None = None, **options: Any) -> Any:
        from aire.vision.pipelines import VisionPipeline

        resolved = self._resolve_model(model)
        return VisionPipeline(resolved, **options)

    def generate(self, model: str | Model | None = None, **options: Any) -> Any:
        from aire.vision.pipelines import ImageGenerationPipeline

        return ImageGenerationPipeline(self._resolve_model(model), **options)

    def video(self, model: str | Model | None = None, **options: Any) -> Any:
        from aire.vision.video import VideoPipeline

        if model is None:
            return VideoPipeline(None, **options)
        return VideoPipeline(self._resolve_model(model), **options)

    def _resolve_model(self, model: str | Model | None) -> Model:
        if isinstance(model, Model):
            return model
        return run_sync(ModelRegistry(self._rt()).use(model or self._rt().settings.model.ref))

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "vision",
            "pipelines": ["VisionPipeline", "ImageGenerationPipeline", "VideoPipeline"],
        }


class _AudioNamespace(_Namespace):
    def pipeline(self, model: str | Model | None = None, **options: Any) -> Any:
        from aire.audio.pipelines import AudioPipeline

        if isinstance(model, Model):
            resolved = model
        else:
            resolved = run_sync(
                ModelRegistry(self._rt()).use(model or self._rt().settings.model.ref)
            )
        return AudioPipeline(resolved, **options)

    def voice(
        self,
        agent: Any,
        *,
        asr: Any = None,
        tts: Any = None,
    ) -> Any:
        from aire.audio.voice import VoiceAgent

        return VoiceAgent(agent, asr=asr, tts=tts)

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "audio",
            "pipelines": ["AudioPipeline", "VoiceAgent", "TTSBackend"],
        }


class _DocsNamespace(_Namespace):
    def load_pdf(self, path: str | Any, **options: Any) -> Any:
        from aire.docs.pdf import load_pdf

        return load_pdf(path, **options)

    def to_dataset(self, path: str | Any, **options: Any) -> Any:
        from aire.docs.pdf import pdf_to_dataset

        return pdf_to_dataset(path, **options)

    def describe(self) -> dict[str, Any]:
        from aire.docs.pdf import describe

        return describe()


class AI:
    """Unified entry point to the aire library.

    Namespaces operate on the process-wide default runtime; use
    :meth:`AI.configure` to replace it (e.g. with a specific config file).
    """

    models = _ModelsNamespace()
    data = _DataNamespace()
    rag = _RagNamespace()
    graph = _GraphNamespace()
    memory = _MemoryNamespace()
    mcp = _McpNamespace()
    agents = _AgentsNamespace()
    observe = _ObserveNamespace()
    deploy = _DeployNamespace()
    gateway = _GatewayNamespace()
    workflows = _WorkflowNamespace()
    training = _TrainingNamespace()
    ml = _MLNamespace()
    safety = _SafetyNamespace()
    skills = _SkillsNamespace()
    schedule = _ScheduleNamespace()
    workers = _WorkersNamespace()
    recipes = _RecipesNamespace()
    locks = _ProjectNamespace()
    vision = _VisionNamespace()
    audio = _AudioNamespace()
    docs = _DocsNamespace()

    # -- runtime -----------------------------------------------------------------

    @classmethod
    def runtime(cls) -> Runtime:
        return default_runtime()

    @classmethod
    def configure(
        cls,
        settings: Settings | None = None,
        *,
        lock: str | Path | Any | None = None,
        **overrides: Any,
    ) -> Runtime:
        """Rebuild the default runtime from explicit settings or overrides.

        Pass ``lock=`` (path or :class:`~aire.project.lock.ProjectLock`) to pin
        model/embedder refs from ``aire.lock`` into settings before build.
        """
        global _default_runtime
        settings = settings or Settings.load(overrides=overrides or None)
        if lock is not None:
            from aire.project.lock import ProjectLock, apply_lock, load_lock

            loaded = lock if isinstance(lock, ProjectLock) else load_lock(lock)
            settings = apply_lock(settings, loaded)
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

    # -- recipes ------------------------------------------------------------------------

    @classmethod
    def recipe(cls, name: str, **options: Any) -> Any:
        """One-call scaffold: ``rag`` | ``agent`` | ``finetune`` | ``gateway``."""
        return cls.recipes.create(name, **options)

    # -- topologies ---------------------------------------------------------------------

    @classmethod
    def topologies(cls) -> Any:
        """Multi-agent topology helpers (swarm, debate, auction, blackboard)."""
        from aire.agents import topologies

        return topologies

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

    # -- ui -----------------------------------------------------------------------------

    @classmethod
    def ui(cls, **options: Any) -> Any:
        """Minimal local FastAPI UI for traces/costs (requires aire[serve])."""
        from aire.ui import create_ui_app

        return create_ui_app(runtime=default_runtime(), **options)

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
                "graph",
                "memory",
                "mcp",
                "agents",
                "skills",
                "workflows",
                "training",
                "ml",
                "observe",
                "deploy",
                "gateway",
                "safety",
                "schedule",
                "workers",
                "recipes",
                "locks",
                "vision",
                "audio",
                "docs",
                "synthetic",
            ],
        }

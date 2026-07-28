"""The fluent project builder — the flagship aire experience.

    assistant = (
        AI.project("knowledge_assistant")
        .documents("./docs")
        .model("openai:gpt-4o-mini")
        .vector_store("local")
        .citations(True)
    )
    assistant.index()
    answer = assistant.ask("What does the documentation say about authentication?")
    assistant.evaluate("./evals.jsonl")
    app = assistant.deploy()

Every method is chainable; ``index``/``ask``/``evaluate``/``deploy`` execute.
Async variants (``index_async`` etc.) exist for event-loop environments.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from aire.core.runtime import Runtime
from aire.models.base import run_sync

if TYPE_CHECKING:
    from aire.data.chunking import Chunker
    from aire.evaluation.types import EvalReport
    from aire.rag.pipeline import Knowledge
    from aire.rag.store import VectorStore
    from aire.rag.types import Answer, IndexReport


class Assistant:
    """A configured knowledge assistant project (RAG vertical slice)."""

    def __init__(self, name: str, runtime: Runtime) -> None:
        self.name = name
        self.runtime = runtime
        self._sources: list[Any] = []
        self._model_spec: str | None = None
        self._embedder_spec: str | None = None
        self._store_spec: str | None = None
        self._store_instance: VectorStore | None = None
        self._chunker: str | Chunker = "recursive"
        self._reranker: str = "lexical"
        self._citations = True
        self._hybrid = True
        self._knowledge: Knowledge | None = None

    # -- configuration (chainable) -------------------------------------------------

    def documents(self, source: Any, *more: Any) -> Assistant:
        """Add document sources (paths, directories, URLs, lists)."""
        self._sources.extend([source, *more])
        return self

    def model(self, spec: str) -> Assistant:
        """Set the answering model (``provider:name``)."""
        self._model_spec = spec
        return self

    def embedder(self, spec: str) -> Assistant:
        """Set the embedding model (``provider:name``)."""
        self._embedder_spec = spec
        return self

    def vector_store(self, spec: str, **options: Any) -> Assistant:
        """Set the vector store (``provider:name``); ``local`` needs no services."""
        self._store_spec = spec
        if spec.split(":", 1)[0] == "local":
            from aire.rag.store import LocalVectorStore

            self._store_instance = LocalVectorStore(**options)
        else:
            from aire.ai import _RagNamespace

            self._store_instance = _RagNamespace(self.runtime).vector_store(spec, **options)
        return self

    def chunker(self, name: str, **options: Any) -> Assistant:
        from aire.data.chunking import get_chunker

        self._chunker = get_chunker(name, **options)
        return self

    def reranker(self, name: str) -> Assistant:
        self._reranker = name
        return self

    def citations(self, enabled: bool = True) -> Assistant:
        self._citations = enabled
        return self

    def hybrid(self, enabled: bool = True) -> Assistant:
        self._hybrid = enabled
        return self

    # -- build ------------------------------------------------------------------------

    def knowledge(self) -> Knowledge:
        """Materialize the underlying Knowledge pipeline."""
        if self._knowledge is None:
            from aire.rag.pipeline import Knowledge

            self._knowledge = Knowledge(
                self.runtime,
                store=self._store_instance,
                chunker=self._chunker,
                reranker=self._reranker,
                hybrid=self._hybrid,
            )
        return self._knowledge

    async def _resolve_embedder(self) -> None:
        knowledge = self.knowledge()
        if knowledge.embedder is None:
            from aire.models.registry import ModelRegistry

            knowledge.embedder = await ModelRegistry(self.runtime).embedder(self._embedder_spec)

    # -- execution ------------------------------------------------------------------------

    async def index_async(self) -> IndexReport:
        """Ingest all configured sources (chunk → embed → store)."""
        if not self._sources:
            from aire.core.errors import ConfigurationError

            raise ConfigurationError(
                "no documents configured; call .documents(source) first",
                code="project.no_documents",
            )
        await self._resolve_embedder()
        knowledge = self.knowledge()
        from aire.rag.types import IndexReport

        total = IndexReport(documents=0, chunks=0, store="", embedder="")
        for source in self._sources:
            report = await knowledge.ingest(source)
            total.documents += report.documents
            total.chunks += report.chunks
            total.store = report.store
            total.embedder = report.embedder
            total.duration_ms += report.duration_ms
        return total

    def index(self) -> IndexReport:
        return run_sync(self.index_async())

    async def ask_async(self, question: str, *, k: int = 5, **kwargs: Any) -> Answer:
        """Ask a grounded question; returns an Answer with citations."""
        await self._resolve_embedder()
        model = self._model_spec or self.runtime.settings.model.ref
        return await self.knowledge().ask(
            question, model=model, k=k, citations=self._citations, **kwargs
        )

    def ask(self, question: str, **kwargs: Any) -> Answer:
        return run_sync(self.ask_async(question, **kwargs))

    async def evaluate_async(self, dataset: Any, *, metrics: list[str] | None = None) -> EvalReport:
        """Evaluate the assistant against a QA dataset."""
        from aire.evaluation.runner import Evaluator

        await self._resolve_embedder()
        model = self._model_spec or self.runtime.settings.model.ref

        async def _target(question: str) -> Any:
            return await self.knowledge().ask(question, model=model, citations=self._citations)

        return await Evaluator(name=f"{self.name}-eval").run(
            _target, dataset, metrics=metrics or ["accuracy", "groundedness"]
        )

    def evaluate(self, dataset: Any, **kwargs: Any) -> EvalReport:
        return run_sync(self.evaluate_async(dataset, **kwargs))

    def deploy(self, **options: Any) -> Any:
        """Wrap the assistant in a production FastAPI app (requires aire[serve])."""
        from aire.deployment.fastapi_app import create_app

        return create_app(self.knowledge(), title=f"{self.name} API", **options)

    def deploy_artifacts(self, directory: str | Path) -> Any:
        """Generate Dockerfile, entrypoint and env template for deployment."""
        from aire.deployment.artifacts import generate_artifacts

        return generate_artifacts(directory, project=self.name)

    # -- introspection -----------------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "assistant",
            "name": self.name,
            "sources": [str(s) for s in self._sources],
            "model": self._model_spec or self.runtime.settings.model.ref,
            "embedder": self._embedder_spec or self.runtime.settings.model.embedder,
            "store": self._store_spec or "local",
            "citations": self._citations,
            "knowledge": self._knowledge.describe() if self._knowledge else None,
        }

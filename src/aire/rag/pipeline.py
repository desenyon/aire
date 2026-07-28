"""The Knowledge pipeline: ingest → index → retrieve → grounded answer.

This is the reference RAG implementation and the backbone of the ``AI.project``
vertical slice. Every stage is replaceable: chunker, embedder, store,
reranker, prompt template, and the answering model.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aire.core.errors import RetrievalError
from aire.data.chunking import Chunker, get_chunker
from aire.data.loaders import load as load_dataset
from aire.models.base import EmbeddingModel, Model
from aire.rag.rerank import Reranker, get_reranker
from aire.rag.retriever import Retriever
from aire.rag.store import LocalVectorStore, VectorStore
from aire.rag.types import Answer, Chunk, Citation, Document, IndexReport, ScoredChunk

if TYPE_CHECKING:
    from aire.core.runtime import Runtime

DEFAULT_PROMPT = (
    "Answer the question using only the context below. "
    "Cite sources as [1], [2], ... matching the context entries. "
    "If the context does not contain the answer, say so.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
)


class Knowledge:
    """A self-contained retrieval-augmented generation pipeline."""

    def __init__(
        self,
        runtime: Runtime,
        *,
        store: VectorStore | None = None,
        embedder: EmbeddingModel | None = None,
        chunker: Chunker | str = "recursive",
        reranker: Reranker | str = "lexical",
        hybrid: bool = True,
        prompt_template: str = DEFAULT_PROMPT,
    ) -> None:
        self.runtime = runtime
        self.store = store or LocalVectorStore()
        self.embedder = embedder
        self.chunker = get_chunker(chunker) if isinstance(chunker, str) else chunker
        self.reranker = get_reranker(reranker) if isinstance(reranker, str) else reranker
        self.hybrid = hybrid
        self.prompt_template = prompt_template
        self._retriever: Retriever | None = None

    # -- ingestion -----------------------------------------------------------------

    async def _embedder(self) -> EmbeddingModel:
        if self.embedder is None:
            from aire.models.registry import ModelRegistry

            self.embedder = await ModelRegistry(self.runtime).embedder()
        return self.embedder

    async def ingest(
        self,
        source: str | Path | list[str] | list[dict[str, Any]] | list[Document],
        *,
        chunker: Chunker | str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> IndexReport:
        """Load, chunk, embed and index a source. Returns an IndexReport."""
        started = time.perf_counter()
        if source and isinstance(source, list) and isinstance(source[0], Document):
            documents = list(source)
        else:
            dataset = load_dataset(source)  # type: ignore[arg-type]
            documents = [Document(text=r.text, metadata=r.metadata) for r in dataset]
        if metadata:
            for doc in documents:
                doc.metadata.update(metadata)
        chunks = await self.ingest_documents(documents, chunker=chunker)
        report = IndexReport(
            documents=len(documents),
            chunks=chunks,
            store=type(self.store).__name__,
            embedder=(await self._embedder()).name,
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )
        self.runtime.events.emit("rag.ingested", report.model_dump(mode="json"), source="rag")
        return report

    async def ingest_documents(
        self, documents: list[Document], *, chunker: Chunker | str | None = None
    ) -> int:
        """Chunk, embed and upsert pre-built documents. Returns chunk count."""
        active_chunker = (
            self.chunker
            if chunker is None
            else (get_chunker(chunker) if isinstance(chunker, str) else chunker)
        )
        chunks: list[Chunk] = []
        for doc in documents:
            for piece in active_chunker.chunk(doc.text):
                chunks.append(
                    Chunk(
                        document_id=doc.id,
                        text=piece.text,
                        index=piece.index,
                        metadata={**doc.metadata, "start": piece.start, "end": piece.end},
                    )
                )
        if not chunks:
            raise RetrievalError(
                "ingestion produced zero chunks",
                code="rag.empty_ingestion",
                context={"documents": len(documents)},
            )
        embedder = await self._embedder()
        vectors = await embedder.embed_texts([c.text for c in chunks])
        for chunk, vector in zip(chunks, vectors, strict=True):
            chunk.embedding = vector
        await self.store.upsert(chunks)
        self._retriever = None  # store contents changed; rebuild lazily
        return len(chunks)

    # -- retrieval --------------------------------------------------------------------

    async def retriever(self) -> Retriever:
        if self._retriever is None:
            self._retriever = Retriever(self.store, await self._embedder(), hybrid=self.hybrid)
        return self._retriever

    async def retrieve(
        self,
        query: str,
        *,
        k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        """Retrieve and rerank relevant chunks for a query."""
        retriever = await self.retriever()
        hits = await retriever.retrieve(query, k=max(k * 2, 6), filter=filter)
        return await self.reranker.rerank(query, hits, k=k)

    # -- answering ----------------------------------------------------------------------

    async def ask(
        self,
        question: str,
        *,
        model: Model | str | None = None,
        k: int = 5,
        citations: bool = True,
        filter: dict[str, Any] | None = None,
        template: str | None = None,
        **generate_kwargs: Any,
    ) -> Answer:
        """Retrieve context and generate a grounded answer with citations."""
        hits = await self.retrieve(question, k=k, filter=filter)
        context = "\n\n".join(f"[{i + 1}] {hit.chunk.text}" for i, hit in enumerate(hits))
        prompt = (template or self.prompt_template).format(context=context, question=question)
        resolved = await self._resolve_model(model)
        from aire.models.types import GenerationRequest

        request = GenerationRequest.of(prompt, **generate_kwargs)
        tracer = self.runtime.tracer
        if tracer is not None:
            async with tracer.aspan("rag.ask", attributes={"question": question, "k": k}):
                gen = await resolved.generate(request)
        else:
            gen = await resolved.generate(request)

        answer = Answer(
            text=gen.text,
            model=resolved.info.ref,
            retrieved=len(hits),
            usage=gen.usage,
        )
        if citations:
            answer.citations = [
                Citation(
                    source=str(hit.chunk.metadata.get("source", hit.chunk.document_id)),
                    chunk_id=hit.chunk.id,
                    excerpt=hit.chunk.text[:200],
                    score=hit.score,
                    metadata={"index": hit.chunk.index},
                )
                for hit in hits
            ]
        self.runtime.events.emit(
            "rag.answered",
            {"question": question, "retrieved": len(hits), "model": resolved.info.ref},
            source="rag",
        )
        return answer

    async def _resolve_model(self, model: Model | str | None) -> Model:
        if isinstance(model, Model):
            return model
        from aire.models.registry import ModelRegistry

        spec = model or self.runtime.settings.model.ref
        return await ModelRegistry(self.runtime).use(spec)

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "knowledge",
            "store": self.store.describe().model_dump(mode="json"),
            "chunker": type(self.chunker).__name__,
            "reranker": type(self.reranker).__name__,
            "hybrid": self.hybrid,
        }

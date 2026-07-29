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
    from aire.safety.guardrails import Guardrail, GuardrailChain

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
        rewriter: str | Any | None = None,
        compressor: str | Any | None = "truncate",
        max_context_chars: int = 4000,
        default_acl: dict[str, Any] | None = None,
        guardrails: GuardrailChain | list[Guardrail] | bool | None = None,
    ) -> None:
        self.runtime = runtime
        self.store = store or LocalVectorStore()
        self.embedder = embedder
        self.chunker = get_chunker(chunker) if isinstance(chunker, str) else chunker
        self.reranker = get_reranker(reranker) if isinstance(reranker, str) else reranker
        self.hybrid = hybrid
        self.prompt_template = prompt_template
        self.rewriter_spec = rewriter
        self.compressor_spec = compressor
        self.max_context_chars = max_context_chars
        self.default_acl = default_acl
        self._retriever: Retriever | None = None
        self._rewriter: Any | None = None
        self._compressor: Any | None = None
        from aire.safety.guardrails import resolve_guardrails

        self.guardrails = resolve_guardrails(
            guardrails, safety=runtime.settings.safety
        )
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

    async def reindex_document(
        self,
        document: Document,
        *,
        chunker: Chunker | str | None = None,
    ) -> int:
        """Incremental update: delete prior chunks for ``document.id``, then ingest."""
        if hasattr(self.store, "delete_by_document"):
            deleted = await self.store.delete_by_document(document.id)
            _ = deleted
        else:
            # fallback: keyword-scan delete
            hits = await self.store.search_text("", k=1_000_000)
            ids = [h.chunk.id for h in hits if h.chunk.document_id == document.id]
            if ids:
                await self.store.delete(ids)
        return await self.ingest_documents([document], chunker=chunker)

    # -- retrieval --------------------------------------------------------------------

    async def retriever(self) -> Retriever:
        if self._retriever is None:
            self._retriever = Retriever(self.store, await self._embedder(), hybrid=self.hybrid)
        return self._retriever

    async def _get_rewriter(self, model: Model | None = None) -> Any:
        if self._rewriter is not None:
            return self._rewriter
        if self.rewriter_spec is None:
            from aire.rag.rewrite import IdentityRewriter

            self._rewriter = IdentityRewriter()
            return self._rewriter
        if not isinstance(self.rewriter_spec, str):
            self._rewriter = self.rewriter_spec
            return self._rewriter
        from aire.rag.rewrite import get_rewriter

        self._rewriter = get_rewriter(self.rewriter_spec, model=model)
        return self._rewriter

    async def _get_compressor(self, model: Model | None = None) -> Any:
        if self._compressor is not None:
            return self._compressor
        from aire.rag.compress import TruncateCompressor, get_compressor

        if self.compressor_spec is None:
            self._compressor = TruncateCompressor()
            return self._compressor
        if not isinstance(self.compressor_spec, str):
            self._compressor = self.compressor_spec
            return self._compressor
        self._compressor = get_compressor(self.compressor_spec, model=model)
        return self._compressor

    def _merge_acl(self, filter: dict[str, Any] | None) -> dict[str, Any] | None:
        from aire.rag.acl import merge_filter

        return merge_filter(filter, self.default_acl)

    async def retrieve(
        self,
        query: str,
        *,
        k: int = 5,
        filter: dict[str, Any] | None = None,
        rewrite: bool = True,
    ) -> list[ScoredChunk]:
        """Retrieve and rerank relevant chunks for a query (optional rewrite)."""
        from aire.rag.acl import filter_hits

        retriever = await self.retriever()
        merged = self._merge_acl(filter)
        queries = [query]
        if rewrite and self.rewriter_spec is not None:
            rewriter = await self._get_rewriter()
            queries = await rewriter.rewrite(query)
        fused: dict[str, ScoredChunk] = {}
        for q in queries:
            hits = await retriever.retrieve(q, k=max(k * 2, 6), filter=merged)
            for hit in hits:
                prev = fused.get(hit.chunk.id)
                if prev is None or hit.score > prev.score:
                    fused[hit.chunk.id] = hit
        ranked = sorted(fused.values(), key=lambda h: h.score, reverse=True)
        ranked = filter_hits(ranked, self.default_acl)
        return await self.reranker.rerank(query, ranked, k=k)

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
        compress: bool = True,
        rewrite: bool = True,
        **generate_kwargs: Any,
    ) -> Answer:
        """Retrieve context and generate a grounded answer with citations."""
        question_text = question
        if self.guardrails is not None:
            question_text, _ = await self.guardrails.aapply(question, stage="input")
        hits = await self.retrieve(question_text, k=k, filter=filter, rewrite=rewrite)
        resolved = await self._resolve_model(model)
        if compress:
            compressor = await self._get_compressor(resolved)
            context = await compressor.compress(
                question_text, hits, max_chars=self.max_context_chars
            )
        else:
            context = "\n\n".join(f"[{i + 1}] {hit.chunk.text}" for i, hit in enumerate(hits))
        prompt = (template or self.prompt_template).format(
            context=context, question=question_text
        )
        from aire.models.types import GenerationRequest

        request = GenerationRequest.of(prompt, **generate_kwargs)
        tracer = self.runtime.tracer
        if tracer is not None:
            async with tracer.aspan("rag.ask", attributes={"question": question_text, "k": k}):
                gen = await resolved.generate(request)
        else:
            gen = await resolved.generate(request)

        answer_text = gen.text
        if self.guardrails is not None:
            answer_text, _ = await self.guardrails.aapply(answer_text, stage="output")

        answer = Answer(
            text=answer_text,
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
            {"question": question_text, "retrieved": len(hits), "model": resolved.info.ref},
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
            "rewriter": self.rewriter_spec,
            "compressor": self.compressor_spec,
            "max_context_chars": self.max_context_chars,
            "default_acl": self.default_acl,
            "guardrails": None if self.guardrails is None else self.guardrails.describe(),
        }

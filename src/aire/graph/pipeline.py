"""The KnowledgeGraph pipeline: documents → triples → graph-grounded answers.

GraphRAG, aire-native: ingestion chunks documents, extracts entities and
relations (model-driven or lexical) into an embedded graph store, and indexes
the chunks into a vector store. Querying links question terms to entities,
expands their neighborhood, fuses graph context with vector retrieval, and
answers with citations — the same :class:`Answer` contract as classic RAG.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aire.data.chunking import Chunker, get_chunker
from aire.data.loaders import load as load_dataset
from aire.graph.extract import GraphExtractor, LexicalGraphExtractor, ModelGraphExtractor
from aire.graph.store import GraphStore, SQLiteGraphStore
from aire.graph.types import GraphIndexReport, Subgraph
from aire.models.base import EmbeddingModel, Model
from aire.rag.store import LocalVectorStore, VectorStore
from aire.rag.types import Answer, Chunk, Citation, Document

if TYPE_CHECKING:
    from aire.core.runtime import Runtime

GRAPH_PROMPT = (
    "Answer the question using only the knowledge graph facts and context below. "
    "Cite sources as [1], [2], ... matching the context entries. "
    "If the facts do not contain the answer, say so.\n\n"
    "Graph facts:\n{facts}\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
)


class KnowledgeGraph:
    """A self-contained GraphRAG pipeline (ingest → query → cited answer)."""

    def __init__(
        self,
        runtime: Runtime,
        *,
        store: GraphStore | None = None,
        extractor: GraphExtractor | None = None,
        model: Model | None = None,
        vector_store: VectorStore | None = None,
        embedder: EmbeddingModel | None = None,
        chunker: Chunker | str = "recursive",
        prompt_template: str = GRAPH_PROMPT,
    ) -> None:
        self.runtime = runtime
        self.store = store or SQLiteGraphStore()
        self._model = model
        if extractor is not None:
            self.extractor = extractor
        elif model is not None:
            self.extractor = ModelGraphExtractor(model)
        else:
            self.extractor = LexicalGraphExtractor()
        self.vector_store = vector_store or LocalVectorStore()
        self.embedder = embedder
        self.chunker = get_chunker(chunker) if isinstance(chunker, str) else chunker
        self.prompt_template = prompt_template

    # -- ingestion -----------------------------------------------------------------

    async def _embedder(self) -> EmbeddingModel:
        if self.embedder is None:
            from aire.models.registry import ModelRegistry

            self.embedder = await ModelRegistry(self.runtime).embedder()
        return self.embedder

    async def _answering_model(self) -> Model:
        if self._model is None:
            from aire.models.registry import ModelRegistry

            self._model = await ModelRegistry(self.runtime).use(self.runtime.settings.model.ref)
        return self._model

    async def ingest(
        self,
        source: str | Path | list[str] | list[dict[str, Any]] | list[Document],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> GraphIndexReport:
        """Load, chunk, extract triples and index. Returns a GraphIndexReport."""
        started = time.perf_counter()
        if source and isinstance(source, list) and isinstance(source[0], Document):
            documents = list(source)
        else:
            dataset = load_dataset(source)  # type: ignore[arg-type]
            documents = [Document(text=r.text, metadata=r.metadata) for r in dataset]
        if metadata:
            documents = [
                d.model_copy(update={"metadata": {**d.metadata, **metadata}}) for d in documents
            ]

        chunks: list[Chunk] = []
        for document in documents:
            for piece in self.chunker.chunk(document.text):
                chunks.append(
                    Chunk(
                        document_id=document.id,
                        text=piece.text,
                        index=piece.index,
                        metadata={**document.metadata, "start": piece.start, "end": piece.end},
                    )
                )

        entities = relations = 0
        for chunk in chunks:
            extraction = await self.extractor.extract(chunk.text)
            added_e, added_r = await self.store.upsert(extraction, chunk_id=chunk.id)
            entities += added_e
            relations += added_r

        if chunks:
            embedder = await self._embedder()
            from aire.models.types import EmbeddingRequest

            vectors = await embedder.embed(EmbeddingRequest(inputs=[c.text for c in chunks]))
            chunks = [
                c.model_copy(update={"embedding": v})
                for c, v in zip(chunks, vectors.vectors, strict=True)
            ]
            await self.vector_store.upsert(chunks)

        return GraphIndexReport(
            documents=len(documents),
            chunks=len(chunks),
            entities=entities,
            relations=relations,
            store=self.store.describe().name,
            extractor=self.extractor.describe()["type"],
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )

    # -- querying ------------------------------------------------------------------

    async def subgraph(self, question: str, *, depth: int = 1, limit: int = 8) -> Subgraph:
        """Link question terms to entities and expand their neighborhood."""
        matches = await self.store.match_entities(question, limit=limit)
        if not matches:
            return Subgraph()
        return await self.store.neighborhood([e.name for e in matches], depth=depth)

    async def query(self, question: str, *, k: int = 5, depth: int = 1) -> Answer:
        """Graph-grounded answering: facts + fused vector context → cited answer."""
        from aire.models.types import EmbeddingRequest, GenerationRequest

        subgraph = await self.subgraph(question, depth=depth)
        embedder = await self._embedder()
        vector = (await embedder.embed(EmbeddingRequest(inputs=[question]))).vectors[0]
        hits = await self.vector_store.search(vector, k=k)

        facts = subgraph.as_context() or "(no graph facts matched)"
        context_blocks: list[str] = []
        citations: list[Citation] = []
        for i, hit in enumerate(hits, start=1):
            context_blocks.append(f"[{i}] {hit.chunk.text}")
            citations.append(
                Citation(
                    source=str(hit.chunk.metadata.get("source", hit.chunk.document_id)),
                    chunk_id=hit.chunk.id,
                    excerpt=hit.chunk.text[:240],
                    score=hit.score,
                    metadata=dict(hit.chunk.metadata),
                )
            )
        model = await self._answering_model()
        prompt = self.prompt_template.format(
            facts=facts, context="\n\n".join(context_blocks), question=question
        )
        result = await model.generate(GenerationRequest.of(prompt))
        return Answer(
            text=result.text,
            citations=citations,
            usage=result.usage,
            model=result.model,
            retrieved=len(hits) + len(subgraph.relations),
        )

    def describe(self) -> dict[str, Any]:
        """Machine-readable pipeline manifest — for agents."""
        return {
            "kind": "knowledge_graph",
            "store": self.store.describe().model_dump(mode="json"),
            "extractor": self.extractor.describe(),
            "vector_store": self.vector_store.describe().model_dump(mode="json"),
        }

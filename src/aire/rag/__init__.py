"""Retrieval-augmented generation: stores, retrievers, rerankers, pipeline."""

from aire.rag.incremental import IncrementalIndex
from aire.rag.incremental import wrap as wrap_incremental
from aire.rag.pipeline import Knowledge
from aire.rag.rerank import (
    EmbeddingReranker,
    HFCrossEncoderReranker,
    IdentityReranker,
    LexicalOverlapReranker,
    ModelReranker,
    Reranker,
    get_reranker,
    register_reranker,
)
from aire.rag.retriever import Retriever
from aire.rag.store import LocalVectorStore, VectorStore, cosine_similarity, register
from aire.rag.types import Answer, Chunk, Citation, Document, IndexReport, ScoredChunk

__all__ = [
    "Answer",
    "Chunk",
    "Citation",
    "Document",
    "EmbeddingReranker",
    "HFCrossEncoderReranker",
    "IdentityReranker",
    "IncrementalIndex",
    "IndexReport",
    "Knowledge",
    "LexicalOverlapReranker",
    "LocalVectorStore",
    "ModelReranker",
    "Reranker",
    "Retriever",
    "ScoredChunk",
    "VectorStore",
    "cosine_similarity",
    "get_reranker",
    "register",
    "register_reranker",
    "wrap_incremental",
]

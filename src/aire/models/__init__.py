"""Universal model layer: normalized interfaces, registry and builtin providers."""

from aire.models.base import EmbeddingModel, Model, estimate_tokens, run_sync
from aire.models.builtin import CallableModel, EchoModel, HashingEmbedder
from aire.models.registry import ModelRegistry, register_callable
from aire.models.retry import with_retry
from aire.models.types import (
    CostInfo,
    EmbeddingRequest,
    EmbeddingResult,
    GenerationChunk,
    GenerationRequest,
    GenerationResult,
    ModelInfo,
    StructuredOutputSpec,
    ToolCall,
    ToolDefinition,
)

__all__ = [
    "CallableModel",
    "CostInfo",
    "EchoModel",
    "EmbeddingModel",
    "EmbeddingRequest",
    "EmbeddingResult",
    "GenerationChunk",
    "GenerationRequest",
    "GenerationResult",
    "HashingEmbedder",
    "Model",
    "ModelInfo",
    "ModelRegistry",
    "StructuredOutputSpec",
    "ToolCall",
    "ToolDefinition",
    "estimate_tokens",
    "register_callable",
    "run_sync",
    "with_retry",
]

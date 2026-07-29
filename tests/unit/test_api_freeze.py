"""Public API freeze: top-level exports must remain importable."""

from __future__ import annotations

import aire

# Keep in sync with docs/api_freeze.md — bump minor when removing symbols.
_FROZEN_EXPORTS = (
    "AI",
    "Agent",
    "AgentConfig",
    "AgentResult",
    "AgentStatus",
    "Team",
    "TeamResult",
    "tool",
    "Tool",
    "ToolSpec",
    "ToolResult",
    "SideEffect",
    "Knowledge",
    "Answer",
    "Citation",
    "Assistant",
    "Model",
    "EmbeddingModel",
    "ModelInfo",
    "GenerationRequest",
    "GenerationResult",
    "Dataset",
    "Record",
    "EvalCase",
    "EvalReport",
    "Evaluator",
    "KnowledgeGraph",
    "LongTermMemory",
    "MemoryEntry",
    "Workflow",
    "WorkflowResult",
    "Runtime",
    "Settings",
    "Message",
    "TextContent",
    "Capability",
    "HealthStatus",
    "Manifest",
    "Ref",
    "Usage",
    "AireError",
    "ConfigurationError",
    "PermissionDeniedError",
    "ProviderError",
    "__version__",
)


def test_frozen_exports_present() -> None:
    missing = [name for name in _FROZEN_EXPORTS if not hasattr(aire, name)]
    assert missing == [], f"public API freeze broken; missing: {missing}"


def test_version_is_semverish() -> None:
    parts = aire.__version__.split(".")
    assert len(parts) >= 2
    assert all(p.isdigit() for p in parts[:2])

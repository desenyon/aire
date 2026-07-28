"""aire — the agent-first AI creation library.

One consistent interface from idea to deployed AI system:

    from aire import AI, tool

    assistant = AI.project("docs-bot").documents("./docs").model("mock:echo")
    assistant.index()
    answer = assistant.ask("How do I authenticate?")
    print(answer.text, answer.citations)

Heavy dependencies are never imported until the subsystem needing them is
actually used. Everything is discoverable via ``.describe()`` manifests.
"""

from aire._version import __version__
from aire.agents import Agent, AgentConfig, AgentResult, AgentStatus
from aire.ai import AI
from aire.core.config import Settings
from aire.core.content import (
    AudioContent,
    Content,
    DocumentContent,
    ImageContent,
    Message,
    StructuredContent,
    TextContent,
    VideoContent,
)
from aire.core.errors import AireError, ConfigurationError, PermissionDeniedError, ProviderError
from aire.core.runtime import Runtime
from aire.core.types import Capability, HealthStatus, Manifest, Ref, Usage
from aire.data import Dataset, Record
from aire.evaluation import EvalCase, EvalReport, Evaluator
from aire.knowledge_assistant import Assistant
from aire.models import (
    EmbeddingModel,
    GenerationRequest,
    GenerationResult,
    Model,
    ModelInfo,
)
from aire.rag import Answer, Citation, Knowledge
from aire.tools import SideEffect, Tool, ToolResult, ToolSpec, tool
from aire.workflows import Workflow, WorkflowResult

__all__ = [
    "AI",
    "Agent",
    "AgentConfig",
    "AgentResult",
    "AgentStatus",
    "AireError",
    "Answer",
    "Assistant",
    "AudioContent",
    "Capability",
    "Citation",
    "ConfigurationError",
    "Content",
    "Dataset",
    "DocumentContent",
    "EmbeddingModel",
    "EvalCase",
    "EvalReport",
    "Evaluator",
    "GenerationRequest",
    "GenerationResult",
    "HealthStatus",
    "ImageContent",
    "Knowledge",
    "Manifest",
    "Message",
    "Model",
    "ModelInfo",
    "PermissionDeniedError",
    "ProviderError",
    "Record",
    "Ref",
    "Runtime",
    "Settings",
    "SideEffect",
    "StructuredContent",
    "TextContent",
    "Tool",
    "ToolResult",
    "ToolSpec",
    "Usage",
    "VideoContent",
    "Workflow",
    "WorkflowResult",
    "__version__",
    "tool",
]

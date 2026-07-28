"""aire core runtime: vendor-neutral foundation for all subsystems."""

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
    coerce_messages,
)
from aire.core.context import Budget, ExecutionContext
from aire.core.errors import (
    AireError,
    AuthenticationError,
    BudgetExceededError,
    ConfigurationError,
    ContextLengthError,
    DataError,
    NotFoundError,
    OutputValidationError,
    PermissionDeniedError,
    PluginError,
    ProviderError,
    RateLimitError,
    RetrievalError,
    SafetyError,
    ToolError,
    WorkflowError,
)
from aire.core.events import Event, EventBus
from aire.core.lifecycle import ResourceManager
from aire.core.plugins import PluginInfo, PluginManager
from aire.core.registry import Registries, Registry
from aire.core.runtime import Runtime
from aire.core.types import Capability, HealthStatus, Manifest, Ref, Usage, new_id

__all__ = [
    "AireError",
    "AudioContent",
    "AuthenticationError",
    "Budget",
    "BudgetExceededError",
    "Capability",
    "ConfigurationError",
    "Content",
    "ContextLengthError",
    "DataError",
    "DocumentContent",
    "Event",
    "EventBus",
    "ExecutionContext",
    "HealthStatus",
    "ImageContent",
    "Manifest",
    "Message",
    "NotFoundError",
    "OutputValidationError",
    "PermissionDeniedError",
    "PluginError",
    "PluginInfo",
    "PluginManager",
    "ProviderError",
    "RateLimitError",
    "Ref",
    "Registries",
    "Registry",
    "ResourceManager",
    "RetrievalError",
    "Runtime",
    "SafetyError",
    "Settings",
    "StructuredContent",
    "TextContent",
    "ToolError",
    "Usage",
    "VideoContent",
    "WorkflowError",
    "coerce_messages",
    "new_id",
]

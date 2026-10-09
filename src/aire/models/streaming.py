"""Assemble provider argument fragments before exposing an executable tool call."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from aire.core.errors import ProviderError
from aire.models.types import ToolCall


@dataclass
class ToolCallBuffer:
    id: str = ""
    name: str = ""
    fragments: list[str] = field(default_factory=list)
    initial: dict[str, Any] = field(default_factory=dict)

    def finish(self, provider: str) -> ToolCall:
        try:
            arguments = json.loads("".join(self.fragments)) if self.fragments else self.initial
            if not self.id or not self.name or not isinstance(arguments, dict):
                raise ValueError("tool call requires an id, name and object arguments")
        except (ValueError, TypeError) as exc:
            raise ProviderError(
                provider,
                "invalid streamed tool call",
                code="provider.stream_tool_invalid",
                retryable=False,
                context={"tool_call_id": self.id},
                cause=exc,
            ) from exc
        return ToolCall(id=self.id, name=self.name, arguments=arguments)

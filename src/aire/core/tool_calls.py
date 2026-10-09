"""Provider-neutral tool invocation shared by messages and model results."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ToolCall(BaseModel):
    """A model's request to invoke a tool."""

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_json(cls, id: str, name: str, arguments: str | dict[str, Any]) -> ToolCall:
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError:
                parsed = {"_raw": arguments}
        else:
            parsed = arguments
        return cls(id=id, name=name, arguments=parsed)

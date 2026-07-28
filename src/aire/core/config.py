"""Layered, validated configuration.

Priority (highest first):
1. Explicit Python arguments
2. Project configuration file (``aire.yaml`` / ``aire.json`` / ``pyproject.toml``)
3. Environment variables (``AIRE_`` prefix, ``__`` as nesting separator)
4. User configuration (``~/.config/aire/config.yaml``)
5. Library defaults
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from aire.core.errors import ConfigurationError

ENV_PREFIX = "AIRE_"
PROJECT_FILES = ("aire.yaml", "aire.yml", "aire.json")


class ModelConfig(BaseModel):
    """Default model selection and generation parameters."""

    model_config = ConfigDict(extra="allow")

    ref: str = "mock:echo"
    embedder: str = "local:hashing"
    temperature: float | None = None
    max_tokens: int | None = None


class AgentConfig(BaseModel):
    """Agent runtime defaults."""

    model_config = ConfigDict(extra="allow")

    planning: bool = False
    max_steps: int = 12
    token_budget: int | None = None
    cost_budget_usd: float | None = None
    require_approval: list[str] = Field(default_factory=lambda: ["high_impact"])


class SafetyConfig(BaseModel):
    """Safety and governance defaults."""

    model_config = ConfigDict(extra="allow")

    require_approval: dict[str, bool] = Field(
        default_factory=lambda: {"external_side_effect": True, "high_impact": True}
    )
    pii_detection: bool = True
    injection_detection: bool = True
    secret_redaction: bool = True


class ObservabilityConfig(BaseModel):
    """Tracing and export defaults."""

    model_config = ConfigDict(extra="allow")

    enabled: bool = True
    exporter: str = "memory"  # memory | jsonl | otlp | none
    trace_file: str | None = None
    otlp_endpoint: str | None = None  # e.g. http://localhost:4318
    service_name: str = "aire"
    mask_fields: list[str] = Field(default_factory=lambda: ["api_key", "authorization", "password"])


class GatewayConfig(BaseModel):
    """Model gateway defaults (``aire gateway`` / ``AI.gateway.create()``)."""

    model_config = ConfigDict(extra="allow")

    models: list[str] = Field(default_factory=list)
    aliases: dict[str, str | list[str]] = Field(default_factory=dict)
    embeddings: dict[str, str | list[str]] = Field(default_factory=dict)
    routing: str = "first"  # first | round_robin
    objective: str | None = None  # e.g. lowest_cost, highest_quality
    auth_token: str | None = None
    rate_limit_per_minute: int | None = None
    budgets: dict[str, float] = Field(default_factory=dict)  # alias/ref → USD/day cap
    circuit_breaker: bool = True
    failure_threshold: int = 3
    cooldown_seconds: float = 30.0
    request_log: str | None = None  # JSONL audit path


class ProviderCredential(BaseModel):
    """Credentials for one provider, resolved from config or environment."""

    model_config = ConfigDict(extra="allow")

    api_key: SecretStr | None = None
    base_url: str | None = None
    default_headers: dict[str, str] = Field(default_factory=dict)

    def resolve_key(self, env_var: str) -> str | None:
        if self.api_key is not None:
            return self.api_key.get_secret_value()
        return os.environ.get(env_var)


class Settings(BaseModel):
    """Root settings object for a project or process."""

    model_config = ConfigDict(extra="allow")

    project: str = "aire-project"
    model: ModelConfig = Field(default_factory=ModelConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    providers: dict[str, ProviderCredential] = Field(default_factory=dict)

    # -- loaders ---------------------------------------------------------------

    @classmethod
    def load(
        cls,
        path: str | Path | None = None,
        *,
        overrides: dict[str, Any] | None = None,
        env: bool = True,
    ) -> Settings:
        """Load settings by merging file, environment and explicit overrides."""
        data: dict[str, Any] = {}
        file_path = _find_project_file(path)
        if file_path is not None:
            data = _deep_merge(data, _load_file(file_path))
        user_file = Path.home() / ".config" / "aire" / "config.yaml"
        if path is None and _find_project_file(None) is None and user_file.is_file():
            data = _deep_merge(data, _load_file(user_file))
        if env:
            data = _deep_merge(data, _load_env())
        if overrides:
            data = _deep_merge(data, overrides)
        try:
            return cls.model_validate(data)
        except Exception as exc:  # pydantic ValidationError
            raise ConfigurationError(
                f"invalid configuration: {exc}",
                code="config.validation",
                context={"path": str(file_path) if file_path else None},
                cause=exc,
            ) from exc

    # -- persistence -------------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        payload = self.model_dump(mode="json", exclude_none=True)
        if p.suffix == ".json":
            p.write_text(json.dumps(payload, indent=2))
        else:
            p.write_text(yaml.safe_dump(payload, sort_keys=False))
        return p

    def credential(self, provider: str) -> ProviderCredential:
        return self.providers.get(provider, ProviderCredential())


def _find_project_file(path: str | Path | None) -> Path | None:
    if path is not None:
        p = Path(path)
        if p.is_dir():
            for name in PROJECT_FILES:
                candidate = p / name
                if candidate.is_file():
                    return candidate
            return None
        return p if p.is_file() else None
    cwd = Path.cwd()
    for directory in (cwd, *cwd.parents):
        for name in PROJECT_FILES:
            candidate = directory / name
            if candidate.is_file():
                return candidate
    return None


def _load_file(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text()
        data = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ConfigurationError(
            f"failed to load config file {path}: {exc}",
            code="config.load_failed",
            context={"path": str(path)},
            cause=exc,
        ) from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigurationError(
            f"config file {path} must contain a mapping at the top level",
            code="config.invalid_shape",
            context={"path": str(path)},
        )
    return data


def _load_env() -> dict[str, Any]:
    data: dict[str, Any] = {}
    for key, value in os.environ.items():
        if not key.startswith(ENV_PREFIX) or key == "AIRE_CONFIG":
            continue
        parts = [p.lower() for p in key[len(ENV_PREFIX) :].split("__") if p]
        if not parts:
            continue
        node = data
        for part in parts[:-1]:
            child = node.get(part)
            if not isinstance(child, dict):
                child = {}
                node[part] = child
            node = child
        node[parts[-1]] = _coerce_scalar(value)
    return data


def _coerce_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        existing = result.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            result[key] = _deep_merge(existing, value)
        else:
            result[key] = value
    return result

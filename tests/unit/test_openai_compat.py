"""Universal OpenAI-compatible provider: aliases, env resolution, lazy hints."""

from __future__ import annotations

import pytest

from aire.core.config import Settings
from aire.core.errors import AuthenticationError, ConfigurationError
from aire.core.runtime import Runtime
from aire.integrations.openai_compat import (
    KNOWN_ENDPOINTS,
    describe_endpoints,
    register,
)
from aire.models.registry import ModelRegistry
from tests.conftest import arun


def _runtime(settings: Settings | None = None) -> Runtime:
    return Runtime(settings or Settings(project="test-project"))


def test_known_endpoints_cover_local_and_hosted() -> None:
    local = {name for name, spec in KNOWN_ENDPOINTS.items() if spec.local}
    hosted = {name for name, spec in KNOWN_ENDPOINTS.items() if spec.requires_api_key}
    assert {"lmstudio", "llamacpp", "vllm", "mlx", "localai", "llamafile", "tgi"} <= local
    assert {"groq", "together", "fireworks", "deepseek", "mistral", "xai", "openrouter"} <= hosted


def test_describe_endpoints_machine_readable() -> None:
    catalog = describe_endpoints()
    assert catalog["lmstudio"]["base_url"] == "http://localhost:1234/v1"
    assert catalog["lmstudio"]["local"] is True
    assert catalog["groq"]["env_key"] == "GROQ_API_KEY"


def test_local_alias_uses_default_base_url() -> None:
    rt = _runtime()
    register(rt)
    model = arun(ModelRegistry(rt).use("lmstudio:qwen2.5-7b"))
    assert model.info.ref == "lmstudio:qwen2.5-7b"
    assert str(model._client.raw.base_url).rstrip("/") == "http://localhost:1234/v1"
    assert "Authorization" not in model._client.raw.headers


def test_hosted_alias_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    rt = _runtime()
    register(rt)
    with pytest.raises(AuthenticationError):
        arun(ModelRegistry(rt).use("groq:llama-3.3-70b-versatile"))


def test_hosted_alias_picks_up_env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_key")
    rt = _runtime()
    register(rt)
    model = arun(ModelRegistry(rt).use("groq:llama-3.3-70b-versatile"))
    assert model._client.raw.headers["authorization"] == "Bearer gsk_test_key"
    assert str(model._client.raw.base_url).rstrip("/") == "https://api.groq.com/openai/v1"


def test_config_file_credentials_win_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk_env")
    settings = Settings(
        project="test-project",
        providers={"groq": {"api_key": "gsk_config", "base_url": "http://proxy:9000/v1"}},
    )
    rt = _runtime(settings)
    register(rt)
    model = arun(ModelRegistry(rt).use("groq:anything"))
    assert str(model._client.raw.base_url).rstrip("/") == "http://proxy:9000/v1"
    assert model._client.raw.headers["authorization"] == "Bearer gsk_config"


def test_generic_provider_requires_base_url() -> None:
    rt = _runtime()
    register(rt)
    with pytest.raises(ConfigurationError):
        arun(ModelRegistry(rt).use("openai_compatible:my-model"))


def test_generic_provider_with_explicit_base_url() -> None:
    rt = _runtime()
    register(rt)
    model = arun(
        ModelRegistry(rt).use(
            "openai_compatible:my-model", base_url="http://gpu-box:8000/v1", api_key="k"
        )
    )
    assert model.info.ref == "openai_compatible:my-model"
    assert str(model._client.raw.base_url).rstrip("/") == "http://gpu-box:8000/v1"


def test_embedder_factory_registered() -> None:
    rt = _runtime()
    register(rt)
    embedder = arun(ModelRegistry(rt).embedder("lmstudio:text-embedding-nomic"))
    assert embedder.name == "lmstudio:text-embedding-nomic"


def test_lazy_hint_registers_whole_catalog() -> None:
    rt = _runtime()
    assert not rt.model_providers.has("groq")
    arun(ModelRegistry(rt).use("llamacpp:llama-3.2"))  # triggers lazy registration
    for alias in KNOWN_ENDPOINTS:
        assert rt.model_providers.has(alias)
    assert rt.model_providers.has("openai_compatible")

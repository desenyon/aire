"""aire knowledge exposed over MCP: resources (docs/manifests) and prompts.

Resources let any MCP-speaking agent read the aire usage guide, live library
manifest, error taxonomy and ref catalog; prompts give task-shaped templates
(build a RAG assistant, an agent, a gateway, ...) with argument substitution.
This is how an agent learns to operate aire through MCP itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from aire.core.errors import NotFoundError

_GUIDE_PATH = Path(__file__).parent / "guide.md"


class MCPResource(BaseModel):
    """A readable document addressable by URI."""

    uri: str
    name: str
    description: str = ""
    mime_type: str = "text/markdown"


class MCPPromptArgument(BaseModel):
    name: str
    description: str = ""
    required: bool = False


class MCPPrompt(BaseModel):
    """A task template returning conversation messages."""

    name: str
    description: str = ""
    arguments: list[MCPPromptArgument] = Field(default_factory=list)
    template: str = ""


# -- resources ------------------------------------------------------------------


def _guide_text() -> str:
    return _GUIDE_PATH.read_text()


def _manifest_text() -> str:
    import json

    from aire.ai import AI

    return json.dumps(AI.describe(), indent=2, default=str)


def _errors_text() -> str:
    from aire.core import errors

    lines = ["# aire error taxonomy", ""]
    for name in sorted(dir(errors)):
        cls = getattr(errors, name)
        if (
            isinstance(cls, type)
            and issubclass(cls, errors.AireError)
            and cls is not errors.AireError
        ):
            retryable = getattr(cls, "retryable", False)
            doc = (cls.__doc__ or "").strip().splitlines()
            lines.append(f"- `{name}` (retryable={retryable}) — {doc[0] if doc else ''}")
    lines.append("")
    lines.append("All carry `.code`, `.message`, `.context`; catch `AireError` to handle all.")
    return "\n".join(lines)


def _refs_text() -> str:
    return "\n".join(
        [
            "# aire ref catalog (provider:name)",
            "",
            "## models",
            "openai, anthropic, ollama, huggingface, mock, echo, lmstudio, llamacpp,",
            "vllm, mlx, groq, together, mistral, openrouter, deepseek, xai, fireworks,",
            "cerebras, openai_compatible (name=base_url=http://...)",
            "",
            "## embedders",
            "builtin:hash (offline), plus every model provider above",
            "",
            "## vector stores",
            "local:default, sqlite:<path>, qdrant:<collection>, chroma:<collection>,",
            "pinecone:<index>, weaviate:<class>, milvus:<collection>",
            "",
            "## graph stores",
            "sqlite:<path> (use sqlite:memory for in-memory)",
            "",
            "## estimators (AI.ml)",
            "simple:majority, simple:centroid, simple:knn, simple:linear_regression,",
            "sklearn:<name|dotted.path>, torch:mlp",
            "",
            "## memory",
            "buffer, jsonl:<path>, AI.memory.create(path=...)",
        ]
    )


def builtin_resources() -> list[MCPResource]:
    """Resources every aire MCP server exposes."""
    return [
        MCPResource(
            uri="aire://guide",
            name="aire usage guide",
            description=(
                "How to use every aire subsystem: refs, models, RAG, graphs, "
                "memory, agents, ML, gateway, errors."
            ),
        ),
        MCPResource(
            uri="aire://manifest",
            name="library manifest",
            description="Live AI.describe(): version, namespaces, registered components.",
            mime_type="application/json",
        ),
        MCPResource(
            uri="aire://errors",
            name="error taxonomy",
            description="Every AireError subclass with retryability — how to handle failures.",
        ),
        MCPResource(
            uri="aire://refs",
            name="ref catalog",
            description="Every provider:name scheme for models, stores, estimators, memory.",
        ),
    ]


def read_resource(uri: str) -> str:
    """Resolve a resource URI to its text content."""
    readers = {
        "aire://guide": _guide_text,
        "aire://manifest": _manifest_text,
        "aire://errors": _errors_text,
        "aire://refs": _refs_text,
    }
    try:
        return readers[uri]()
    except KeyError:
        raise NotFoundError("resource", uri, context={"available": sorted(readers)}) from None


# -- prompts ----------------------------------------------------------------------


def builtin_prompts() -> list[MCPPrompt]:
    """Task-shaped prompt templates for common aire builds."""
    return [
        MCPPrompt(
            name="aire_quickstart",
            description="Orient an agent in aire: what exists and which refs to reach for.",
            template=(
                "You are building with the aire AI library. Read the aire://guide resource "
                "first. Use provider:name refs everywhere, offline defaults for tests "
                "(mock: models, builtin:hash embedders, simple:* estimators), and call "
                ".describe() to discover components. Handle failures via AireError.code "
                "and only retry when retryable is true."
            ),
        ),
        MCPPrompt(
            name="aire_rag",
            description="Build a cited RAG assistant over a document set.",
            arguments=[
                MCPPromptArgument(
                    name="docs", description="Path/URL of the documents", required=True
                ),
                MCPPromptArgument(name="store", description="Vector store ref"),
            ],
            template=(
                "Build a RAG assistant with aire over the documents at '{docs}'. "
                "Use AI.rag.create(store='{store}') with the builtin:hash embedder for "
                "offline runs, ingest the documents, and answer questions via "
                "rag.query(question) returning answers with numbered citations. "
                "Show the citations to the user."
            ),
        ),
        MCPPrompt(
            name="aire_agent",
            description="Create a tool-using agent with budgets and memory.",
            arguments=[
                MCPPromptArgument(name="model", description="Model ref", required=True),
                MCPPromptArgument(name="task", description="What the agent should do"),
            ],
            template=(
                "Create an aire agent on model '{model}' to {task}. Define tools with the "
                "@tool decorator (typed signature, description, side_effect), pass them to "
                "AI.agents.create(tools=[...], max_steps=8), attach long-term memory with "
                "AI.memory.create() if the task spans sessions, and run with "
                "await agent.run(task). Report result.output and result.steps."
            ),
        ),
        MCPPrompt(
            name="aire_gateway",
            description="Stand up an OpenAI/Anthropic-compatible model gateway.",
            arguments=[
                MCPPromptArgument(
                    name="routes", description="alias → candidate model refs", required=True
                )
            ],
            template=(
                "Stand up an aire gateway mapping these aliases to candidate models: "
                "{routes}. Use AI.gateway.create(routes={{...}}) or `aire gateway` with an "
                "aire.yaml containing routes, per-alias daily budgets and circuit breaker "
                "settings. Verify GET /v1/models and POST /v1/chat/completions."
            ),
        ),
        MCPPrompt(
            name="aire_ml",
            description="Train, cross-validate, and select an ML model on an aire Dataset.",
            arguments=[
                MCPPromptArgument(name="estimator", description="Estimator ref", required=True),
                MCPPromptArgument(name="target", description="Target metadata field"),
            ],
            template=(
                "Train '{estimator}' with aire on the loaded dataset (target field "
                "'{target}'): est = await AI.ml.fit('{estimator}', dataset, "
                "target='{target}'); report evaluate metrics (classification report "
                "or MAE/RMSE/R2), optionally await AI.ml.cross_validate / "
                "AI.ml.grid_search for selection, then est.save(path). Check "
                "AI.ml.catalog() and AI.ml.backends() first; use simple:* when "
                "sklearn/torch are not installed. For neural stacks "
                "(aire[torch]): AI.ml.arch.compose(layers=[...]) or "
                "AI.ml.arch.available() / register_attention / register_ffn; "
                "train with AI.ml.optim + AI.ml.loss."
            ),
        ),
    ]


def get_prompt(name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    """Render a prompt template into MCP getPrompt result form."""
    prompts = {p.name: p for p in builtin_prompts()}
    prompt = prompts.get(name)
    if prompt is None:
        raise NotFoundError("prompt", name, context={"available": sorted(prompts)})
    values = {str(k): str(v) for k, v in (arguments or {}).items()}
    defaults = {"store": "local:default", "target": "label"}
    for key, default in defaults.items():
        values.setdefault(key, default)
    text = prompt.template.format_map(_SafeFormat(values))
    return {
        "description": prompt.description,
        "messages": [{"role": "user", "content": {"type": "text", "text": text}}],
    }


class _SafeFormat(dict[str, Any]):
    """format_map helper: leave unknown placeholders intact instead of KeyError."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"

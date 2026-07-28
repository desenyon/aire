"""``aire`` command line interface.

aire init <name>            scaffold a project
aire run <prompt>           one-shot generation with the configured model
aire evaluate <dataset>     run an evaluation suite
aire serve                  serve the project's build_target() via uvicorn
aire gateway                serve an OpenAI-compatible model gateway
aire inspect <what>         inspect models|tools|plugins|config
aire plugins                list discovered plugins
aire doctor                 diagnose environment, credentials and config
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import typer

from aire._version import __version__
from aire.core.errors import AireError
from aire.models.base import Model

app = typer.Typer(
    name="aire",
    help="aire — the agent-first AI creation library.",
    add_completion=False,
    no_args_is_help=True,
)


def _echo_json(data: Any) -> None:
    typer.echo(json.dumps(data, indent=2, default=str))


def _fail(exc: AireError) -> None:
    typer.secho(str(exc), fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1) from exc


@app.command()
def init(
    name: str,
    directory: Path = typer.Option(Path("."), "--dir", "-d", help="Parent directory."),
    model: str = typer.Option("mock:echo", help="Default model reference."),
) -> None:
    """Create a new aire project from the standard template."""
    root = directory / name
    if root.exists() and any(root.iterdir()):
        typer.secho(f"directory {root} is not empty", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    (root / "docs").mkdir(parents=True, exist_ok=True)
    (root / "evals").mkdir(parents=True, exist_ok=True)
    (root / "aire.yaml").write_text(
        f"project: {name}\nmodel:\n  ref: {model}\n  embedder: local:hashing\n"
    )
    (root / "app.py").write_text(
        '"""aire project entrypoint."""\n\n'
        "from aire import AI\n\n\n"
        "def build_target():\n"
        f'    return AI.project("{name}", config="aire.yaml").documents("./docs")\n\n\n'
        'if __name__ == "__main__":\n'
        "    assistant = build_target()\n"
        "    assistant.index()\n"
        '    print(assistant.ask("What is this project about?").text)\n'
    )
    (root / "evals" / "questions.jsonl").write_text(
        '{"input": "What is this project about?", "expected": ""}\n'
    )
    (root / "docs" / "welcome.md").write_text(
        f"# {name}\n\nReplace the files in this directory with your documentation.\n"
    )
    typer.secho(f"created project in {root}", fg=typer.colors.GREEN)
    typer.echo(f"next: cd {root} && aire run 'hello' && aire doctor")


@app.command()
def run(
    prompt: str,
    model: str | None = typer.Option(None, help="Model ref, e.g. openai:gpt-4o-mini."),
    config: Path | None = typer.Option(None, help="Path to aire.yaml."),
) -> None:
    """One-shot generation with the configured (or given) model."""
    from aire.ai import AI

    try:
        runtime = (
            AI.configure()
            if config is None
            else AI.configure(
                __import__("aire.core.config", fromlist=["Settings"]).Settings.load(config)
            )
        )
        resolved = AI.models.use_sync(model or runtime.settings.model.ref)
        text = run_generate(resolved, prompt)
        typer.echo(text)
    except AireError as exc:
        _fail(exc)


def run_generate(model: Model, prompt: str) -> str:
    from aire.models.base import run_sync

    return run_sync(model.ask(prompt))


@app.command()
def evaluate(
    dataset: Path,
    model: str | None = typer.Option(None, help="Model ref to evaluate."),
    metrics: str = typer.Option("accuracy", help="Comma-separated metric names."),
    output: Path | None = typer.Option(None, help="Write the report JSON here."),
) -> None:
    """Run an evaluation suite against a model."""
    from aire.ai import AI

    try:
        runtime = AI.configure()
        target = AI.models.use_sync(model or runtime.settings.model.ref)
        report = AI.evaluate(target, dataset, metrics=[m.strip() for m in metrics.split(",")])
        _echo_json(report.summary())
        if output:
            from aire.evaluation.runner import save_report

            save_report(report, output)
            typer.secho(f"report written to {output}", fg=typer.colors.GREEN)
    except AireError as exc:
        _fail(exc)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8000),
    app_file: Path = typer.Option(Path("app.py"), help="Module exposing build_target()."),
) -> None:
    """Serve the project's build_target() through a production FastAPI app."""
    try:
        import uvicorn
    except ImportError:
        typer.secho("uvicorn required: pip install 'aire[serve]'", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None
    import importlib.util

    spec = importlib.util.spec_from_file_location("aire_user_app", app_file)
    if spec is None or spec.loader is None:
        typer.secho(f"cannot load {app_file}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    build = getattr(module, "build_target", None)
    if not callable(build):
        typer.secho(f"{app_file} must define build_target()", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    from aire.ai import AI

    api = AI.deploy.api(build())
    typer.secho(f"serving on http://{host}:{port} (docs at /docs)", fg=typer.colors.GREEN)
    uvicorn.run(api, host=host, port=port)


def parse_alias_options(values: list[str] | None) -> dict[str, str | list[str]]:
    """Parse ``--alias public=ref1,ref2`` options into an alias mapping."""
    aliases: dict[str, str | list[str]] = {}
    for raw in values or []:
        public, sep, refs = raw.partition("=")
        if not sep or not public.strip() or not refs.strip():
            typer.secho(
                f"invalid --alias {raw!r}: expected public=provider:name[,provider:name...]",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=2)
        candidates = [r.strip() for r in refs.split(",") if r.strip()]
        aliases[public.strip()] = candidates[0] if len(candidates) == 1 else candidates
    return aliases


@app.command()
def gateway(
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(4000),
    model: list[str] | None = typer.Option(
        None, "--model", "-m", help="Model refs to expose (repeatable)."
    ),
    alias: list[str] | None = typer.Option(
        None, "--alias", "-a", help="public=ref[,ref...] mapping (repeatable)."
    ),
    embed_alias: list[str] | None = typer.Option(
        None, "--embed-alias", help="public=embedder-ref mapping (repeatable)."
    ),
    routing: str | None = typer.Option(None, help="first | round_robin"),
    objective: str | None = typer.Option(
        None, help="Route by objective: lowest_cost|highest_quality|..."
    ),
    auth_token: str | None = typer.Option(None, help="Require this bearer token."),
    rate_limit: int | None = typer.Option(None, help="Requests per minute per client."),
) -> None:
    """Serve an OpenAI-compatible gateway in front of any providers.

    Examples:

        aire gateway -m ollama:llama3.2

        aire gateway -a smart=anthropic:claude-sonnet-4-5,openai:gpt-4o-mini

        aire gateway --objective lowest_cost --auth-token sk-internal
    """
    try:
        import uvicorn
    except ImportError:
        typer.secho("uvicorn required: pip install 'aire[serve]'", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None
    from aire.ai import AI

    try:
        app = AI.gateway.create(
            models=model or None,
            aliases=parse_alias_options(alias) or None,
            embeddings=parse_alias_options(embed_alias) or None,
            routing=routing,
            objective=objective,
            auth_token=auth_token,
            rate_limit_per_minute=rate_limit,
        )
    except AireError as exc:
        _fail(exc)
    typer.secho(f"gateway listening on http://{host}:{port} (docs at /docs)", fg=typer.colors.GREEN)
    uvicorn.run(app, host=host, port=port)


@app.command(name="mcp-serve")
def mcp_serve() -> None:
    """Serve builtin + registered tools over MCP (stdio JSON-RPC).

    Point any MCP host (Claude Code, IDEs) at this command::

        aire mcp-serve
    """
    from aire.mcp.server import main as serve_main

    typer.secho("aire MCP server on stdio", fg=typer.colors.GREEN, err=True)
    serve_main()


@app.command()
def inspect(
    what: str = typer.Argument("all", help="models|tools|plugins|config|all"),
) -> None:
    """Inspect models, tools, plugins and configuration."""
    from aire.ai import AI

    runtime = AI.runtime()
    report: dict[str, Any] = {}
    if what in {"models", "all"}:
        report["models"] = AI.models.describe()
    if what in {"tools", "all"}:
        report["tools"] = runtime.tools.names()
    if what in {"plugins", "all"}:
        report["plugins"] = list(runtime.plugins.loaded)
    if what in {"config", "all"}:
        report["config"] = runtime.settings.model_dump(mode="json", exclude={"providers"})
    _echo_json(report)


@app.command()
def plugins() -> None:
    """List discovered and loaded plugins."""
    from aire.ai import AI

    runtime = AI.runtime()
    discovered = runtime.plugins.load_entry_points(runtime)
    _echo_json(
        {
            "loaded": runtime.plugins.describe(),
            "newly_discovered": [p.model_dump(mode="json") for p in discovered],
        }
    )


def _doctor_live_checks(check: Any) -> None:
    """Connectivity probes used by ``aire doctor --live``."""
    import os

    from aire.ai import AI
    from aire.models.base import run_sync

    for spec in ("mock:echo",):
        try:
            model = AI.models.use_sync(spec)
            text = run_generate(model, "ping")
            check(f"live:{spec}", bool(text), text[:40])
        except Exception as exc:
            check(f"live:{spec}", False, f"{type(exc).__name__}: {exc}")
    try:
        emb = AI.models.embedder_sync("local:hashing")
        vec = run_sync(emb.embed_one("ping"))
        check("live:local:hashing", bool(vec), f"dim={len(vec)}")
    except Exception as exc:
        check("live:local:hashing", False, f"{type(exc).__name__}: {exc}")
    for env_var, spec in (
        ("OPENAI_API_KEY", "openai:gpt-4o-mini"),
        ("ANTHROPIC_API_KEY", "anthropic:claude-sonnet-4-5"),
    ):
        if not os.environ.get(env_var):
            check(f"live:{spec}", True, "skipped (no credentials)")
            continue
        try:
            model = AI.models.use_sync(spec)
            health = run_sync(model.health())
            ok = getattr(health, "ok", None)
            if ok is None:
                ok = getattr(health, "status", "ok") == "ok"
            check(f"live:{spec}", bool(ok), str(health))
        except Exception as exc:
            check(f"live:{spec}", False, f"{type(exc).__name__}: {exc}")


@app.command()
def doctor(
    live: bool = typer.Option(
        False,
        "--live",
        help="Also probe provider connectivity (may use network/credentials).",
    ),
) -> None:
    """Check Python, dependencies, credentials, providers and configuration."""
    import importlib.util
    import os

    from aire.ai import AI
    from aire.models.builtin import EchoModel, HashingEmbedder

    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"check": name, "ok": ok, "detail": detail})

    check("python>=3.11", sys.version_info >= (3, 11), sys.version.split()[0])
    for package in ("pydantic", "httpx", "yaml", "typer"):
        check(f"dependency:{package}", importlib.util.find_spec(package) is not None)
    check(
        "dependency:fastapi(optional)",
        importlib.util.find_spec("fastapi") is not None,
        "needed for aire deploy/serve",
    )

    runtime = AI.runtime()
    for env_var, provider in (
        ("OPENAI_API_KEY", "openai"),
        ("ANTHROPIC_API_KEY", "anthropic"),
        ("HF_TOKEN", "huggingface"),
    ):
        present = bool(os.environ.get(env_var)) or (
            runtime.settings.credential(provider).api_key is not None
        )
        check(f"credentials:{provider}", present, env_var if not present else "configured")

    check("provider:mock", True, EchoModel().info.ref)
    check("embedder:local", True, f"dimension={HashingEmbedder().dimension}")
    check("config:loaded", True, f"project={runtime.settings.project}")

    if live:
        _doctor_live_checks(check)

    failures = [
        c
        for c in checks
        if not c["ok"] and "optional" not in c["check"] and not c["check"].startswith("credentials")
    ]
    _echo_json({"checks": checks, "healthy": not failures, "live": live})
    if failures:
        raise typer.Exit(code=1)


@app.command()
def scaffold(
    kind: str = typer.Argument("rag", help="rag|agent|finetune|gateway|workflow"),
    name: str = typer.Option("demo", "--name", "-n"),
    directory: Path = typer.Option(Path("."), "--dir", "-d"),
) -> None:
    """Scaffold a recipe-based project snippet into a directory."""
    root = directory / name
    root.mkdir(parents=True, exist_ok=True)
    kind_key = kind.strip().lower()
    templates: dict[str, str] = {
        "rag": (
            "from aire import AI\n\n"
            "stack = AI.recipe('rag')\n"
            "knowledge = stack['knowledge']\n"
            "# knowledge.index('./docs')\n"
            "# print(knowledge.ask('What is this?').text)\n"
        ),
        "agent": (
            "from aire import AI\n\n"
            "stack = AI.recipe('agent')\n"
            "agent = stack['agent']\n"
            "print(agent.run_sync('Say hello').output)\n"
        ),
        "finetune": (
            "from aire import AI\n\n"
            "stack = AI.recipe('finetune', backend='lm')\n"
            "print(stack['trainer'].describe())\n"
        ),
        "gateway": (
            "from aire import AI\n\n"
            "app = AI.recipe('gateway')['app']\n"
            "# uvicorn.run(app, host='127.0.0.1', port=4000)\n"
        ),
        "workflow": (
            "from aire import AI\n\n"
            "wf = AI.workflow('demo')\n"
            "AI.workflows.hitl_node(wf, 'approve', lambda x, ctx: x)\n"
            "wf.approver = AI.workflows.interactive_approver(auto={'approve': True})\n"
            "print(wf)\n"
        ),
    }
    if kind_key not in templates:
        typer.secho(
            f"unknown scaffold kind {kind!r}; choose from {sorted(templates)}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)
    (root / "main.py").write_text(templates[kind_key])
    (root / "aire.yaml").write_text(f"project: {name}\nmodel:\n  ref: mock:echo\n")
    from aire.project.lock import create_lock, write_lock

    write_lock(create_lock(name, model="mock:echo", embedder="local:hashing"), root / "aire.lock")
    typer.secho(f"scaffolded {kind_key} project in {root}", fg=typer.colors.GREEN)
    typer.echo(f"next: cd {root} && python main.py")


@app.command()
def version() -> None:
    """Print the aire version."""
    typer.echo(__version__)


def main() -> None:
    app()


if __name__ == "__main__":
    main()

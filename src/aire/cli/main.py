"""``aire`` command line interface.

aire init <name>            scaffold a project
aire run <prompt>           one-shot generation with the configured model
aire evaluate <dataset>     run an evaluation suite
aire serve                  serve the project's build_target() via uvicorn
aire gateway                serve an OpenAI-compatible model gateway
aire inspect <what>         inspect models|tools|plugins|config
aire plugins                list discovered plugins
aire doctor [--live]        diagnose env; --live probes providers + Ollama
aire foundation             toy architecture only (NOT pretrained weights)
aire analytics              demo counters unless process metrics attached
"""

from __future__ import annotations

import json
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
        typer.secho("uvicorn required: pip install 'aire-ai[serve]'", fg=typer.colors.RED, err=True)
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
        typer.secho("uvicorn required: pip install 'aire-ai[serve]'", fg=typer.colors.RED, err=True)
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
    """Inspect models, tools (incl. builtins), plugins and configuration."""
    from aire.ai import AI
    from aire.tools.builtins import builtin_tools

    if what not in {"models", "tools", "plugins", "config", "all"}:
        raise typer.BadParameter(f"unknown inspection selector: {what}")
    runtime = AI.runtime()
    report: dict[str, Any] = {}
    if what in {"models", "all"}:
        report["models"] = AI.models.describe()
    if what in {"tools", "all"}:
        report["tools"] = {
            "registered": runtime.tools.names(),
            "builtin_tools": [t.name for t in builtin_tools()],
        }
    if what in {"plugins", "all"}:
        report["plugins"] = list(runtime.plugins.loaded)
    if what in {"config", "all"}:
        report["config"] = runtime.settings.safe_dict()
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


@app.command()
def doctor(
    live: bool = typer.Option(
        False, "--live", help="Probe configured providers and local Ollama (may incur usage)."
    ),
    smoke: bool = typer.Option(
        False, "--smoke", help="Exercise local generation, embedding, tools and cited RAG offline."
    ),
    output: Path | None = typer.Option(None, "--output", help="Save the JSON diagnostic report."),
    config: Path | None = typer.Option(None, "--config", help="Explicit project configuration."),
) -> None:
    """Report environment/config checks with pass, fail and skipped evidence."""
    from aire.cli.diagnostics import run_diagnostics
    from aire.core.serialization import write_json_file

    report = run_diagnostics(live=live, smoke=smoke, config=config)
    if output is not None:
        try:
            write_json_file(output, report)
        except OSError:
            typer.secho("could not write diagnostic report", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from None
    _echo_json(report)
    if not report["healthy"]:
        raise typer.Exit(code=1)


@app.command()
def scaffold(
    kind: str = typer.Argument("rag", help="rag|agent|finetune|gateway|workflow"),
    name: str = typer.Option("demo", "--name", "-n"),
    directory: Path = typer.Option(Path("."), "--dir", "-d"),
) -> None:
    """Scaffold a recipe-based project snippet into a directory."""
    root = directory / name
    if root.exists() and any(root.iterdir()):
        typer.secho(f"directory {root} is not empty", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    root.mkdir(parents=True, exist_ok=True)
    kind_key = kind.strip().lower()
    templates: dict[str, str] = {
        "rag": (
            "from aire import AI\n\n"
            "stack = AI.recipe('rag')\n"
            "knowledge = stack['knowledge']\n"
            "from aire.models.base import run_sync\n"
            "from aire.rag.types import Document\n"
            "run_sync(knowledge.ingest([Document(text='Aire uses short lived tokens.')]))\n"
            "answer = run_sync(knowledge.ask('What does Aire use?'))\n"
            "assert answer.citations\n"
            "print(answer.text)\n"
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
            "from aire.models.base import run_sync\n"
            "from aire.data.dataset import Dataset\n"
            "result = run_sync(stack['trainer'].fit(Dataset.from_texts(['small local example'])))\n"
            "assert result.epochs_completed > 0\n"
            "print('offline toy LM result:', result.model_dump())\n"
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
            "from aire.models.base import run_sync\n"
            "result = run_sync(wf.run('approved'))\n"
            "assert result.ok\n"
            "print(result.output)\n"
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


deploy_app = typer.Typer(help="Deployment helpers (artifacts, scale packs).")
app.add_typer(deploy_app, name="deploy")


@deploy_app.command("scale")
def deploy_scale(
    directory: Path = typer.Argument(Path("."), help="Output directory."),
    project: str = typer.Option("aire-app", help="Project name."),
    replicas: int = typer.Option(2, help="Desired replicas."),
    max_replicas: int = typer.Option(10, help="HPA max replicas."),
    image: str = typer.Option("aire-app:latest", help="Container image."),
) -> None:
    """Generate Docker Compose + Kubernetes scale pack."""
    from aire.deployment.scale import ScaleConfig, generate_scale_pack

    try:
        artifacts = generate_scale_pack(
            directory,
            config=ScaleConfig(
                project=project,
                name=project,
                replicas=replicas,
                max_replicas=max_replicas,
                image=image,
            ),
        )
        _echo_json(artifacts.model_dump(mode="json"))
        typer.secho(f"wrote {len(artifacts.files)} files under {directory}", fg=typer.colors.GREEN)
    except AireError as exc:
        _fail(exc)


@app.command("foundation")
def foundation_cmd(
    family: str = typer.Option("gpt2", help="gpt2|llama|mistral|moe|custom (toy family)"),
    n_layer: int = typer.Option(4, help="Transformer layers."),
    describe_only: bool = typer.Option(False, "--describe", help="Print catalog only."),
) -> None:
    """Create or describe a toy foundational architecture (NOT pretrained weights)."""
    from aire.training.foundation import catalog, create_foundation

    if describe_only:
        _echo_json(catalog())
        return
    try:
        model = create_foundation(family, n_layer=n_layer)
        payload = model.describe()
        payload["cli_note"] = "toy architecture only — not pretrained weights"
        _echo_json(payload)
    except AireError as exc:
        _fail(exc)
    except Exception as exc:  # torch may be missing
        typer.secho(f"foundation unavailable: {exc}", fg=typer.colors.YELLOW, err=True)
        _echo_json(catalog())


def _load_analytics_source(env_ref: str) -> Any:
    import importlib

    try:
        module_name, attribute = env_ref.split(":", 1)
        analytics_obj = getattr(importlib.import_module(module_name), attribute)
        if callable(analytics_obj):
            analytics_obj = analytics_obj()
        if not callable(getattr(analytics_obj, "report", None)) or not callable(
            getattr(analytics_obj, "prometheus", None)
        ):
            raise TypeError("analytics source must expose report() and prometheus()")
    except (ImportError, AttributeError, ValueError, TypeError) as exc:
        typer.secho(
            "Cannot load AIRE_ANALYTICS; expected module:attribute with an Analytics source",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2) from exc
    return analytics_obj


@app.command()
def analytics(
    prometheus: bool = typer.Option(False, help="Emit Prometheus text instead of JSON."),
) -> None:
    """Print an analytics report (demo mode unless AIRE_ANALYTICS is set)."""
    import os

    from aire.observability.analytics import create_analytics

    # Prefer a module-level singleton if present; else env-named import; else demo.
    analytics_obj = None
    try:
        from aire import observability as obs

        analytics_obj = getattr(obs, "ANALYTICS", None) or getattr(obs, "default_analytics", None)
        if callable(analytics_obj):
            analytics_obj = analytics_obj()
    except Exception:
        analytics_obj = None
    env_ref = os.environ.get("AIRE_ANALYTICS")
    if env_ref:
        analytics_obj = _load_analytics_source(env_ref)
    demo = analytics_obj is None
    if analytics_obj is None:
        analytics_obj = create_analytics()
        analytics_obj.record_run(model="mock:echo", cost_usd=0.0, latency_ms=1.0, operation="cli")
    if prometheus:
        if demo:
            typer.echo("# aire analytics: demo mode (no process metrics attached)")
        typer.echo(analytics_obj.prometheus())
    else:
        report = analytics_obj.report().describe()
        report["demo_mode"] = demo
        if demo:
            report["note"] = (
                "demo counters only — attach Metrics via create_analytics(metrics=...) "
                "or set observability.ANALYTICS / AIRE_ANALYTICS for live data"
            )
        _echo_json(report)


def main() -> None:
    app()


if __name__ == "__main__":
    main()

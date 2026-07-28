"""``aire`` command line interface.

aire init <name>            scaffold a project
aire run <prompt>           one-shot generation with the configured model
aire evaluate <dataset>     run an evaluation suite
aire serve                  serve the project's build_target() via uvicorn
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


@app.command()
def doctor() -> None:
    """Check Python, dependencies, credentials, providers and configuration."""
    import importlib.util

    from aire.ai import AI

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
    import os

    for env_var, provider in (
        ("OPENAI_API_KEY", "openai"),
        ("ANTHROPIC_API_KEY", "anthropic"),
        ("HF_TOKEN", "huggingface"),
    ):
        present = bool(os.environ.get(env_var)) or (
            runtime.settings.credential(provider).api_key is not None
        )
        check(f"credentials:{provider}", present, env_var if not present else "configured")

    from aire.models.builtin import EchoModel, HashingEmbedder

    check("provider:mock", True, EchoModel().info.ref)
    check("embedder:local", True, f"dimension={HashingEmbedder().dimension}")
    check("config:loaded", True, f"project={runtime.settings.project}")

    failures = [
        c
        for c in checks
        if not c["ok"] and "optional" not in c["check"] and not c["check"].startswith("credentials")
    ]
    _echo_json({"checks": checks, "healthy": not failures})
    if failures:
        raise typer.Exit(code=1)


@app.command()
def version() -> None:
    """Print the aire version."""
    typer.echo(__version__)


def main() -> None:
    app()


if __name__ == "__main__":
    main()

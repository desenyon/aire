"""0.3.0 — agent-operable AI platform pillars."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from aire import AI
from aire.core.errors import ConfigurationError, PermissionDeniedError
from aire.data.dataset import Dataset
from aire.data.types import Record
from aire.graph.community import detect_communities, summarize_communities
from aire.graph.types import Entity, Relation
from aire.optimization.prompt_opt import optimize_prompt
from aire.project.lock import create_lock, load_lock, write_lock
from aire.safety.policy import PolicyEngine, PolicyRule
from aire.schedule import Scheduler
from aire.tools.types import SideEffect
from aire.training.hpo import SearchSpace, random_search
from aire.training.lm_trainer import LMTrainer
from aire.training.lora import LoRATrainer
from aire.workers import FileQueueWorker, InProcessWorker
from aire.workflows.hitl import hitl_node
from tests.conftest import arun


def test_version_030() -> None:
    from aire import __version__

    assert __version__ == "0.3.0"


def test_community_detection() -> None:
    entities = [
        Entity(name="Alice"),
        Entity(name="Bob"),
        Entity(name="Carol"),
        Entity(name="Dave"),
    ]
    relations = [
        Relation(subject="Alice", predicate="knows", object="Bob"),
        Relation(subject="Bob", predicate="knows", object="Alice"),
        Relation(subject="Carol", predicate="knows", object="Dave"),
    ]
    report = detect_communities(entities, relations)
    assert report.entity_count >= 4
    assert len(report.communities) >= 1
    refreshed = summarize_communities(report)
    assert all(c.summary for c in refreshed.communities)


def test_neo4j_missing_hint() -> None:
    if importlib.util.find_spec("neo4j") is not None:
        pytest.skip("neo4j installed")
    from aire.graph.neo4j_store import Neo4jGraphStore

    with pytest.raises(ConfigurationError, match="aire\\[neo4j\\]"):
        Neo4jGraphStore()


def test_workflow_hitl_node() -> None:
    wf = AI.workflow("hitl")
    hitl_node(wf, "gate", lambda x, ctx: x * 2)
    wf.approver = lambda name: True
    result = arun(wf.run(3))
    assert result.ok and result.output == 6

    wf2 = AI.workflow("hitl-deny")
    AI.workflows.hitl_node(wf2, "gate", lambda x, ctx: x)
    wf2.approver = lambda name: False
    denied = arun(wf2.run(1))
    assert not denied.ok


def test_lora_describe_without_peft() -> None:
    trainer = LoRATrainer("gpt2")
    desc = trainer.describe()
    assert desc["kind"] == "lora_trainer"
    if not desc["available"]:
        with pytest.raises(ConfigurationError, match="aire\\[peft\\]"):
            trainer.prepare()


def test_hpo_random_search() -> None:
    space = SearchSpace(
        discrete={"epochs": [1, 2]},
        continuous={"learning_rate": (0.001, 0.01)},
    )

    async def objective(params: dict) -> float:
        return float(params["epochs"]) + params["learning_rate"]

    result = arun(random_search(objective, space, n_trials=5, seed=1))
    assert result.best_score > 0
    assert "epochs" in result.best_params


def test_lm_trainer_toy() -> None:
    ds = Dataset([Record(text="hello world"), Record(text="aire rocks")])
    trainer = LMTrainer(backend="toy", config=__import__(
        "aire.training.trainer", fromlist=["TrainingConfig"]
    ).TrainingConfig(epochs=2))
    result = arun(trainer.fit(ds))
    assert result.epochs_completed == 2
    assert result.history


def test_prompt_opt() -> None:
    def evaluate(prompt: str) -> float:
        return float(len(prompt))

    result = arun(optimize_prompt("Answer briefly.", evaluate, generations=2, population=3, seed=0))
    assert result.best_score >= len("Answer briefly.")
    assert result.history


def test_redis_missing_hint() -> None:
    if importlib.util.find_spec("redis") is not None:
        pytest.skip("redis installed")
    from aire.optimization.redis_cache import RedisCacheBackend

    with pytest.raises(ConfigurationError, match="aire\\[redis\\]"):
        RedisCacheBackend()


def test_pypdf_missing_hint() -> None:
    if importlib.util.find_spec("pypdf") is not None:
        pytest.skip("pypdf installed")
    from aire.docs.pdf import load_pdf

    with pytest.raises(ConfigurationError, match="aire\\[pypdf\\]"):
        load_pdf("/tmp/nope.pdf")


def test_voice_agent_text_path() -> None:
    from aire.audio.voice import VoiceAgent

    agent = AI.agents.create_sync("mock:echo")
    voice = VoiceAgent(agent)
    turn = arun(voice.handle(text="hello"))
    assert turn.transcript == "hello"
    assert turn.response_text
    assert turn.audio is not None


def test_video_summarize_stub() -> None:
    from aire.vision.video import VideoPipeline

    pipe = VideoPipeline()
    summary = arun(pipe.summarize("https://example.com/video.mp4"))
    assert "offline stub" in summary.summary


def test_workers_in_process_and_file(tmp_path: Path) -> None:
    wf = AI.workflow("w")
    wf.add("a", lambda x, ctx: (x or 0) + 1)
    worker = InProcessWorker({"w": wf})
    result = arun(worker.submit("w", 10))
    assert result.job.status == "completed"
    assert result.job.result == 11

    fq = FileQueueWorker(tmp_path / "queue", {"w": wf})
    job = fq.enqueue("w", 5)
    drained = arun(fq.drain())
    assert len(drained) == 1
    assert drained[0].job.id == job.id
    assert drained[0].job.result == 6


def test_scheduler_tick() -> None:
    wf = AI.workflow("sched")
    wf.add("n", lambda x, ctx: "ok")
    sched = Scheduler()
    sched.register_workflow("sched", wf)
    sched.every(0.0, "sched", name="job")
    ran = arun(sched.tick(now=1.0))
    assert ran and ran[0]["ok"]


def test_otel_sdk_bridge_describe() -> None:
    from aire.observability.otel_sdk import SdkBridgeExporter

    exporter = SdkBridgeExporter(use_sdk=False)
    assert exporter.describe()["kind"] == "otel_sdk_bridge"


def test_doctor_live_cli() -> None:
    from typer.testing import CliRunner

    from aire.cli.main import app

    runner = CliRunner()
    result = runner.invoke(app, ["doctor", "--live"])
    assert result.exit_code == 0
    assert "live:mock:echo" in result.stdout


def test_ui_app_factory() -> None:
    if importlib.util.find_spec("fastapi") is None:
        pytest.skip("fastapi missing")
    app = AI.ui()
    assert app.title


def test_project_lock(tmp_path: Path) -> None:
    lock = create_lock("demo", model="mock:echo", embedder="local:hashing")
    path = write_lock(lock, tmp_path / "aire.lock")
    loaded = load_lock(path)
    assert loaded.get("model") == "mock:echo"
    assert AI.locks.describe()["file"] == "aire.lock"


def test_policy_engine() -> None:
    engine = PolicyEngine(
        [
            PolicyRule(name="deny_shell", action="deny", tool="shell"),
            PolicyRule(
                name="approve_write",
                action="require_approval",
                side_effect_at_or_above=SideEffect.REVERSIBLE_WRITE,
            ),
        ]
    )
    assert engine.decide(tool="shell") == "deny"
    with pytest.raises(PermissionDeniedError):
        engine.assert_allowed(tool="shell")
    assert engine.requires_approval(side_effect=SideEffect.EXTERNAL_SIDE_EFFECT)


def test_recipes_and_facade() -> None:
    rag = AI.recipe("rag")
    assert rag["kind"] == "rag"
    agent = AI.recipe("agent")
    assert agent["agent"].name
    assert "skills" in AI.describe()["namespaces"]
    assert AI.skills.get("research").name == "research"
    assert AI.workers.describe()["backends"]
    assert AI.schedule.describe()["kind"] == "schedule"


def test_scaffold_cli(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from aire.cli.main import app

    runner = CliRunner()
    result = runner.invoke(app, ["scaffold", "agent", "--name", "demo", "--dir", str(tmp_path)])
    assert result.exit_code == 0
    assert (tmp_path / "demo" / "main.py").is_file()
    assert (tmp_path / "demo" / "aire.lock").is_file()


def test_optuna_missing_hint() -> None:
    if importlib.util.find_spec("optuna") is not None:
        pytest.skip("optuna installed")
    from aire.training.hpo import SearchSpace, optuna_search

    with pytest.raises(ConfigurationError, match="aire\\[optuna\\]"):
        optuna_search(lambda p: 1.0, SearchSpace(discrete={"x": [1]}), n_trials=1)

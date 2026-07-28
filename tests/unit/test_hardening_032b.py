"""0.3.2B hardening: skills/session, LoRA fit, GraphRAG communities, gateway cache."""

from __future__ import annotations

from pathlib import Path

from aire.agents import Agent, DurableSession, apply_skill, default_skills
from aire.agents.topologies import debate, swarm
from aire.ai import AI
from aire.graph.community import detect_communities, summarize_communities_async
from aire.graph.pipeline import KnowledgeGraph
from aire.graph.types import Entity, Relation
from aire.models.builtin import EchoModel, HashingEmbedder
from aire.rag.types import Document
from aire.recipes import agent_recipe, finetune_recipe
from aire.training import DistillTrainer
from aire.training.lora import create_lora
from tests.conftest import arun


def test_lora_fit_dry_run() -> None:
    trainer = create_lora("gpt2", dry_run=True)
    result = trainer.fit(["hello world", "another sample"], epochs=2)
    assert result.epochs_completed == 2
    assert "fit" in trainer.describe()["methods"]


def test_skill_apply_binds_tools_and_prompt() -> None:
    agent = Agent(EchoModel())
    apply_skill(agent, "research")
    assert agent.registry.has("http_get")
    assert agent.config.system_prompt and "skill:research" in agent.config.system_prompt
    assert "research" in agent.describe()["skills"]


def test_agent_recipe_applies_skills() -> None:
    pack = agent_recipe(skills=["code"])
    agent = pack["agent"]
    assert agent.registry.has("calculator")
    assert agent._skills == ["code"]


def test_durable_session_persists(tmp_path: Path) -> None:
    path = tmp_path / "sess.json"
    agent = Agent(EchoModel(), session=path)
    result = arun(agent.run("ping"))
    assert result.output
    loaded = DurableSession(path)
    assert loaded.state.status == "completed"
    assert loaded.state.goal == "ping"
    assert loaded.state.result is not None


def test_graph_query_includes_communities(runtime) -> None:  # type: ignore[no-untyped-def]
    kg = KnowledgeGraph(runtime, model=EchoModel(), embedder=HashingEmbedder())
    arun(
        kg.ingest(
            [
                Document(text="Alice works at Acme. Bob works at Acme. Carol knows Alice."),
            ]
        )
    )
    answer = arun(kg.query("Who works at Acme?"))
    assert "communities" in answer.metadata
    assert answer.metadata["communities"]["communities"] >= 0


def test_summarize_communities_async() -> None:
    report = detect_communities(
        [Entity(name="A"), Entity(name="B")],
        [Relation(subject="A", predicate="knows", object="B")],
    )

    class _AsyncAsk(EchoModel):
        async def ask(self, prompt: str, **kwargs):  # type: ignore[no-untyped-def]
            return "Async community summary."

    updated = arun(summarize_communities_async(report, model=_AsyncAsk()))
    assert "Async community summary" in updated.communities[0].summary


def test_finetune_recipe_next_mentions_fit() -> None:
    pack = finetune_recipe(backend="lora", dry_run=True)
    assert "fit" in pack["next"]
    result = pack["trainer"].fit(["a", "b"], dry_run=True)
    assert result.epochs_completed >= 1


def test_distill_trainer() -> None:
    trainer = DistillTrainer()
    result = arun(trainer.fit([([1.0, 0.0], [0.8, 0.2]), ([0.5, 0.5], [0.4, 0.6])], epochs=2))
    assert result.epochs_completed == 2


def test_swarm_topology() -> None:
    a = Agent(EchoModel("a"), name="a")
    b = Agent(EchoModel("b"), name="b")
    result = arun(swarm([a, b], "goal"))
    assert result.mode == "swarm"
    assert len(result.transcripts) == 2


def test_debate_topology() -> None:
    a = Agent(EchoModel("a"), name="a")
    b = Agent(EchoModel("b"), name="b")
    result = arun(debate([a, b], "goal", rounds=1))
    assert result.mode == "debate"
    assert result.rounds >= 1


def test_gateway_describe_semantic_stats(runtime) -> None:  # type: ignore[no-untyped-def]
    from aire.deployment.gateway import Gateway

    gw = Gateway(runtime, chat_routes={"default": ["mock:echo"]}, semantic_cache=True)
    desc = gw.describe()
    assert desc["semantic_cache"]["enabled"] is True
    assert "hits" in desc["semantic_cache"]


def test_ai_agents_create_with_skills() -> None:
    agent = AI.agents.create_sync("mock:echo", skills=["research"])
    assert agent.registry.has("http_get")


def test_skills_registry_describe() -> None:
    desc = default_skills().describe()
    assert "apply" in desc["methods"]
    assert "research" in {s["name"] for s in desc["skills"]}

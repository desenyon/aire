"""Long-term agent memory: episodic log + semantic recall + consolidation.

Implements the agent :class:`~aire.agents.memory.Memory` interface, so any
agent can use it directly::

    memory = AI.memory.create(path=".aire/memory")
    agent = AI.agents.create_sync(model, memory=memory)

Conversation messages land in the episodic buffer (and on disk when a path is
set). :meth:`LongTermMemory.consolidate` folds episodes into durable semantic
facts with any model; :meth:`recall_semantic` retrieves by meaning, weighted
by salience and recency.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from aire.agents.memory import Memory
from aire.core.content import Message
from aire.core.serialization import iter_jsonl
from aire.memory.types import MemoryEntry, MemoryKind
from aire.rag.store import LocalVectorStore, VectorStore
from aire.rag.types import Chunk

if TYPE_CHECKING:
    from aire.models.base import EmbeddingModel, Model

_CONSOLIDATE_PROMPT = (
    "Distill the conversation episodes below into durable facts worth remembering "
    "long-term (user preferences, decisions, stable facts). Return at most {max_facts} "
    "facts, each one self-contained sentence. Skip transient chatter.\n\nEpisodes:\n{episodes}"
)


class _Facts(BaseModel):
    facts: list[str]


class LongTermMemory(Memory):
    """Episodic + semantic memory with model-driven consolidation."""

    def __init__(
        self,
        *,
        embedder: EmbeddingModel | None = None,
        store: VectorStore | None = None,
        path: str | Path | None = None,
        window: int = 200,
    ) -> None:
        self.embedder = embedder
        self.store = store or LocalVectorStore(
            Path(path) / "semantic.json" if path else None, name="memory"
        )
        self.path = Path(path) if path else None
        self.window = window
        self._episodes: list[Message] = []
        if self.path:
            episodes_file = self.path / "episodes.jsonl"
            if episodes_file.is_file():
                for row in iter_jsonl(episodes_file):
                    self._episodes.append(Message.model_validate(row))

    # -- embedding -------------------------------------------------------------------

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        if self.embedder is None:
            from aire.models.builtin import HashingEmbedder

            self.embedder = HashingEmbedder()
        from aire.models.types import EmbeddingRequest

        return (await self.embedder.embed(EmbeddingRequest(inputs=texts))).vectors

    def _persist_store(self) -> None:
        if self.path and isinstance(self.store, LocalVectorStore):
            self.path.mkdir(parents=True, exist_ok=True)
            self.store.save(self.path / "semantic.json")

    # -- agent Memory interface --------------------------------------------------------

    async def add(self, message: Message) -> None:
        self._episodes.append(message)
        if len(self._episodes) > self.window:
            del self._episodes[: len(self._episodes) - self.window]
        if self.path:
            self.path.mkdir(parents=True, exist_ok=True)
            with (self.path / "episodes.jsonl").open("a") as fh:
                fh.write(message.model_dump_json() + "\n")

    async def recall(self, *, limit: int | None = None) -> list[Message]:
        return self._episodes[-limit:] if limit else list(self._episodes)

    async def clear(self) -> None:
        self._episodes.clear()
        await self.store.clear()
        self._persist_store()
        self._rewrite_episodes_file()

    # -- semantic memory -----------------------------------------------------------------

    async def remember(
        self,
        text: str,
        *,
        kind: MemoryKind | str = MemoryKind.SEMANTIC,
        salience: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEntry:
        """Store a durable memory, embedded for semantic recall."""
        entry = MemoryEntry(
            kind=MemoryKind(kind), text=text, salience=salience, metadata=metadata or {}
        )
        vectors = await self._embed([text])
        chunk = Chunk(
            id=entry.id,
            text=text,
            metadata={
                "kind": str(entry.kind),
                "salience": entry.salience,
                "created_at": entry.created_at,
                **entry.metadata,
            },
            embedding=vectors[0],
        )
        await self.store.upsert([chunk])
        self._persist_store()
        return entry

    async def recall_semantic(
        self, query: str, *, k: int = 5, kind: MemoryKind | str | None = None
    ) -> list[MemoryEntry]:
        """Recall memories by meaning, weighted by salience and recency."""
        vectors = await self._embed([query])
        filter_ = {"kind": str(MemoryKind(kind))} if kind else None
        hits = await self.store.search(vectors[0], k=k * 3, filter=filter_)
        now = time.time()

        def _weight(hit: Any) -> float:
            salience = float(hit.chunk.metadata.get("salience", 1.0))
            created = float(hit.chunk.metadata.get("created_at", now))
            age_days = max(0.0, (now - created) / 86400.0)
            recency = 1.0 / (1.0 + age_days / 30.0)  # half-life ≈ 30 days
            return float(hit.score * salience * (0.5 + 0.5 * recency))

        hits.sort(key=_weight, reverse=True)
        return [
            MemoryEntry(
                id=hit.chunk.id,
                kind=MemoryKind(str(hit.chunk.metadata.get("kind", "semantic"))),
                text=hit.chunk.text,
                salience=float(hit.chunk.metadata.get("salience", 1.0)),
                created_at=float(hit.chunk.metadata.get("created_at", now)),
                metadata={
                    k_: v
                    for k_, v in hit.chunk.metadata.items()
                    if k_ not in {"kind", "salience", "created_at"}
                },
            )
            for hit in hits[:k]
        ]

    # -- consolidation ---------------------------------------------------------------------

    async def consolidate(
        self, model: Model, *, max_facts: int = 8, keep: int = 20
    ) -> list[MemoryEntry]:
        """Fold recent episodes into durable semantic facts with a model.

        Returns the new semantic entries. Episodes beyond ``keep`` are dropped
        after consolidation (they live on in the distilled facts).
        """
        episodes = self._episodes[:-keep] if len(self._episodes) > keep else []
        if not episodes:
            return []
        transcript = "\n".join(f"{m.role}: {m.text_content}" for m in episodes)
        result = await model.ask_structured(
            _CONSOLIDATE_PROMPT.format(max_facts=max_facts, episodes=transcript), _Facts
        )
        facts = _Facts.model_validate(result)
        entries = []
        for fact in facts.facts[:max_facts]:
            entries.append(await self.remember(fact, kind=MemoryKind.SEMANTIC, salience=1.5))
        del self._episodes[:-keep]
        self._rewrite_episodes_file()
        return entries

    def _rewrite_episodes_file(self) -> None:
        """Rewrite the compacted episodic store (not append-forever)."""
        if not self.path:
            return
        self.path.mkdir(parents=True, exist_ok=True)
        episodes_file = self.path / "episodes.jsonl"
        with episodes_file.open("w") as fh:
            for message in self._episodes:
                fh.write(message.model_dump_json() + "\n")

    async def add_procedural(
        self,
        text: str,
        *,
        salience: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEntry:
        """Store a procedural memory (how to do something / successful plan)."""
        return await self.remember(
            text, kind=MemoryKind.PROCEDURAL, salience=salience, metadata=metadata
        )

    async def recall_procedural(self, query: str, *, k: int = 5) -> list[MemoryEntry]:
        """Recall procedural memories by meaning."""
        return await self.recall_semantic(query, k=k, kind=MemoryKind.PROCEDURAL)

    async def count(self) -> dict[str, int]:
        return {"episodes": len(self._episodes), "semantic": await self.store.count()}

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "memory",
            "type": "long_term",
            "kinds": ["episodic", "semantic", "procedural"],
            "window": self.window,
            "path": str(self.path) if self.path else None,
            "episodes": len(self._episodes),
        }

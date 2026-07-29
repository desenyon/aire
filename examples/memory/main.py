"""Long-term memory: episodic log + semantic recall (hashing embedder)."""

from __future__ import annotations

import tempfile
from pathlib import Path

from aire import AI
from aire.core.content import Message
from aire.models.base import run_sync


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        memory = AI.memory.create(path=tmp)
        run_sync(memory.add(Message.text("user", "I prefer dark mode and short answers.")))
        run_sync(memory.add(Message.text("assistant", "Got it — dark mode, brief replies.")))
        run_sync(
            memory.remember(
                "User prefers dark mode UI",
                kind="semantic",
                salience=0.9,
            )
        )
        episodes = run_sync(memory.recall(limit=5))
        hits = run_sync(memory.recall_semantic("dark mode preference", k=3))
        print("path:", Path(tmp))
        print("episodes:", len(episodes))
        print("semantic hits:", [h.text[:60] for h in hits])
        print("describe:", memory.describe())


if __name__ == "__main__":
    main()

"""Entity/relation extraction: model-driven and zero-dependency lexical.

The lexical extractor keeps the whole GraphRAG pipeline useful offline and in
CI (capitalized multi-word phrases become entities; sentence co-occurrence
becomes relations). The model extractor produces semantically typed triples
through any aire model with structured output.
"""

from __future__ import annotations

import re
from typing import Any, Protocol

from aire.graph.types import Extraction
from aire.models.base import Model

EXTRACT_PROMPT = (
    "Extract the entities and relations from the text below. "
    "Entities have a name and a short type (person, org, product, concept, place). "
    "Relations connect two entity names with a short predicate in snake_case. "
    "Only include facts explicitly stated in the text.\n\nText:\n{text}"
)


class GraphExtractor(Protocol):
    """Anything that turns text into an :class:`Extraction`."""

    async def extract(self, text: str) -> Extraction: ...

    def describe(self) -> dict[str, Any]: ...


class ModelGraphExtractor:
    """Extract typed triples with any model via validated structured output."""

    def __init__(self, model: Model, *, prompt_template: str = EXTRACT_PROMPT) -> None:
        self.model = model
        self.prompt_template = prompt_template

    async def extract(self, text: str) -> Extraction:
        result = await self.model.ask_structured(
            self.prompt_template.format(text=text), Extraction, retries=1
        )
        return Extraction.model_validate(result)

    def describe(self) -> dict[str, Any]:
        return {"kind": "graph_extractor", "type": "model", "model": self.model.info.ref}


_CAPITALIZED = re.compile(r"\b([A-Z][a-z0-9]+(?:[ \t]+[A-Z][a-z0-9]+){0,3})\b")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


class LexicalGraphExtractor:
    """Heuristic extractor: capitalized phrases as entities, sentence
    co-occurrence as ``related_to`` relations. Deterministic and offline."""

    def __init__(self, *, max_entities_per_chunk: int = 12) -> None:
        self.max_entities_per_chunk = max_entities_per_chunk

    async def extract(self, text: str) -> Extraction:
        from aire.graph.types import ExtractedEntity, ExtractedRelation

        sentences = _SENTENCE_SPLIT.split(text.strip())
        seen: dict[str, str] = {}
        relations: list[ExtractedRelation] = []
        for sentence in sentences:
            phrases = []
            for match in _CAPITALIZED.finditer(sentence):
                phrase = match.group(1).strip()
                if phrase.lower() in {"the", "a", "an", "in", "on", "at", "for", "to"}:
                    continue
                phrases.append(phrase)
                seen.setdefault(phrase.lower(), phrase)
            unique = list(dict.fromkeys(phrases))
            for i, subject in enumerate(unique):
                for obj in unique[i + 1 :]:
                    relations.append(
                        ExtractedRelation(subject=subject, predicate="related_to", object=obj)
                    )
        entities = [
            ExtractedEntity(name=name)
            for name in list(seen.values())[: self.max_entities_per_chunk]
        ]
        return Extraction(entities=entities, relations=relations)

    def describe(self) -> dict[str, Any]:
        return {"kind": "graph_extractor", "type": "lexical", "offline": True}

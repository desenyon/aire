"""Community detection + summary generation for GraphRAG (offline lexical).

Uses connected-component / label-propagation style clustering over the
entity-relation graph — no networkx or GPU required.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from pydantic import BaseModel, Field

from aire.graph.types import Entity, Relation, Subgraph


class Community(BaseModel):
    """A cluster of related entities with an optional summary."""

    id: str
    entities: list[str] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    summary: str = ""
    size: int = 0


class CommunityReport(BaseModel):
    communities: list[Community] = Field(default_factory=list)
    algorithm: str = "label_propagation"
    entity_count: int = 0
    relation_count: int = 0

    def describe(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "communities": len(self.communities),
            "entities": self.entity_count,
            "relations": self.relation_count,
            "sizes": [c.size for c in self.communities],
        }


def detect_communities(  # noqa: C901
    entities: list[Entity],
    relations: list[Relation],
    *,
    max_iterations: int = 20,
    min_size: int = 1,
) -> CommunityReport:
    """Label-propagation community detection over undirected entity links."""
    names = {e.key for e in entities}
    for r in relations:
        names.add(r.subject.strip().lower())
        names.add(r.object.strip().lower())
    if not names:
        return CommunityReport()

    adjacency: dict[str, set[str]] = defaultdict(set)
    for r in relations:
        a = r.subject.strip().lower()
        b = r.object.strip().lower()
        if a == b:
            continue
        adjacency[a].add(b)
        adjacency[b].add(a)
    for n in names:
        adjacency.setdefault(n, set())

    labels = {n: n for n in names}
    order = sorted(names)
    for _ in range(max_iterations):
        changed = 0
        for node in order:
            neighbors = adjacency[node]
            if not neighbors:
                continue
            votes: dict[str, int] = defaultdict(int)
            for nb in neighbors:
                votes[labels[nb]] += 1
            best = max(votes.items(), key=lambda kv: (kv[1], kv[0]))[0]
            if labels[node] != best:
                labels[node] = best
                changed += 1
        if changed == 0:
            break

    buckets: dict[str, list[str]] = defaultdict(list)
    for node, label in labels.items():
        buckets[label].append(node)

    name_lookup = {e.key: e.name for e in entities}
    rel_by_pair: list[Relation] = list(relations)
    communities: list[Community] = []
    for idx, (_label, members) in enumerate(sorted(buckets.items(), key=lambda kv: -len(kv[1]))):
        if len(members) < min_size:
            continue
        member_set = set(members)
        community_rels = [
            r
            for r in rel_by_pair
            if r.subject.strip().lower() in member_set and r.object.strip().lower() in member_set
        ]
        display = [name_lookup.get(m, m) for m in sorted(members)]
        communities.append(
            Community(
                id=f"community-{idx}",
                entities=display,
                relations=community_rels,
                size=len(members),
                summary=_lexical_summary(display, community_rels),
            )
        )
    return CommunityReport(
        communities=communities,
        entity_count=len(names),
        relation_count=len(relations),
    )


def summarize_communities(
    report: CommunityReport,
    *,
    model: Any | None = None,
) -> CommunityReport:
    """Refresh community summaries. Offline default is lexical; optional model."""
    updated: list[Community] = []
    for community in report.communities:
        if model is None:
            summary = _lexical_summary(community.entities, community.relations)
        else:
            summary = _model_summary(model, community)
        updated.append(community.model_copy(update={"summary": summary}))
    return report.model_copy(update={"communities": updated})


def communities_from_subgraph(subgraph: Subgraph, **options: Any) -> CommunityReport:
    """Convenience: detect communities from a :class:`Subgraph`."""
    return detect_communities(subgraph.entities, subgraph.relations, **options)


def _lexical_summary(entities: list[str], relations: list[Relation]) -> str:
    if not entities:
        return ""
    head = ", ".join(entities[:8])
    if len(entities) > 8:
        head += f" (+{len(entities) - 8} more)"
    if not relations:
        return f"Community of {len(entities)} entities: {head}."
    preds = sorted({r.predicate for r in relations})[:5]
    sample = "; ".join(r.as_text() for r in relations[:3])
    return (
        f"Community of {len(entities)} entities ({head}). "
        f"Key relations ({', '.join(preds)}): {sample}."
    )


def _model_summary(model: Any, community: Community) -> str:
    prompt = (
        "Summarize this knowledge-graph community in 1-2 sentences.\n"
        f"Entities: {', '.join(community.entities[:20])}\n"
        f"Relations:\n" + "\n".join(r.as_text() for r in community.relations[:20])
    )
    ask = getattr(model, "ask", None)
    if callable(ask):
        import inspect

        result = ask(prompt)
        if inspect.isawaitable(result):
            # Callers that pass an async model should use summarize_communities
            # from an async context; fall back to lexical here.
            return _lexical_summary(community.entities, community.relations)
        return str(result).strip()
    return _lexical_summary(community.entities, community.relations)


def describe() -> dict[str, Any]:
    return {
        "kind": "graph_community",
        "algorithms": ["label_propagation"],
        "offline": True,
        "summary": ["lexical", "model (optional)"],
    }

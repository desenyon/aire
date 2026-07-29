"""GraphRAG communities: offline label-propagation clustering."""

from __future__ import annotations

from aire import AI
from aire.graph.types import Entity, Relation


def main() -> None:
    entities = [
        Entity(name="Alice", type="person"),
        Entity(name="Bob", type="person"),
        Entity(name="Acme", type="org"),
        Entity(name="BetaCorp", type="org"),
    ]
    relations = [
        Relation(subject="Alice", predicate="works_at", object="Acme"),
        Relation(subject="Bob", predicate="works_at", object="Acme"),
        Relation(subject="Acme", predicate="competes_with", object="BetaCorp"),
    ]
    report = AI.graph.communities(entities, relations)
    print(report.describe())
    for community in report.communities:
        print(
            f"- {community.id}: size={community.size} "
            f"entities={community.entities} summary={community.summary!r}"
        )


if __name__ == "__main__":
    main()

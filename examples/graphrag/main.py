"""GraphRAG offline: documents → knowledge graph → grounded, cited answers.

Runs fully offline with the lexical extractor (no model calls for extraction)
and the mock embedder/model defaults. Swap in a real model ref for typed,
semantic triple extraction:

    graph = AI.graph.create(model="ollama:llama3.2")  # model-driven triples
"""

from aire import AI


def main() -> None:
    graph = AI.graph.create()  # embedded sqlite graph store + lexical extractor

    report = AI.graph.describe()
    print("graph subsystem:", report)

    docs = [
        "Ada Lovelace wrote extensive notes on the Analytical Engine.",
        "The Analytical Engine was designed by Charles Babbage in London.",
        "Grace Hopper popularized the idea of machine-independent programming.",
    ]

    import asyncio

    async def run() -> None:
        index = await graph.ingest(docs)
        print(f"\ningested: {index.model_dump_json(indent=2)}")

        question = "What did Ada Lovelace write about?"
        facts = await graph.subgraph(question)
        print(f"\ngraph facts for {question!r}:")
        print(facts.as_context())

        answer = await graph.query(question)
        print(f"\nanswer (via {answer.model}):\n{answer.text[:400]}")
        print(f"\ncitations: {len(answer.citations)}")
        for citation in answer.citations:
            print(f"  - [{citation.source}] {citation.excerpt[:80]}...")

    asyncio.run(run())


if __name__ == "__main__":
    main()

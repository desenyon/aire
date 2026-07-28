"""RAG assistant — the aire vertical slice, fully offline.

Run:  python examples/rag_assistant/main.py

Ingests text documents, indexes them with the hashing embedder into the local
vector store, answers a question with citations using the offline echo model,
and prints an evaluation report. Swap "mock:echo" for "openai:gpt-4o-mini" to
go live — nothing else changes.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from aire import AI

DOCS = {
    "authentication.md": (
        "aire authenticates to providers through environment variables. "
        "Set OPENAI_API_KEY or ANTHROPIC_API_KEY before resolving models. "
        "Credentials are read from Settings.providers and are never logged."
    ),
    "refunds.md": (
        "Refunds are allowed within 30 days of purchase with a receipt. "
        "Refunds after 30 days require manager approval and are credited "
        "to the original payment method within 5 business days."
    ),
    "deployment.md": (
        "Deploy any agent, knowledge pipeline or model with AI.deploy.api(). "
        "The generated FastAPI app exposes /health, /ready, /manifest and "
        "/metrics endpoints plus optional bearer auth and rate limiting."
    ),
}


def main() -> None:
    docs_dir = Path(tempfile.mkdtemp(prefix="aire-docs-"))
    for name, text in DOCS.items():
        (docs_dir / name).write_text(text)

    assistant = (
        AI.project("knowledge_assistant")
        .documents(str(docs_dir))
        .model("mock:echo")
        .vector_store("local:default")
        .citations(True)
    )

    report = assistant.index()
    print(f"indexed {report.documents} documents into {report.chunks} chunks")

    answer = assistant.ask("What does the documentation say about authentication?")
    print("\nanswer:", answer.text)
    for citation in answer.citations:
        print(f"  citation: {citation.source} (score={citation.score:.2f})")

    evals = docs_dir / "evals.jsonl"
    evals.write_text(
        '{"input": "How long is the refund window?", "expected": "30 days"}\n'
        '{"input": "Which endpoints does deployment expose?", "expected": "/health"}\n'
    )
    report = assistant.evaluate(str(evals), metrics=["contains", "groundedness"])
    print(f"\nevaluation: {report.total} cases, {report.failures} failures")
    for name, summary in report.metric_summary().items():
        print(f"  {name}: mean={summary['mean']:.2f}")


if __name__ == "__main__":
    main()

"""RAG assistant offline with local hashing + mock:echo."""

from pathlib import Path

from aire import AI


def main() -> None:
    docs = Path(__file__).parent / "sample_docs"
    docs.mkdir(exist_ok=True)
    (docs / "auth.md").write_text(
        "Authentication uses API keys from the environment.\n",
        encoding="utf-8",
    )
    assistant = (
        AI.project("rag-demo")
        .documents(str(docs))
        .model("mock:echo")
        .embedder("local:hashing")
        .vector_store("local")
    )
    report = assistant.index()
    print("indexed:", report)
    answer = assistant.ask("How does authentication work?")
    print("answer:", answer.text)
    print("citations:", len(answer.citations))


if __name__ == "__main__":
    main()

# RAG

The RAG backbone is `Knowledge` (`AI.rag.create`) and the fluent `Assistant` (`AI.project`).

## Quick path

```python
from aire import AI

assistant = (
    AI.project("docs")
    .documents("./docs")
    .model("mock:echo")
    .embedder("local:hashing")
    .vector_store("local")
)
assistant.index()
answer = assistant.ask("What is aire?")
print(answer.text, answer.citations)
```

## `Knowledge` pipeline

```python
from aire import AI
from aire.models.base import run_sync
from aire.rag.types import Document

kb = AI.rag.create(embedder=AI.models.embedder_sync("local:hashing"))
run_sync(kb.ingest([Document(text="Auth uses API keys.", metadata={"source": "sec.md"})]))
answer = run_sync(kb.ask("How does auth work?", model="mock:echo"))
```

Stages (all replaceable): chunker → embedder → vector store → optional rewrite → retrieve → rerank → compress → answer model.

### Stores

- Default: in-process `LocalVectorStore` (`AI.rag.vector_store("local:default")`)
- Integrations (optional extras): Qdrant, Chroma, Milvus, Weaviate, pgvector, …

Remote adapters vary in how well they honor filters/ACL — prefer local for offline tests.

## Rewrite / compress / ACL / incremental

| Feature | Reality (0.3.5) |
|---------|-----------------|
| Rewriters | Lexical multi-query / model-backed options; some names fall back without a model |
| Compressors | Truncate / extractive / model; extractive is not a full LLM extractive pack |
| ACL | Metadata convention + `split_acl_filter` / post-filter helpers; identity binding from gateway auth is thin |
| Incremental | `AI.rag.incremental(knowledge)` helpers; depends on store `delete_by_document` support |

```python
from aire.rag.filters import split_acl_filter

store_filter, acl = split_acl_filter({"tenant": "a", "__acl__": {"roles": ["reader"]}})
```

Chunkers: `fixed`, `sentence`, `recursive`, `semantic` / `semantic_sentence`.  
`get_chunker("semantic")` **without** an embedder uses sentence-boundary packing and advertises that in `describe()` / a `UserWarning`.

## GraphRAG

`AI.graph.create(...)` builds a `KnowledgeGraph` (lexical extractor by default; Neo4j optional). Community detection is label-propagation style.

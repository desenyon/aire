# Examples guide

Runnable offline samples under [`examples/`](https://github.com/desenyon/aire/tree/main/examples) use `mock:echo` and `local:hashing` — no API keys.

| Example | What it shows |
|---------|----------------|
| [`00_quickstart`](https://github.com/desenyon/aire/tree/main/examples/00_quickstart) | Facade + one-shot generate |
| [`rag_assistant`](https://github.com/desenyon/aire/tree/main/examples/rag_assistant) | `AI.project` index + ask |
| [`agent_tools`](https://github.com/desenyon/aire/tree/main/examples/agent_tools) | Agent + calculator builtin |
| [`hitl`](https://github.com/desenyon/aire/tree/main/examples/hitl) | RuleApprover HITL |
| [`memory`](https://github.com/desenyon/aire/tree/main/examples/memory) | Long-term memory |
| [`mcp_roundtrip`](https://github.com/desenyon/aire/tree/main/examples/mcp_roundtrip) | MCP in-process round-trip |
| [`graphrag_communities`](https://github.com/desenyon/aire/tree/main/examples/graphrag_communities) | Graph communities |
| [`graph_store`](https://github.com/desenyon/aire/tree/main/examples/graph_store) | SQLite / Neo4j note |
| [`multimodal_offline`](https://github.com/desenyon/aire/tree/main/examples/multimodal_offline) | Multimodal capability probes |
| [`lora_dry_run`](https://github.com/desenyon/aire/tree/main/examples/lora_dry_run) | LoRA dry-run |
| [`gateway_offline`](https://github.com/desenyon/aire/tree/main/examples/gateway_offline) | Build gateway app (no long bind) |
| [`eval_gates`](https://github.com/desenyon/aire/tree/main/examples/eval_gates) | Metrics + `check_gates` |
| [`openapi_tools`](https://github.com/desenyon/aire/tree/main/examples/openapi_tools) | `load_openapi` from dict |

```bash
cd examples/00_quickstart && python main.py
```

Index: [`examples/README.md`](https://github.com/desenyon/aire/blob/main/examples/README.md).

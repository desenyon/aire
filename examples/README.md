# aire examples

All examples run **offline** with `mock:echo` / hashing embedders unless noted.

```bash
pip install -e ".[dev]"
cd examples/<name> && python main.py
```

| Directory | Description |
|-----------|-------------|
| [`00_quickstart`](00_quickstart/) | One-shot generate |
| [`rag_assistant`](rag_assistant/) | Project index + ask |
| [`agent_tools`](agent_tools/) | Agent + calculator |
| [`hitl`](hitl/) | RuleApprover HITL policy |
| [`memory`](memory/) | LongTermMemory episodic + semantic |
| [`mcp_roundtrip`](mcp_roundtrip/) | In-process MCP tools/list|call |
| [`graphrag_communities`](graphrag_communities/) | Label-propagation communities |
| [`graph_store`](graph_store/) | SQLite graph store (Neo4j via live env) |
| [`multimodal_offline`](multimodal_offline/) | Vision/audio capability probes |
| [`lora_dry_run`](lora_dry_run/) | LoRATrainer dry_run without peft |
| [`gateway_offline`](gateway_offline/) | Construct gateway app |
| [`eval_gates`](eval_gates/) | Metrics + gates |
| [`openapi_tools`](openapi_tools/) | OpenAPI → tools from dict |

Docs index: [`docs/GUIDE.md`](../docs/GUIDE.md).

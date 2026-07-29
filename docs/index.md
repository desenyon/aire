# aire

**aire** (v0.3.5) is an agent-first AI creation library: one consistent Python interface from a local prototype to a deployed system.

```python
from aire import AI

assistant = AI.project("docs-bot").documents("./docs").model("mock:echo")
assistant.index()
answer = assistant.ask("How do I authenticate?")
print(answer.text, answer.citations)
```

## Why aire

- **Offline-first** — `mock:echo` and `local:hashing` run the full stack with zero credentials.
- **One facade** — `AI.models`, `AI.rag`, `AI.agents`, `AI.gateway`, …
- **Discoverable** — almost every object exposes `.describe()`.
- **Honest about stubs** — see [Honesty](honesty.md) for experimental surfaces.

## Quick paths

| Goal | Start here |
|------|------------|
| Public surface | [Public API](public_api.md) |
| Agents & tools | [Agents](agents.md) |
| Retrieval | [RAG](rag.md) |
| OpenAI-compat proxy | [Gateway](gateway.md) |
| CLI | [CLI](cli.md) |
| Runnable samples | [GUIDE](GUIDE.md) / `examples/` |

Install:

```bash
pip install -e ".[dev]"
aire doctor
```

Run offline examples with `mock:echo` — see `examples/README.md`.

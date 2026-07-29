# CLI

Entry point: `aire` → `aire.cli.main:app` (Typer).

| Command | Purpose |
|---------|---------|
| `aire init <name>` | Scaffold project (`aire.yaml`, `app.py`, `docs/`, `evals/`) |
| `aire run <prompt>` | One-shot generation (`--model`, `--config`) |
| `aire evaluate <dataset>` | Eval suite (`--model`, `--metrics`, `--output`) |
| `aire serve` | Serve `build_target()` from `app.py` via FastAPI/uvicorn |
| `aire gateway` | OpenAI-compat gateway (`-m/--model`, `-a/--alias`, `--embed-alias`, `--routing`, `--objective`, `--auth-token`, `--rate-limit`) |
| `aire mcp-serve` | Builtin + registered tools over MCP stdio |
| `aire inspect [what]` | `models` \| `tools` \| `plugins` \| `config` \| `all` |
| `aire plugins` | List discovered/loaded plugins |
| `aire doctor [--live]` | Env/deps/credentials; `--live` probes mock/local (+ optional cloud/Ollama) |
| `aire scaffold <kind>` | Recipe snippet: `rag` \| `agent` \| `finetune` \| `gateway` \| `workflow` |
| `aire version` | Print version |
| `aire deploy scale` | Docker Compose + K8s scale pack |
| `aire foundation` | **Toy** architecture only (`--describe` for catalog) — not pretrained weights |
| `aire analytics` | Analytics report (**demo counters** unless process metrics / `AIRE_ANALYTICS` attached) |

Examples:

```bash
aire init mybot --model mock:echo
aire doctor
aire run 'hello' --model mock:echo
aire gateway -m mock:echo
aire foundation --describe
```

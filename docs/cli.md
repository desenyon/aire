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
| `aire doctor [--smoke] [--live] [--output PATH] [--config PATH]` | Versioned diagnostic report; offline smoke or explicit live probes |
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

## Diagnostic evidence

`aire doctor --smoke --output doctor.json` exercises mock generation, local
hashing embeddings, a real tool and cited retrieval without network or model
downloads. The JSON report records version, Python, `healthy`, and explicit
`passed`, `failed` and `skipped` statuses. The saved report matches stdout;
required failures exit 1. Invalid configuration is reported without echoing
its contents. `--config` requires the requested file/directory to exist.

Default doctor checks availability/configuration without executing model calls.
Unrelated missing credentials and optional serving dependencies are skipped.
`--live` adds the offline smoke, probes OpenAI/Anthropic only when a resolved
configuration/environment credential exists, and probes local Ollama. Cloud
health probes are bounded to ten seconds each and may incur usage; errors omit
provider bodies and credentials. Client resources are closed after probing.

`aire inspect config` excludes provider settings and masks common secret-bearing
keys recursively, including gateway bearer tokens. Add custom keys to
`observability.mask_fields`; masking cannot detect secrets in arbitrary text.
Unknown inspection selectors exit 2.

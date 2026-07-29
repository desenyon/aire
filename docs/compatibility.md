# Compatibility matrix

| Component | Supported |
|-----------|-----------|
| Python | **3.11**, **3.12**, **3.13** (CI matrix on every PR) |
| OS | Linux / macOS (Windows best-effort) |
| Default model | `mock:echo` offline; providers via extras |
| Docs site | MkDocs Material → GitHub Pages (`Docs` workflow) |

## Optional extras

| Extra | Purpose |
|-------|---------|
| `serve` | FastAPI gateway / UI |
| `redis` | Redis cache + queue worker |
| `eval` | sacreBLEU + sentence-transformers cross-encoder |
| `ocr` | PDF OCR (`pypdfium2` + pytesseract) |
| `pgvector` / `neo4j` / `peft` / `training` / `torch` | Store / train backends |

## Live integrations

Opt-in probes live under `tests/live/` and are skipped unless env vars are set:

- `AIRE_LIVE_REDIS`
- `AIRE_LIVE_PGVECTOR`
- `AIRE_LIVE_NEO4J`
- `AIRE_LIVE_QDRANT`

Run: `pytest -m live`.

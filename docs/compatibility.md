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
- `AIRE_LIVE_CHROMA`
- `AIRE_LIVE_WEAVIATE`
- `AIRE_LIVE_MILVUS`

Run: `pytest -m live`.

Optional CPU backend checks run with `OMP_NUM_THREADS=1`,
`OPENBLAS_NUM_THREADS=1` and `KERAS_BACKEND=torch`; mixing unbounded native runtimes
in one process is not a tested performance configuration. GPU quantization needs
compatible bitsandbytes hardware and is not verified by CPU or dry-run tests.
The [capability matrix](capabilities.md) records concrete evidence and limits.

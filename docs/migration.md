# Migration guides

## 0.3.4 → 0.3.5

Honesty + depth release. Most public names remain; behaviour becomes stricter.

| Change | Action |
|--------|--------|
| Guardrails auto-wire on `Knowledge.ask` / `create_gateway` | Pass `guardrails=False` to opt out; tune via `SafetyConfig` |
| `bleu` metric | Now BLEU-4 + brevity penalty; old F1 approx is `bleu_approx` |
| `cross_encoder` reranker | Defaults to HF CrossEncoder (`aire[eval]`); pass `model=` LLM for old scorer, or use `reranker="model"` |
| `embedding_similarity` / `nli_faithfulness` / `model_judge` | Raise `ConfigurationError` when judge/embedder missing (no silent 0.0) |
| Gateway images | Returns **501** without image capability |
| Workers | `create_worker("redis")` available; `"sqs"` raises until boto3 is bundled |
| Foundation | `foundation` remains toy; use `foundation_pretrained` / `from_pretrained` for HF weights |
| Scale pack Postgres | `AIRE_DATABASE_URL` auto-wires `PgVectorStore` Knowledge in generated `app.py` |
| Anthropic `/v1/messages` | Response emits real `tool_use` blocks; request maps `tool_result` → `role=tool` |

## Pre-0.3.4

See [CHANGELOG](https://github.com/desenyon/aire/blob/main/CHANGELOG.md). Prefer upgrading to the latest 0.3.x patch before jumping minors.

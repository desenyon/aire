# Honesty — experimental & stub surfaces

aire documents incomplete areas instead of silent magic. Prefer these APIs knowing their limits.

| Surface | Behavior |
|---------|----------|
| **Foundation toy** | `AI.training.foundation` / `aire foundation` builds random-init / config-driven stacks. `describe()` → `kind: foundation_toy_architecture`. **Not** pretrained GPT/LLaMA weights. |
| **TTS echo** | `EchoTTSBackend` returns `data:text/plain,...` placeholders with `stub=True`. Real TTS needs a model with `Capability.TEXT_TO_SPEECH`. |
| **Video stub** | `VideoPipeline.summarize` without a vision/video-capable model returns offline stub summaries (`stub=True`). |
| **Lexical metrics** | `semantic_overlap` is **token-set F1** (prefer `token_overlap`). `bleu` is sentence BLEU-4+BP (pure Python); use `sacrebleu` / `aire[eval]` for sacreBLEU. `groundedness` / `faithfulness` stay lexical; prefer `nli_faithfulness` with a judge. |
| **embedding_similarity** | Requires `ctx.embedder`; raises `ConfigurationError` if missing (no silent lexical fallback). |
| **nli_faithfulness** | Requires `ctx.judge`; raises if missing. |
| **cross_encoder rerank** | `hf_cross_encoder` / `cross_encoder` need `sentence-transformers` (`aire[eval]`). Pass `model=` (LLM) to keep the old prompt scorer via `model` reranker. |
| **Semantic chunker** | Without `embedder=`, sentence-boundary approximation + warning; `describe()` reports `mode: sentence_approximation`. |
| **MCP subset** | Stdio + streamable HTTP tools subset — not full MCP (sampling/roots/progress still open). |
| **Workers** | `in_process`, `file`, `redis` (`aire[redis]`); SQS raises until boto3 is bundled. |
| **Gateway images** | Returns **501** when `IMAGE_GENERATION` capability is absent. |
| **CLI analytics** | Demo counters unless live metrics are attached. |
| **web_search** | DuckDuckGo HTML scrape, not an official search API. |
| **Guardrails auto-wire** | `Knowledge.ask` and `create_gateway` apply `SafetyConfig` regex rails by default (`guardrails=False` to disable). Model classifiers are opt-in via `AI.safety.guardrails("model_injection", model=...)`. |

Also see root [GAPS.md](https://github.com/desenyon/aire/blob/main/GAPS.md) for the prioritized backlog.

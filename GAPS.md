# aire — Gap Backlog (release 0.3.5)

**Version:** 0.3.5  
**Status legend:** ✅ done · 🔶 intentional subset / documented limits

All prioritized depth items from the graphify audit waves are closed for 0.3.5.
Remaining 🔶 rows are honest capability boundaries, not unfinished stubs.

| Area | Status |
|------|--------|
| Tools / agents / RAG / safety P0 honesty | ✅ |
| Multimodal OpenAI media + OCR | ✅ |
| Training HF load / distill / LoRA resume | ✅ |
| Redis workers + gateway Redis semantic cache | ✅ |
| Scale pack Redis + PgVectorStore wiring | ✅ |
| MCP HTTP + roots/sampling/progress | ✅ |
| Anthropic Messages tool_use/tool_result + images | ✅ |
| BLEU-4 / NLI / HF CE / model guardrails + auto-wire | ✅ |
| Docs Pages / live probes / benches / freeze / CodeQL | ✅ |
| Examples (HITL, memory, MCP, GraphRAG, LoRA, …) | ✅ |

### Documented subsets (not false advertising)
- MCP: tools/resources/prompts + client roots/sampling/progress — not full MCP ecosystem.
- Anthropic: chat/tools/images/tool_use — not extended thinking / prompt caching / documents.
- SQS worker: clear `ConfigurationError` until boto3 is an optional extra.
- Detection: vision-JSON with capable models, not YOLO.

Quality gates: **ruff · mypy --strict · pytest -m "not live" · mkdocs --strict**.

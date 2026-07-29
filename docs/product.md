# Product principles

aire aims to be the **agent-first** path from idea → evaluated system → deployable API, without forcing a cloud vendor.

## Principles

1. **One interface** — Prefer `AI.*` and public exports from `aire` over deep imports.
2. **Works offline** — Core demos, tests, and `aire doctor` succeed with `mock:echo` / `local:hashing`.
3. **Discoverable** — Capabilities and honesty notes live in `.describe()` and docs, not marketing copy alone.
4. **Budgeted & auditable** — Agents have step/token/cost limits; steps and sessions are inspectable.
5. **Safety by default** — Tool side-effects are classified; policies can deny/require approval; sandbox file tools.
6. **Lazy extras** — Optional stacks (serve, torch, neo4j, …) stay out of the critical import path.

## Non-goals

- **Not a hosted platform** — aire is a library; hosting is your process / containers.
- **Not a full MCP host** — see [MCP](mcp.md); stdio subset only.
- **Not pretrained foundation weights** — `AI.training.foundation(...)` builds toy architectures.
- **Not a replacement for specialized ML frameworks** — `aire.ml` is complementary, not PyTorch/sklearn.
- **Not silent magic** — misnamed or stub metrics/backends are documented in [Honesty](honesty.md).

## Maturity

v0.3.5 is **Alpha**. Core RAG + agents + gateway + eval work offline; several platform extras remain experimental. Read [GAPS.md](https://github.com/desenyon/aire/blob/main/GAPS.md) for documented subsets.

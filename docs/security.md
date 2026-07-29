# Security

## Threat model (summary)

| Asset | Threat | Mitigation in aire |
|-------|--------|--------------------|
| Host filesystem | Path traversal via tools | `read_file` / `list_files` confined to sandbox root; escapes raise `SafetyError` |
| Secrets / PII in prompts | Leakage | Regex guardrails + redaction helpers (`AI.safety`) — not a full DLP product |
| Tool side effects | Unwanted network / destructive actions | `SideEffect` classification; approval levels; `PolicyEngine` deny / require_approval |
| Gateway abuse | Unauthenticated / high-volume use | Optional bearer token + rate limits on gateway |
| Prompt injection | Tool misuse | Injection guardrail (regex); pair with approvals for external tools |
| Supply chain | Malicious plugins | Entry-point plugins; review before loading |

**Out of scope / limited:** model-based classifiers, full multi-tenant ACL identity binding, sandboxed arbitrary code execution (calculator is AST-only arithmetic).

## Policy defaults

`AI.safety.policy()` → deny `prohibited`, `require_approval` for `external_side_effect` and above.

## Further reading

Full security notes and reporting process: **[SECURITY.md](https://github.com/desenyon/aire/blob/main/SECURITY.md)** at the repository root.

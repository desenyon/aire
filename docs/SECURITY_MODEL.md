# aire — Security Model

## Threat model (initial)

Assets: API credentials, user data flowing through prompts/documents, the host
filesystem, and downstream systems reachable through tools.

Adversaries: untrusted prompt/document content (injection), compromised or
misconfigured providers, malicious plugin packages, and accidental developer
misconfiguration (over-broad permissions, missing budgets).

Out of scope: model weight integrity, hardware/OS isolation, and malicious
code *intentionally installed* with full environment trust.

## Controls

### 1. Side-effect risk classification

Every tool declares a `SideEffect` level:

```
read_only  →  reversible_write  →  external_side_effect  →  high_impact  →  prohibited
```

`prohibited` tools never execute. `AgentConfig.approval_levels` plus an
`Approver` callback gate higher-risk actions behind explicit (human or policy)
approval. `ApprovalPolicy` codifies which levels require approval and supports
trusted permission grants.

### 2. Permission enforcement

Tools declare `permissions=["database.read", ...]`. The agent executor checks
the `ExecutionContext.permissions` set before every tool call; a missing
permission produces a `PERMISSION_DENIED` step and an error observation — never
a silent execution. Deny-by-default is the fallback approver.

### 3. Guardrails

`GuardrailChain` runs `PIIGuardrail`, `InjectionGuardrail` and
`SecretGuardrail` over text, with `block`/`warn`/`redact` actions. Patterns
live in `aire.safety.patterns` and are conservative (precision over recall).
`redact`, `redact_pii` and `redact_secrets` are available standalone and are
applied to trace attributes automatically (sensitive keys masked).

### 4. Sandboxed file access

Builtin file tools and dataset loaders enforce `sandbox_root`: resolved paths
must stay inside the root, blocking `../` traversal. The calculator tool uses
a restricted AST evaluator — no `eval`, no attribute access, no names beyond
an allow-list of math functions.

### 5. Unsafe deserialization prevention

All YAML goes through `yaml.safe_load` (python-object tags raise).
The library contains no `pickle` usage (enforced by a security test).
Structured outputs are validated with Pydantic models configured
`extra="forbid"`, so tool calls with unexpected arguments are rejected.

### 6. Secret handling

- Provider credentials come from `Settings.providers` (env-driven) and are
  excluded from `model_dump()` serialization (`SecretStr`).
- Tracing masks attributes whose keys match sensitive patterns.
- Budget limits (`max_tokens`, `max_cost_usd`, `max_steps`) bound blast radius
  of runaway agents; cancellation is cooperative via `ExecutionContext`.

### 7. Deployment surface

`create_app` adds optional bearer auth and per-client rate limiting; all
`AireError`s render as structured JSON with machine-readable codes, and
internal exception details are not leaked in responses beyond the error payload
itself. Generated artifacts ship a `.env.template` (placeholders only) so real
secrets never land in the image.

## Security testing

`tests/security` covers: prompt injection blocking, path traversal, unsafe
YAML tags, pickle absence, secret redaction, unknown-argument rejection,
permission bypass attempts, approval enforcement. Quality gates require this
suite to pass on every change.

## Reporting

Do not open public issues for vulnerabilities. Email the maintainers (see
`SECURITY.md` at repository root when published) with reproduction details;
expect acknowledgment within 72 hours.

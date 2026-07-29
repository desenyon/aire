# Security Policy

## Supported versions

aire is currently in **Alpha** (`0.x`). Security fixes are applied on a
best-effort basis to the latest release on the default branch.

| Version | Supported          |
| ------- | ------------------ |
| 0.3.x   | :white_check_mark: |
| < 0.3   | :x:                |

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

Prefer one of:

1. **GitHub private vulnerability reporting** — open a private security advisory
   on [desenyon/aire](https://github.com/desenyon/aire/security/advisories/new)
2. **Email** — `security@example.com` (placeholder; replace with a maintained
   contact when available)

Include:

- A description of the issue and its impact
- Steps to reproduce or a proof of concept
- Affected versions / commit SHAs if known
- Any suggested remediations

We will acknowledge reports when possible and coordinate disclosure after a fix
is available. Please give us reasonable time before public discussion.

## Automated scanning

Beyond Dependabot:

- **pip-audit** runs on every CI job (informational / best-effort).
- **CodeQL** analyzes Python on push/PR and weekly (``.github/workflows/codeql.yml``).

These do not replace application threat modeling for production deployments.

## Scope notes

- Provider API keys and credentials must never be committed; use env vars /
  `aire.yaml` / runtime options.
- Offline defaults (`mock:echo`, `local:hashing`) are intentional for local and
  CI use — they are not production security controls.
- Experimental surfaces (vision, audio, foundation model helpers) may return
  stub results; treat their outputs accordingly.

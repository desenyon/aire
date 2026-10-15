# 0.4.0 release QA record

Status: 0.4.0 implementation, independent review, QA and merge complete.
[PR #15](https://github.com/desenyon/aire/pull/15) merged as
`afb168c3cd5cb54606ff5206cc29aa75ab734270`, preserving all twelve dependency
PR heads. PRs #1–#12 are marked merged and audit tracker #14 is closed.
This historical implementation-merge audit did not publish a PyPI tag/upload.
The subsequent publication flow is documented in [publishing](../publishing.md);
consult [GitHub Releases](https://github.com/desenyon/aire/releases) and
[PyPI](https://pypi.org/project/aire-ai/) for the actual publication state. The [capability matrix](../capabilities.md)
separates executed backends from deterministic transports and intentional limits.

| Gate | Evidence so far |
| --- | --- |
| Core + optional offline suite | 308 passed, 11 live tests deselected; combined line/branch coverage 57.52% (not a product-completion percentage) |
| Minimum core | Final GitHub Python 3.11–3.13 lowest-direct binary-compatible matrix passes; local Python 3.13 run: 290 passed, one optional skip, using Pydantic 2.8.1, httpx 0.27.0, PyYAML 6.0.2, Typer 0.16.0, FastAPI 0.110.0 |
| Static checks | Ruff/formatting, strict mypy with and without optional packages, actionlint and lock validation pass |
| Vector services | All five real CRUD/filter/replacement/delete tests pass, including an authorized candidate below seven denied candidates |
| Durable services | Real Neo4j triples/neighborhood; Redis cancellation → retained claim → recover once → same job ID/result |
| ML | Six estimator fit/predict/save/load roundtrips; real local LoRA resume and teacher/student distillation |
| Media | Real scanned-PDF OCR and FFmpeg JPEG frame request; hosted model quality unverified |
| Aggregate extras | Built-wheel metadata expands all 20 functional profiles to exactly the same 25 requirements; published profile names preserved |
| Installed wheel | Isolated external-venv import/CLI/metadata/entrypoints/lazy imports passed on macOS Python 3.13 |
| Generated deployment | Hash-locked Linux arm64 container build passed; real RAG HTTP request/citations passed; Postgres outage yielded health 200/ready 503 and recovered to ready 200 |
| Docs | Strict build passes with mkdocstrings 1.0.7; 1.0.6 failed against pymdown-extensions 12.3, so the reviewed floor was raised |
| GitHub/Pages/release | Final branch 8743ed4: CI 38671665317, CodeQL 38671665298, release dry run 38671677456 and benchmark 38671680668 pass. Main merge CI 38671922138 passes. Pages 38671922127 deploys afb168c successfully with main-only protection intact |

QA uses only disposable local containers labeled `aire.qa=completion040`. No paid
provider request or GPU quantization run is implied by this record. CPU native
libraries run with bounded threads; unconstrained mixed-runtime performance is not
verified. Schema validation and MCP remain documented subsets. Remote replacement
outside PostgreSQL remains nontransactional; local files/checkpoints require one
writer and queue effects are at-least-once.

Final review of 70b4633..e3e96de independently ran 78 boundary tests and found
three Important defects. Sixteen focused regressions now pass after observed
failures: cancellation during attempts/backoff, expired/pre-cancelled contexts,
recursive array/union bounds, independent gateway configuration/namespace
isolation and invalid cached route rejection. The complete repaired suite passes
308 tests. See [review decisions](2026-10-14-final-review.md).

## Delivery evidence

- [Final branch CI](https://github.com/desenyon/aire/actions/runs/38671665317): six Linux/macOS/Python core jobs, three minimum jobs, three optional CPU profiles, documentation floor and real services/generated container all pass.
- [Merged-main CI](https://github.com/desenyon/aire/actions/runs/38671922138) rechecks the merge. The local merged tree also passes all 308 tests.
- [Release dry run](https://github.com/desenyon/aire/actions/runs/38671677456): source/artifact verification and uploaded/downloaded wheel verification succeed; publishing is skipped intentionally.
- [Pages deployment](https://github.com/desenyon/aire/actions/runs/38671922127): deployment 7082376175 reports success for afb168c. The public [capabilities](https://desenyon.github.io/aire/capabilities/), [gateway](https://desenyon.github.io/aire/gateway/) and [QA record](https://desenyon.github.io/aire/audits/2026-10-14-release-qa/) returned HTTP 200 and the expected release content.
- All 14 examples pass in temporary working directories. Runtime dependency audit reports no known vulnerabilities for the resolved core set.
- Disposable owned containers/network were removed and the session-created Colima profile stopped. Installed QA tools and its recoverable VM disk are retained.

## Dependency PR disposition

Each PR received an individual evidence comment and its exact head is retained
through the #15 merge commit; all are marked merged.

| PR | Verified outcome |
| --- | --- |
| [#1](https://github.com/desenyon/aire/pull/1) | deploy-pages v5; actual protected-main public deployment succeeds |
| [#2](https://github.com/desenyon/aire/pull/2) | upload-artifact v7/download-artifact v8 release roundtrip and benchmark upload succeed |
| [#3](https://github.com/desenyon/aire/pull/3) | CodeQL v4 scan succeeds |
| [#4](https://github.com/desenyon/aire/pull/4) | checkout v7 runs across the complete hosted matrix |
| [#5](https://github.com/desenyon/aire/pull/5) | setup-python v7 runs Python 3.11–3.13 on Linux/macOS |
| [#6](https://github.com/desenyon/aire/pull/6) | upload-pages-artifact v5 feeds the successful public deployment |
| [#7](https://github.com/desenyon/aire/pull/7) | pytesseract 0.3.13 actual scanned-PDF OCR passes |
| [#8](https://github.com/desenyon/aire/pull/8) | XGBoost 3.2.0 and locked current fit/predict/native save/load pass |
| [#9](https://github.com/desenyon/aire/pull/9) | Pillow 12.3.0 image/OCR fixtures pass |
| [#10](https://github.com/desenyon/aire/pull/10) | mkdocstrings 1.0.6 incompatibility reproduced; superseded by passing 1.0.7 floor |
| [#11](https://github.com/desenyon/aire/pull/11) | Keras 3.15.1 Torch CPU fit/predict/optimizer save/load pass |
| [#12](https://github.com/desenyon/aire/pull/12) | PEFT 0.20.0 tiny local LoRA fit/save/resume passes |

# 0.4.0 release QA record

Status: implementation and final-review repairs complete; exact-head GitHub
checks and main-only Pages/merge delivery are being completed. This record does not claim a
PyPI upload or final merge. The [capability matrix](../capabilities.md) separates
locally executed backends from deterministic transports and intentional limits.

| Gate | Evidence so far |
| --- | --- |
| Core + optional offline suite | 308 passed, 11 live tests deselected; combined line/branch coverage 57.52% (not a product-completion percentage) |
| Minimum core | GitHub Python 3.11–3.13 lowest-direct binary-compatible matrix passes at e3e96de; local Python 3.13 run used Pydantic 2.8.1, httpx 0.27.0, PyYAML 6.0.2, Typer 0.16.0, FastAPI 0.110.0 (250 passed before final additive regressions) |
| Static checks | Ruff/formatting, strict mypy with and without optional packages, actionlint and lock validation pass |
| Vector services | All five real CRUD/filter/replacement/delete tests pass, including an authorized candidate below seven denied candidates |
| Durable services | Real Neo4j triples/neighborhood; Redis cancellation → retained claim → recover once → same job ID/result |
| ML | Six estimator fit/predict/save/load roundtrips; real local LoRA resume and teacher/student distillation |
| Media | Real scanned-PDF OCR and FFmpeg JPEG frame request; hosted model quality unverified |
| Aggregate extras | Built-wheel metadata expands all 20 functional profiles to exactly the same 25 requirements; published profile names preserved |
| Installed wheel | Isolated external-venv import/CLI/metadata/entrypoints/lazy imports passed on macOS Python 3.13 |
| Generated deployment | Hash-locked Linux arm64 container build passed; real RAG HTTP request/citations passed; Postgres outage yielded health 200/ready 503 and recovered to ready 200 |
| Docs | Strict build passes with mkdocstrings 1.0.7; 1.0.6 failed against pymdown-extensions 12.3, so the reviewed floor was raised |
| GitHub/Pages/release | e3e96de complete CI, CodeQL and release dry run passed; 0a9c334 benchmark gate passed. Final-review fix 89e2dcc submitted for fresh CI. All 12 exact dependency heads are merge ancestors. Pages build/upload passed, branch deployment correctly blocked by main-only protection; public deployment will be verified after merge |

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

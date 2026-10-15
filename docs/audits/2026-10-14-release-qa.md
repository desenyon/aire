# 0.4.0 release QA record

Status: implementation and release QA in progress. This record does not claim a
PyPI upload or final merge. The [capability matrix](../capabilities.md) separates
locally executed backends from deterministic transports and intentional limits.

| Gate | Evidence so far |
| --- | --- |
| Core + optional offline suite | 267 passing before the final MCP cancellation additions; final full rerun pending |
| Minimum core, Python 3.11 | 233 passed, 1 optional skip; Pydantic 2.6.0, httpx 0.27.0, PyYAML 6.0, Typer 0.16.0 |
| Static checks | Ruff and strict mypy pass; final post-workflow rerun pending |
| Vector services | All five real CRUD/filter/replacement/delete tests pass, including an authorized candidate below seven denied candidates |
| Durable services | Real Neo4j triples/neighborhood; Redis cancellation → retained claim → recover once → same job ID/result |
| ML | Six estimator fit/predict/save/load roundtrips; real local LoRA resume and teacher/student distillation |
| Media | Real scanned-PDF OCR and FFmpeg JPEG frame request; hosted model quality unverified |
| Installed wheel | Isolated external-venv import/CLI/metadata/entrypoints/lazy imports passed on macOS Python 3.13 |
| Generated deployment | Hash-locked Linux arm64 container build passed; real RAG HTTP request/citations passed; Postgres outage yielded health 200/ready 503 and recovered to ready 200 |
| Docs | Strict build passes with mkdocstrings 1.0.7; 1.0.6 failed against pymdown-extensions 12.3, so the reviewed floor was raised |
| GitHub/Pages/release | All 12 dependency branches integrated for combined QA; GitHub matrix, release dry run, final branch review and merge pending |

QA uses only disposable local containers labeled `aire.qa=completion040`. No paid
provider request or GPU quantization run is implied by this record. CPU native
libraries run with bounded threads; unconstrained mixed-runtime performance is not
verified. Schema validation and MCP remain documented subsets. Remote replacement
outside PostgreSQL remains nontransactional; local files/checkpoints require one
writer and queue effects are at-least-once.

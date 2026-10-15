# 0.4.0 publication repair

The implementation merge and successful dry runs did not publish 0.4.0. At the
start of this repair, PyPI returned 0.3.5 as latest and 404 for 0.4.0; GitHub had
only the historical v0.3.5 release, without package assets. That release's install
command incorrectly named the unrelated `aire` package. The July failed tag job
also showed a 403 uploading that wrong distribution; a subsequent July run had
successfully published `aire-ai` 0.3.5.

The current release workflow was missing GitHub Release creation. Repository
metadata also retained README 0.3.x, citation 0.3.5 and an outdated citation date,
an unreleased migration heading and a 0.3.5 bug-report example. These are corrected
before tagging. The old GitHub release's install command and repository homepage
were corrected without moving old tags or replacing published package files.

## Required publication contract

- Version, citation/date, changelog, README series, docs and issue-template example
  agree. Published distribution metadata is `aire-ai` at the same version.
- The matching tag points to current clean main. CI, CodeQL, Docs and Benchmarks
  must have successful main-push runs for that exact commit; check again just
  before uploading. Environment `release` remains restricted to `v*` tags.
- Build tools and Hatchling come from the lock without build isolation, with
  the commit timestamp as SOURCE_DATE_EPOCH.
  Two consecutive local builds produced byte-identical wheel and sdist.
- A bundle contains exactly the wheel/sdist, source SHA, SHA-256 checksums and
  integrity-checked changelog-derived notes. A separate job verifies the
  downloaded bundle and installs the wheel outside the checkout.
- Existing PyPI files must match these hashes. Matching files permit resumable
  publication; a conflicting file fails. Verify both published downloads and a
  fresh published-wheel installation before creating the GitHub Release.
- GitHub assets include the same wheel/sdist, checksums and source provenance.
  Existing drafts can resume; public assets must match on a repeated run, and verified notes/title are
  reapplied so stale wrong-package instructions cannot survive a green rerun.
- Dry runs explicitly skip publication. Main is merged and all checks plus its
  dry run pass before the version tag is pushed. The actual publishing job,
  public indexes, release assets and a fresh index install establish completion.

## Verification before release

The full local suite passes **331 tests**, with 11 live tests deselected; the 23
new release regressions cover stale source/tag/checks, distribution metadata,
artifact/notes integrity, partial/matching/conflicting published files and
published-download host/hash validation. `make all` passes in a separate locked
base environment, including strict docs and release metadata checks. Ruff,
formatting, strict types, lock validation, actionlint, twine and isolated installed
wheel QA pass. Existing optional/service coverage is rerun by the full GitHub
matrix before merge and on main. Publication evidence is recorded by the tag
workflow, GitHub release assets and PyPI rather than predicted in this document.

## Independent review disposition

A fresh reviewer independently ran the initial 21 release tests and reproduced
two Important defects. Both received failing regressions before repair: isolated
builds resolved an unlocked backend, and an existing public GitHub Release could
keep incorrect notes despite matching assets. Hatchling 1.32.4 and dependencies
are now locked in the release group; canonical builds use --no-isolation. The
exact GitHub shell step now refreshes verified title/notes after public asset
comparison. There were no Critical or deferred Minor findings.

The review's external/unsupported cases were evaluated as follows:

| Case set aside by reviewer | Disposition / cost if unmet |
| --- | --- |
| Live OIDC, PyPI and GitHub publication | Mandatory actual tag/public-index/install verification after all preflight gates; no completion claim from dry runs |
| Environment/ruleset configuration | Verified release environment permits v* tags; preserve it and existing publisher identity; configuration drift can block publication |
| Forced tag movement/deletion during a run | Unsupported and prohibited; operator mutation can invalidate or stop a release |
| Resuming after main advances | Intentionally rejected to honor current-main requirement; a later change requires fresh QA/version decisions |
| Public releases with missing/conflicting assets | Fail without replacing public assets; maintainer repair may be needed |
| Unchanged runtime/optional/service behavior | Full local and hosted matrix verifies existing contracts; GPU/provider limits remain explicit |
| Universal long-term/cross-platform reproducibility | Evidence is scoped to the locked release profile and repeated builds; other toolchains/platforms need separate byte checks |

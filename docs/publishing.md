# Publishing Aire

The distribution is **`aire-ai`**; the Python import and CLI are `aire`.
The unrelated PyPI project `aire` must never appear in an install command for this
repository. [PyPI](https://pypi.org/project/aire-ai/) and
[GitHub Releases](https://github.com/desenyon/aire/releases) are separate outputs.
A successful dry run builds artifacts but publishes neither output.

## Before tagging

1. Update `src/aire/_version.py`, the dated `CHANGELOG.md` entry, `CITATION.cff`
   version/date, README release series, docs home and bug-report version example.
2. Update affected implementation, examples, capability evidence and migration
   notes; retain explicitly historical audit results and unsupported boundaries.
3. Run `uv sync --locked --no-default-groups --group dev`, `make audit`, and the
   relevant optional/service checks. Build and test installed distributions.
4. Merge the reviewed PR. Wait for **CI, CodeQL, Docs and Benchmarks** to succeed
   on that exact current `main` commit, including the public Pages deployment.
5. Run a dry run from that commit:

   ```bash
   gh workflow run release.yml --ref main -f dry_run=true
   ```

   Inspect both source/artifact and downloaded-artifact verification jobs. A green
   dry run with a skipped publishing job is not a published release.

The preflight checks current metadata and rejects stale/mismatched tags, a dirty
checkout, a tag behind current main, or missing exact-commit main checks. The
version remains centralized in `_version.py`; dynamic project metadata and the
editable uv lock do not duplicate it.

## Publish the tested commit

After every check above succeeds and local main matches the tested remote SHA:

```bash
git switch main
git pull --ff-only
# Confirm this exact commit passed all four required main workflows.
git status --short
.venv/bin/python scripts/release.py check
git tag -a v0.4.2 -m "Aire 0.4.2"
git push origin refs/tags/v0.4.2
```

Push only the intended tag. Never move a tag or replace an already published
version. A further source change requires a new version once publication occurs.

The tag workflow:

1. Verifies the current tested main/tag/version and repository metadata.
2. Uses the locked tools/backend without build isolation and the commit timestamp to build reproducible wheel/sdist.
3. Checks metadata, typing/plugin/CLI/lazy-import behavior outside the checkout.
4. Attaches source SHA, SHA-256 checksums and changelog-derived release notes.
5. Downloads the bundle in a separate job and verifies the same bytes again.
6. Rejects an existing PyPI file whose hash differs; uploads missing matching
   files from an isolated upload copy through Trusted Publishing; verifies the published bytes and installs
   the actual published wheel in a fresh environment.
7. Creates the GitHub Release with wheel, sdist, checksums, source provenance and
   correct `aire-ai` install instructions. Existing draft releases can resume;
   an existing public release's assets must match on rerun, then its title/notes are refreshed
   from the verified bundle.

If a tag workflow is interrupted, rerun its failed jobs. To dispatch publication
manually, select the **matching tag**, not main:

```bash
gh workflow run release.yml --ref v0.4.2 -f dry_run=false
```

Matching published files are reused; conflicting files fail instead of being
silently skipped. Publication remains restricted to `v*` tags by the `release`
environment. No API-token secret is used.

## Trusted Publisher configuration

The existing `aire-ai` project already uses PyPI Trusted Publishing. Its GitHub
publisher must continue to match owner `desenyon`, repository `aire`, workflow
`release.yml`, and environment `release`. For a new project, configure a pending
publisher first; for this project, manage the existing publisher. Do not rename
these identifiers without updating PyPI. The GitHub environment permits tags
matching `v*`; preserve that protection.

## Verify the published release

Inspect the publishing and verification jobs, the GitHub release assets and the
exact PyPI version. A missing GitHub release or a skipped publisher is incomplete.

```bash
uv venv /tmp/aire-release-check
uv pip install --python /tmp/aire-release-check/bin/python 'aire-ai==0.4.2'
/tmp/aire-release-check/bin/python -I -c \
  "from aire import AI, __version__; print(__version__, AI.describe()['library'])"
/tmp/aire-release-check/bin/aire doctor --smoke
```

For optional backends, use the selected documented extras and supported hardware;
base-install success does not establish GPU or hosted-provider quality.

## Publisher-generated attestations

Trusted Publishing generates `.publish.attestation` sidecars beside its upload
inputs. The workflow copies the two verified distributions to `release-bundle/upload`
for that action. Its sidecars never enter the immutable `dist` payload or GitHub
asset list; attestations remain enabled and payload/hash checks remain strict.

The first 0.4.0 attempt uploaded successfully but then rejected those unexpected
sidecars in its original payload folder. Rerunning the failed job completed 0.4.0
by reusing the matching published files. Version 0.4.1 isolates upload inputs so
initial publication succeeds without needing that recovery path.

# Publishing to PyPI

The distribution name on PyPI is **`aire-ai`** (the import package remains `aire`).
The name `aire` is already taken by an unrelated project.

Validated, matching version tags on main publish via **Trusted Publishing** (OIDC) — no PyPI API token in GitHub secrets.

## One-time setup (required)

### 1. Create a PyPI account

Sign up at [https://pypi.org/account/register/](https://pypi.org/account/register/)
and enable 2FA.

### 2. Add a pending Trusted Publisher

Until the first successful upload, the project does not exist on PyPI. Create a
**pending** publisher:

1. Open [https://pypi.org/manage/account/publishing/](https://pypi.org/manage/account/publishing/)
2. Under **GitHub**, fill in:

   | Field | Value |
   |-------|-------|
   | PyPI Project Name | `aire-ai` |
   | Owner | `desenyon` |
   | Repository name | `aire` |
   | Workflow name | `release.yml` |
   | Environment name | `release` |

3. Click **Add**

### 3. Create the GitHub `release` environment

1. Open [https://github.com/desenyon/aire/settings/environments](https://github.com/desenyon/aire/settings/environments)
2. Create environment named exactly **`release`**
3. (Recommended) Restrict it to tags matching `v*` and require a reviewer

No secrets are needed for Trusted Publishing.

## How publishing works

`.github/workflows/release.yml` runs on:

- pushes of tags matching `v*` (e.g. `v0.4.0`)
- manual `workflow_dispatch` (dry-run build by default)

Flow: build sdist/wheel → `twine check` → OIDC publish with
`pypa/gh-action-pypi-publish`.

## Publish a new version

```bash
# bump src/aire/_version.py and CHANGELOG.md; metadata reads the version dynamically
git commit -am "release 0.4.0"
git tag -a v0.4.0 -m "aire 0.4.0"
git push origin main --tags
```

Or re-run a failed tag build from the Actions UI after completing the pending
publisher setup above.

## Verify

```bash
pip install aire-ai==0.4.0
python -c "from aire import AI, __version__; print(__version__, AI.describe()['library'])"
```

Project page (after first upload): https://pypi.org/project/aire-ai/

## Reproducible development and release QA

Use `uv sync --locked --group dev` for the tested development tools. Core checks do
not install heavyweight optional backends. Use selected extras (for example
`uv sync --locked --extra ml --group test`) for backend contracts. The `all` extra
aggregates the public profiles without copying version constraints and is expensive;
backend runtime/hardware compatibility still applies.

Build with `uv run --locked --group release python -m build`, check distributions
with `twine check dist/*`, then run `python scripts/verify_distribution.py <wheel>`.
That script installs outside the source checkout and checks metadata, CLI, plugin
entrypoints, typing marker and lazy imports. Release dry runs must pass before a
version tag is used for publication. A merged version bump is not proof of a PyPI
upload; inspect the publishing job and install the exact published version.

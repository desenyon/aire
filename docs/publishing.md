# Publishing to PyPI

The distribution name on PyPI is **`aire-ai`** (the import package remains `aire`).
The name `aire` is already taken by an unrelated project.

Releases publish via **Trusted Publishing** (OIDC) — no PyPI API token in GitHub secrets.

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

- pushes of tags matching `v*` (e.g. `v0.3.5`)
- manual `workflow_dispatch` (optional dry-run build)

Flow: build sdist/wheel → `twine check` → OIDC publish with
`pypa/gh-action-pypi-publish`.

## Publish a new version

```bash
# bump version in pyproject.toml + src/aire/_version.py + CHANGELOG.md
git commit -am "release 0.3.6"
git tag -a v0.3.6 -m "aire 0.3.6"
git push origin main --tags
```

Or re-run a failed tag build from the Actions UI after completing the pending
publisher setup above.

## Verify

```bash
pip install aire-ai==0.3.5
python -c "from aire import AI, __version__; print(__version__, AI.describe()['kind'])"
```

Project page (after first upload): https://pypi.org/project/aire-ai/

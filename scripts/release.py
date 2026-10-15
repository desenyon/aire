"""Check repository release state and promote the same verified distributions."""

# ruff: noqa: S603, S607
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import runpy
import subprocess
import sys
import tarfile
import tempfile
import time
import tomllib
import zipfile
from email import message_from_bytes
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import urlopen

import yaml

REQUIRED_CHECKS = ("ci.yml", "codeql.yml", "docs.yml", "benchmarks.yml")
SOURCE_FILES = (
    "src/aire/_version.py",
    "pyproject.toml",
    "uv.lock",
    "CITATION.cff",
    "README.md",
    "CHANGELOG.md",
    "docs/index.md",
    ".github/ISSUE_TEMPLATE/bug_report.yml",
)


def source_version(root: Path) -> str:
    version = str(runpy.run_path(str(root / "src/aire/_version.py"))["__version__"])
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError("release version must be X.Y.Z")
    project = tomllib.loads((root / "pyproject.toml").read_text())
    if (
        project["project"]["name"] != "aire-ai"
        or project["tool"]["hatch"]["version"]["path"] != "src/aire/_version.py"
    ):
        raise ValueError("distribution name or dynamic version source is inconsistent")
    lock = tomllib.loads((root / "uv.lock").read_text())
    if not any(
        item["name"] == "aire-ai"
        and item.get("version", version) == version
        and item["source"].get("editable") == "."
        for item in lock["package"]
    ):
        raise ValueError("uv.lock project version is stale")
    citation = yaml.safe_load((root / "CITATION.cff").read_text())
    changelog = (root / "CHANGELOG.md").read_text()
    if (
        citation["version"] != version
        or f"## {version} - {citation['date-released']}" not in changelog
    ):
        raise ValueError("CITATION version/date and CHANGELOG must agree with the release")
    if f"**Alpha, {version.rsplit('.', 1)[0]}.x.**" not in (root / "README.md").read_text():
        raise ValueError("README release series is stale")
    if f"(v{version})" not in (root / "docs/index.md").read_text():
        raise ValueError("documentation version is stale")
    issue = yaml.safe_load((root / ".github/ISSUE_TEMPLATE/bug_report.yml").read_text())
    if not any(item.get("attributes", {}).get("placeholder") == version for item in issue["body"]):
        raise ValueError("issue-template version is stale")
    return version


def check_publication(
    version: str,
    ref: str,
    head: str,
    main: str,
    clean: bool,
    runs: dict[str, list[dict[str, Any]]],
) -> None:
    if ref != "refs/tags/v" + version:
        raise ValueError(
            "publication requires the matching version tag; dispatch on the tag, not main"
        )
    if head != main or not clean:
        raise ValueError("publication requires a clean checkout of the current main commit")
    for workflow in REQUIRED_CHECKS:
        if not any(
            run["head_sha"] == head
            and run["head_branch"] == "main"
            and run["event"] == "push"
            and run["conclusion"] == "success"
            for run in runs.get(workflow, [])
        ):
            raise ValueError(f"{workflow} must pass on this exact main commit before publication")


def git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], text=True).strip()


def release_notes(root: Path, version: str) -> str:
    changelog = (root / "CHANGELOG.md").read_text()
    section = (
        changelog.split(f"## {version} - ", 1)[1].split("\n", 1)[1].split("\n## ", 1)[0].strip()
    )
    if not section:
        raise ValueError("release changelog section is empty")
    return (
        section + f"\n\n## Install\n\n```bash\npython -m pip install aire-ai=={version}\n```\n"
        "\nThe distribution is `aire-ai`; Python imports remain `aire`. "
        "The unrelated `aire` distribution is not this project.\n\n"
        "[Documentation](https://desenyon.github.io/aire/) · "
        "[Capability contracts](https://desenyon.github.io/aire/capabilities/)\n\n"
        "Wheel/sdist hashes and source commit are attached in SHA256SUMS and provenance.json.\n"
    )


def distribution_hashes(directory: Path, version: str) -> dict[str, str]:
    expected = {f"aire_ai-{version}-py3-none-any.whl", f"aire_ai-{version}.tar.gz"}
    if {path.name for path in directory.iterdir()} != expected:
        raise ValueError("unexpected or missing release distribution")
    hashes = {}
    for name in sorted(expected):
        path = directory / name
        if name.endswith(".whl"):
            with zipfile.ZipFile(path) as archive:
                raw = archive.read(f"aire_ai-{version}.dist-info/METADATA")
        else:
            with tarfile.open(path) as archive:
                stream = archive.extractfile(f"aire_ai-{version}/PKG-INFO")
                if stream is None:
                    raise ValueError("missing sdist metadata")
                raw = stream.read()
        metadata = message_from_bytes(raw)
        if metadata["Name"] != "aire-ai" or metadata["Version"] != version:
            raise ValueError("distribution metadata name/version differs from source")
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def prepare_bundle(directory: Path, version: str, commit: str, notes: str) -> None:
    hashes = distribution_hashes(directory / "dist", version)
    metadata = directory / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    (metadata / "provenance.json").write_text(
        json.dumps(
            {
                "version": version,
                "commit": commit,
                "files": hashes,
                "notes_sha256": hashlib.sha256(notes.encode()).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )
    (metadata / "SHA256SUMS").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in hashes.items())
    )
    (metadata / "release-notes.md").write_text(notes)


def verify_bundle(directory: Path, version: str, commit: str) -> dict[str, Any]:
    manifest = json.loads((directory / "metadata/provenance.json").read_text())
    if manifest["version"] != version or manifest["commit"] != commit:
        raise ValueError("release provenance differs from checked-out version/commit")
    actual = distribution_hashes(directory / "dist", version)
    if manifest["files"] != actual:
        raise ValueError("release artifact hash mismatch")
    checksums = "".join(f"{digest}  {name}\n" for name, digest in actual.items())
    if (directory / "metadata/SHA256SUMS").read_text() != checksums:
        raise ValueError("release checksum manifest mismatch")
    if (
        hashlib.sha256((directory / "metadata/release-notes.md").read_bytes()).hexdigest()
        != manifest["notes_sha256"]
    ):
        raise ValueError("release notes hash mismatch")
    return manifest


def published_complete(manifest: dict[str, Any], files: list[dict[str, Any]]) -> bool:
    found = {}
    for item in files:
        name, digest = item["filename"], item["digests"]["sha256"]
        if name not in manifest["files"] or digest != manifest["files"][name]:
            raise ValueError(f"published artifact differs from this release: {name}")
        found[name] = digest
    return found == manifest["files"]


def pypi_files(version: str) -> list[dict[str, Any]]:
    try:
        with urlopen(f"https://pypi.org/pypi/aire-ai/{version}/json", timeout=30) as response:
            return list(json.load(response)["urls"])
    except HTTPError as exc:
        if exc.code == 404:
            return []
        raise


def verify_published(manifest: dict[str, Any], install: bool) -> bool:
    deadline = time.monotonic() + (120 if install else 0)
    while True:
        files = pypi_files(manifest["version"])
        complete = published_complete(manifest, files)
        if complete or not install:
            break
        if time.monotonic() >= deadline:
            raise ValueError("published release files did not appear on PyPI")
        time.sleep(3)
    if install:
        # Verify downloaded bytes, then install the published wheel outside the checkout.
        with tempfile.TemporaryDirectory(prefix="aire-published-") as temporary:
            wheel = None
            for item in files:
                url = urlparse(item["url"])
                if url.scheme != "https" or url.hostname != "files.pythonhosted.org":
                    raise ValueError("unexpected published artifact host")
                with urlopen(item["url"], timeout=30) as response:  # noqa: S310
                    data = response.read()
                if hashlib.sha256(data).hexdigest() != manifest["files"][item["filename"]]:
                    raise ValueError("downloaded published artifact hash mismatch")
                destination = Path(temporary) / item["filename"]
                destination.write_bytes(data)
                if destination.suffix == ".whl":
                    wheel = destination
            if wheel is None:
                raise ValueError("published release is missing its wheel")
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("verify_distribution.py")),
                    str(wheel),
                ],
                check=True,
            )
        print("Published sdist/wheel bytes and fresh wheel installation verified")
    return complete


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["check", "prepare", "verify", "pypi"])
    parser.add_argument("directory", nargs="?", type=Path, default=Path("release-bundle"))
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    version, commit = source_version(root), git("rev-parse", "HEAD")
    if args.command == "check":
        if os.environ.get("PUBLISH") == "true":
            git("fetch", "origin", "main")
            runs = {}
            repository = os.environ["GITHUB_REPOSITORY"]
            for workflow in REQUIRED_CHECKS:
                raw = subprocess.check_output(
                    [
                        "gh",
                        "api",
                        f"repos/{repository}/actions/workflows/{workflow}/runs?head_sha={commit}&status=success&per_page=100",
                    ],
                    text=True,
                )
                runs[workflow] = json.loads(raw)["workflow_runs"]
            check_publication(
                version,
                os.environ["GITHUB_REF"],
                commit,
                git("rev-parse", "origin/main"),
                not git("status", "--porcelain"),
                runs,
            )
        print("Repository release metadata verified", version, commit)
    elif args.command == "prepare":
        prepare_bundle(args.directory, version, commit, release_notes(root, version))
    else:
        manifest = verify_bundle(args.directory, version, commit)
        if args.command == "pypi":
            complete = verify_published(manifest, args.install)
            if output := os.environ.get("GITHUB_OUTPUT"):
                with Path(output).open("a") as stream:
                    stream.write(f"complete={str(complete).lower()}\n")
            print("PyPI files already complete:", complete)
        else:
            print("Release artifact hashes/provenance verified", version, commit)


if __name__ == "__main__":
    main()

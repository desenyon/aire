import importlib.util
import io
import json
import os
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


def release_module():
    spec = importlib.util.spec_from_file_location("aire_release", ROOT / "scripts/release.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bundle(tmp_path, *, version="0.4.0", name="aire-ai"):
    directory = tmp_path / "bundle"
    dist = directory / "dist"
    dist.mkdir(parents=True)
    metadata = f"Name: {name}\nVersion: {version}\n"
    with zipfile.ZipFile(dist / "aire_ai-0.4.0-py3-none-any.whl", "w") as archive:
        archive.writestr("aire_ai-0.4.0.dist-info/METADATA", metadata)
    with tarfile.open(dist / "aire_ai-0.4.0.tar.gz", "w:gz") as archive:
        data = metadata.encode()
        item = tarfile.TarInfo("aire_ai-0.4.0/PKG-INFO")
        item.size = len(data)
        archive.addfile(item, io.BytesIO(data))
    return directory


def test_current_repository_release_metadata_agrees():
    from aire import __version__

    assert yaml.safe_load((ROOT / "CITATION.cff").read_text())["version"] == __version__
    assert f"**Alpha, {__version__.rsplit('.', 1)[0]}.x.**" in (ROOT / "README.md").read_text()


def test_release_workflow_finishes_publication_not_just_upload():
    workflow = (ROOT / ".github/workflows/release.yml").read_text()
    assert "gh release create" in workflow
    assert "scripts/release.py pypi" in workflow
    assert "scripts/release.py check" in workflow
    assert "packages-dir: release-bundle/dist" in workflow


@pytest.mark.parametrize("bad", ["tag", "main", "dirty", "checks"])
def test_publication_rejects_inconsistent_source(bad):
    release = release_module()
    options = dict(version="0.4.0", ref="refs/tags/v0.4.0", head="abc", main="abc", clean=True)
    runs = {
        name: [{"head_sha": "abc", "head_branch": "main", "event": "push", "conclusion": "success"}]
        for name in release.REQUIRED_CHECKS
    }
    if bad == "tag":
        options["ref"] = "refs/heads/main"
    elif bad == "main":
        options["main"] = "newer"
    elif bad == "dirty":
        options["clean"] = False
    else:
        runs["docs.yml"][0]["event"] = "workflow_dispatch"
    with pytest.raises(ValueError):
        release.check_publication(**options, runs=runs)


def test_publication_accepts_matching_tag_at_fully_tested_main():
    release = release_module()
    runs = {
        name: [{"head_sha": "abc", "head_branch": "main", "event": "push", "conclusion": "success"}]
        for name in release.REQUIRED_CHECKS
    }
    release.check_publication("0.4.0", "refs/tags/v0.4.0", "abc", "abc", True, runs)


@pytest.mark.parametrize("bad", ["version", "name", "extra_file"])
def test_bundle_rejects_inconsistent_or_unexpected_distribution(tmp_path, bad):
    release = release_module()
    directory = bundle(
        tmp_path,
        version="0.3.5" if bad == "version" else "0.4.0",
        name="aire" if bad == "name" else "aire-ai",
    )
    if bad == "extra_file":
        (directory / "dist/unexpected.whl").write_bytes(b"wrong distribution")
    with pytest.raises(ValueError):
        release.prepare_bundle(directory, "0.4.0", "abc", "release notes")


def test_bundle_detects_modified_artifact_and_provenance(tmp_path):
    release = release_module()
    directory = bundle(tmp_path)
    release.prepare_bundle(directory, "0.4.0", "abc", "release notes")
    manifest = release.verify_bundle(directory, "0.4.0", "abc")
    assert len(manifest["files"]) == 2
    with pytest.raises(ValueError, match="provenance"):
        release.verify_bundle(directory, "0.4.0", "different")
    wheel = next((directory / "dist").glob("*.whl"))
    wheel.write_bytes(wheel.read_bytes() + b"modified")
    with pytest.raises(ValueError, match="hash"):
        release.verify_bundle(directory, "0.4.0", "abc")


@pytest.mark.parametrize("state", ["absent", "partial", "complete", "different", "unexpected"])
def test_pypi_rerun_requires_matching_published_hashes(tmp_path, state):
    release = release_module()
    directory = bundle(tmp_path)
    release.prepare_bundle(directory, "0.4.0", "abc", "release notes")
    manifest = json.loads((directory / "metadata/provenance.json").read_text())
    files = [
        {"filename": name, "digests": {"sha256": digest}}
        for name, digest in manifest["files"].items()
    ]
    if state == "absent":
        files = []
    elif state == "partial":
        files = files[:1]
    elif state == "different":
        files[0]["digests"]["sha256"] = "different"
    elif state == "unexpected":
        files.append({"filename": "unexpected.whl", "digests": {"sha256": "other"}})
    if state in {"different", "unexpected"}:
        with pytest.raises(ValueError, match="published"):
            release.published_complete(manifest, files)
    else:
        assert release.published_complete(manifest, files) == (state == "complete")


def test_source_check_rejects_stale_metadata(tmp_path):
    import shutil

    release = release_module()
    for filename in release.SOURCE_FILES:
        target = tmp_path / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / filename, target)
    release.source_version(tmp_path)
    citation = tmp_path / "CITATION.cff"
    citation.write_text(citation.read_text().replace("version: 0.4.0", "version: 0.3.5"))
    with pytest.raises(ValueError, match="CITATION"):
        release.source_version(tmp_path)


def test_bundle_rejects_modified_release_notes(tmp_path):
    release = release_module()
    directory = bundle(tmp_path)
    release.prepare_bundle(directory, "0.4.0", "abc", "correct install notes")
    (directory / "metadata/release-notes.md").write_text("pip install aire")
    with pytest.raises(ValueError, match="notes"):
        release.verify_bundle(directory, "0.4.0", "abc")


@pytest.mark.parametrize("state", ["valid", "bad_host", "bad_bytes"])
def test_published_verification_checks_downloads_before_install(tmp_path, monkeypatch, state):
    release = release_module()
    directory = bundle(tmp_path)
    release.prepare_bundle(directory, "0.4.0", "abc", "release notes")
    manifest = release.verify_bundle(directory, "0.4.0", "abc")
    files = [
        {
            "filename": name,
            "digests": {"sha256": digest},
            "url": f"https://files.pythonhosted.org/{name}",
        }
        for name, digest in manifest["files"].items()
    ]
    if state == "bad_host":
        files[0]["url"] = "https://untrusted.example/package.whl"
    monkeypatch.setattr(release, "pypi_files", lambda version: files)
    monkeypatch.setattr(
        release,
        "urlopen",
        lambda url, **kw: io.BytesIO(
            b"corrupt"
            if state == "bad_bytes"
            else (directory / "dist" / url.rsplit("/", 1)[1]).read_bytes()
        ),
    )
    installs = []

    def install(command, **kwargs):
        wheel = Path(command[-1])
        assert not wheel.is_relative_to(ROOT)
        assert wheel.read_bytes() == (directory / "dist" / wheel.name).read_bytes()
        installs.append(wheel.name)

    monkeypatch.setattr(release.subprocess, "run", install)
    if state == "valid":
        assert release.verify_published(manifest, True)
        assert installs == ["aire_ai-0.4.0-py3-none-any.whl"]
    else:
        with pytest.raises(ValueError):
            release.verify_published(manifest, True)
        assert installs == []


def test_release_builds_use_locked_backend():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert any(dep.startswith("hatchling") for dep in project["dependency-groups"]["release"])
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    assert any(item["name"] == "hatchling" for item in lock["package"])
    for workflow in ("release.yml", "ci.yml"):
        for line in (ROOT / ".github/workflows" / workflow).read_text().splitlines():
            if "python -m build" in line:
                assert "--no-isolation" in line


def test_existing_public_release_corrects_metadata_after_matching_assets(tmp_path):
    release = release_module()
    directory = bundle(tmp_path)
    release.prepare_bundle(directory, "0.4.0", "abc", "python -m pip install aire-ai==0.4.0")
    directory = directory.rename(tmp_path / "release-bundle")
    workflow = yaml.safe_load((ROOT / ".github/workflows/release.yml").read_text())
    step = next(
        step
        for step in workflow["jobs"]["publish"]["steps"]
        if step.get("name") == "Create GitHub release with verified assets"
    )
    binaries = tmp_path / "bin"
    binaries.mkdir()
    gh = binaries / "gh"
    gh.write_text(
        f"#!{sys.executable}\n"
        + """import json, os, pathlib, shutil, sys
args = sys.argv[1:]
with open(os.environ["AIRE_TEST_CALLS"], "a") as stream:
    stream.write(json.dumps(args) + "\\n")
if args[:2] == ["release", "view"]:
    print(json.dumps({"isDraft": False}))
elif args[:2] == ["release", "download"]:
    destination = pathlib.Path(args[args.index("--dir") + 1])
    destination.mkdir()
    bundle = pathlib.Path(os.environ["AIRE_TEST_BUNDLE"])
    sources = list((bundle / "dist").iterdir())
    sources += [bundle / "metadata/SHA256SUMS", bundle / "metadata/provenance.json"]
    for source in sources:
        shutil.copy(source, destination / source.name)
"""
    )
    gh.chmod(0o755)
    jq = binaries / "jq"
    jq.write_text("#!/bin/sh\nexit 1\n")  # Existing public release: .isDraft is false.
    jq.chmod(0o755)
    calls = tmp_path / "calls.jsonl"
    environment = dict(
        os.environ,
        PATH=f"{binaries}:{os.environ['PATH']}",
        RUNNER_TEMP=str(tmp_path),
        GITHUB_REF_NAME="v0.4.0",
        AIRE_TEST_CALLS=str(calls),
        AIRE_TEST_BUNDLE=str(directory),
    )
    subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", step["run"]],
        cwd=tmp_path,
        env=environment,
        check=True,
    )
    commands = [json.loads(line) for line in calls.read_text().splitlines()]
    edits = [command for command in commands if command[:2] == ["release", "edit"]]
    assert len(edits) == 1
    assert "--notes-file" in edits[0]
    assert "--title" in edits[0]
    assert "--latest" in edits[0]

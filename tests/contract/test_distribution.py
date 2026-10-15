"""Keep aggregate extras and capability documentation tied to published contracts."""

import tomllib
from pathlib import Path

from packaging.requirements import Requirement

from aire import AI

ROOT = Path(__file__).resolve().parents[2]


def test_all_extra_expands_every_functional_profile_once():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
    extras = metadata["project"]["optional-dependencies"]
    aggregate = Requirement(extras["all"][0])
    assert aggregate.name == metadata["project"]["name"]
    functional = {
        name
        for name, dependencies in extras.items()
        if dependencies and name not in {"all", "dev", "docs"}
    }
    assert aggregate.extras == functional
    assert len(extras["all"]) == 1
    assert metadata["project"]["dynamic"] == ["version"]


def test_every_manifest_namespace_has_capability_evidence():
    matrix = (ROOT / "docs/capabilities.md").read_text()
    for namespace in AI.describe()["namespaces"]:
        assert f"| `{namespace}` |" in matrix

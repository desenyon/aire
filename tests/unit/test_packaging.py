"""Deployment requirements must resolve to this distribution, not the unrelated aire project."""

from importlib.metadata import metadata

from packaging.requirements import Requirement

from aire.deployment.artifacts import generate_artifacts
from aire.deployment.scale import ScaleConfig, generate_scale_pack


def test_generated_deployment_installs_own_distribution(tmp_path):
    generate_artifacts(tmp_path, extra_requirements=["example-dependency==1.2.3"])
    lines = (tmp_path / "requirements.lock").read_text().splitlines()
    requirement = Requirement(lines[0])
    assert requirement.name == metadata("aire-ai")["Name"]
    assert requirement.extras == {"serve"}
    assert lines[1] == "example-dependency==1.2.3"


def test_scale_pack_sidecars_install_own_distribution(tmp_path):
    generate_scale_pack(tmp_path, config=ScaleConfig(with_redis=True, with_postgres=True))
    requirements = [
        Requirement(line) for line in (tmp_path / "requirements.lock").read_text().splitlines()
    ]
    assert {requirement.name for requirement in requirements} == {metadata("aire-ai")["Name"]}
    assert set().union(*(requirement.extras for requirement in requirements)) == {
        "serve",
        "redis",
        "pgvector",
    }

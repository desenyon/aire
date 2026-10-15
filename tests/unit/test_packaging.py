"""Deployment requirements must resolve to this distribution, not the unrelated aire project."""

from importlib.metadata import metadata

from packaging.requirements import Requirement

from aire.deployment.artifacts import generate_artifacts
from aire.deployment.scale import ScaleConfig, generate_scale_pack


def test_generated_deployment_installs_own_distribution(tmp_path):
    generate_artifacts(tmp_path, extra_requirements=["example-dependency==1.2.3"])
    lines = (tmp_path / "requirements.in").read_text().splitlines()
    requirement = Requirement(lines[0])
    assert requirement.name == metadata("aire-ai")["Name"]
    assert requirement.extras == {"serve"}
    assert lines[1] == "example-dependency==1.2.3"


def test_scale_pack_sidecars_install_own_distribution(tmp_path):
    generate_scale_pack(tmp_path, config=ScaleConfig(with_redis=True, with_postgres=True))
    requirements = [
        Requirement(line) for line in (tmp_path / "requirements.in").read_text().splitlines()
    ]
    assert {requirement.name for requirement in requirements} == {metadata("aire-ai")["Name"]}
    assert set().union(*(requirement.extras for requirement in requirements)) == {
        "serve",
        "redis",
        "pgvector",
    }


def test_deployment_requirements_pin_runtime_and_are_named_as_input(tmp_path):
    from aire import __version__

    generate_artifacts(tmp_path)
    requirement = Requirement((tmp_path / "requirements.in").read_text().splitlines()[0])
    assert str(requirement.specifier) == f"=={__version__}"
    assert not (tmp_path / "requirements.lock").exists()
    assert "requirements.in" in (tmp_path / "Dockerfile").read_text()


def test_regeneration_preserves_user_requirements_and_entrypoint(tmp_path):
    generate_artifacts(tmp_path)
    (tmp_path / "requirements.in").write_text(
        (tmp_path / "requirements.in").read_text() + "custom-package==1.2.3\n"
    )
    (tmp_path / "app.py").write_text("# user application\n")
    generate_artifacts(tmp_path)
    assert "custom-package==1.2.3" in (tmp_path / "requirements.in").read_text()
    assert (tmp_path / "app.py").read_text() == "# user application\n"


def test_generated_deployment_can_install_tested_local_wheel(tmp_path):
    import zipfile

    from aire import __version__

    wheel = tmp_path / f"aire_ai-{__version__}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            f"aire_ai-{__version__}.dist-info/METADATA", f"Name: aire-ai\nVersion: {__version__}\n"
        )
    output = tmp_path / "deploy"
    generate_artifacts(output, wheel=wheel)
    dockerfile = (output / "Dockerfile").read_text()
    assert "COPY wheels /app/wheels" in dockerfile
    assert "--find-links=/app/wheels" in dockerfile
    assert (output / "wheels" / wheel.name).is_file()

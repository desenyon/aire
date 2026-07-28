"""Project lock file: pin model/store refs for reproducible aire projects."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from aire.core.errors import ConfigurationError
from aire.core.serialization import read_json_file, write_json_file


class LockEntry(BaseModel):
    kind: str  # model | embedder | vector_store | graph_store | tool
    ref: str
    version: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class ProjectLock(BaseModel):
    """``aire.lock`` — pinned refs for a project."""

    version: int = 1
    project: str = ""
    aire_version: str = ""
    entries: list[LockEntry] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def refs(self, kind: str | None = None) -> list[str]:
        return [e.ref for e in self.entries if kind is None or e.kind == kind]

    def get(self, kind: str, default: str | None = None) -> str | None:
        for e in self.entries:
            if e.kind == kind:
                return e.ref
        return default

    def set_ref(self, kind: str, ref: str, **options: Any) -> None:
        self.entries = [e for e in self.entries if e.kind != kind]
        self.entries.append(LockEntry(kind=kind, ref=ref, options=options))

    def describe(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def lock_path(directory: str | Path = ".") -> Path:
    return Path(directory) / "aire.lock"


def load_lock(path: str | Path | None = None) -> ProjectLock:
    target = Path(path) if path else lock_path()
    if not target.is_file():
        raise ConfigurationError(
            f"aire.lock not found at {target}",
            code="project.lock_missing",
            context={"path": str(target), "hint": "AI.project.lock.write(...)"},
        )
    data = read_json_file(target)
    if isinstance(data, ProjectLock):
        return data
    return ProjectLock.model_validate(data)


def write_lock(lock: ProjectLock, path: str | Path | None = None) -> Path:
    target = Path(path) if path else lock_path()
    from aire._version import __version__

    if not lock.aire_version:
        lock.aire_version = __version__
    write_json_file(target, lock)
    return target


def create_lock(
    project: str,
    *,
    model: str | None = None,
    embedder: str | None = None,
    vector_store: str | None = None,
    graph_store: str | None = None,
    **extra: str,
) -> ProjectLock:
    from aire._version import __version__

    lock = ProjectLock(project=project, aire_version=__version__)
    if model:
        lock.set_ref("model", model)
    if embedder:
        lock.set_ref("embedder", embedder)
    if vector_store:
        lock.set_ref("vector_store", vector_store)
    if graph_store:
        lock.set_ref("graph_store", graph_store)
    for kind, ref in extra.items():
        lock.set_ref(kind, ref)
    return lock


def describe() -> dict[str, Any]:
    return {
        "kind": "project_lock",
        "file": "aire.lock",
        "entry_kinds": ["model", "embedder", "vector_store", "graph_store", "tool"],
    }

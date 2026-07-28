"""Safe, uniform serialization.

All persistence (checkpoints, traces, memory, caches) goes through these
helpers. YAML loading is always ``safe_load``; pickle is never used anywhere in
the library (see SECURITY_MODEL.md).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel

from aire.core.errors import DataError

T = TypeVar("T", bound=BaseModel)


def dump_json(model: BaseModel, *, indent: int | None = None) -> str:
    return model.model_dump_json(indent=indent)


def load_json(model_cls: type[T], raw: str | bytes) -> T:
    try:
        return model_cls.model_validate_json(raw)
    except ValueError as exc:
        raise DataError(
            f"invalid JSON payload for {model_cls.__name__}: {exc}",
            code="serialization.json_invalid",
            cause=exc,
        ) from exc


def dump_dict(model: BaseModel) -> dict[str, Any]:
    return model.model_dump(mode="json")


def read_json_file(path: str | Path) -> Any:
    p = Path(path)
    try:
        return json.loads(p.read_text())
    except FileNotFoundError:
        raise DataError(f"file not found: {p}", context={"path": str(p)}) from None
    except json.JSONDecodeError as exc:
        raise DataError(f"invalid JSON in {p}: {exc}", context={"path": str(p)}, cause=exc) from exc


def write_json_file(path: str | Path, data: Any, *, indent: int = 2) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, BaseModel):
        p.write_text(data.model_dump_json(indent=indent))
    else:
        p.write_text(json.dumps(data, indent=indent, default=str))
    return p


def read_yaml_file(path: str | Path) -> Any:
    """Load YAML using safe_load only — never constructs arbitrary objects."""
    p = Path(path)
    try:
        return yaml.safe_load(p.read_text())
    except FileNotFoundError:
        raise DataError(f"file not found: {p}", context={"path": str(p)}) from None
    except yaml.YAMLError as exc:
        raise DataError(f"invalid YAML in {p}: {exc}", context={"path": str(p)}, cause=exc) from exc


def iter_jsonl(path: str | Path) -> Any:
    """Yield parsed objects from a JSON-lines file, with line numbers in errors."""
    p = Path(path)
    if not p.is_file():
        raise DataError(f"file not found: {p}", context={"path": str(p)})
    with p.open() as fh:
        for lineno, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise DataError(
                    f"invalid JSON on line {lineno} of {p}: {exc}",
                    context={"path": str(p), "line": lineno},
                    cause=exc,
                ) from exc


def write_jsonl(path: str | Path, records: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as fh:
        for record in records:
            if isinstance(record, BaseModel):
                fh.write(record.model_dump_json() + "\n")
            else:
                fh.write(json.dumps(record, default=str) + "\n")
    return p

"""Load datasets from files, directories, URLs and in-memory values.

``load(...)`` auto-detects the source type. Every loader validates paths to
prevent directory traversal outside explicitly allowed roots when sandboxing
is requested.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from aire.core.errors import DataError
from aire.core.serialization import iter_jsonl, read_json_file
from aire.data.dataset import Dataset
from aire.data.types import Record

TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst", ".log"}


def load(
    source: str | Path | list[str] | list[dict[str, Any]],
    *,
    text_field: str = "text",
    glob: str | None = None,
    name: str | None = None,
    sandbox_root: str | Path | None = None,
) -> Dataset:
    """Load a dataset from a path, URL, directory, or in-memory values.

    Args:
        source: File path (.jsonl/.json/.csv/text), directory of text files,
            ``http(s)://`` URL, or a list of strings/dicts.
        text_field: Field name carrying the text in structured sources.
        glob: Pattern for directory sources (default: text-like files).
        name: Dataset name (defaults to the source stem).
        sandbox_root: If set, refuse to read files outside this directory.
    """
    if isinstance(source, list):
        rows = [{"text": x} if isinstance(x, str) else x for x in source]
        return Dataset.from_dicts(rows, text_field=text_field, name=name or "memory")
    if isinstance(source, Path) or _looks_local(str(source)):
        return _load_path(
            Path(source), text_field=text_field, glob=glob, name=name, sandbox_root=sandbox_root
        )
    text = str(source)
    if text.startswith(("http://", "https://")):
        return _load_url(text, text_field=text_field, name=name)
    raise DataError(
        f"unsupported data source: {source!r}",
        code="data.source_unsupported",
        context={"source": str(source)},
    )


def _looks_local(value: str) -> bool:
    return not value.startswith(("http://", "https://"))


def _check_sandbox(path: Path, sandbox_root: str | Path | None) -> Path:
    resolved = path.resolve()
    if sandbox_root is not None:
        root = Path(sandbox_root).resolve()
        if root != resolved and root not in resolved.parents:
            raise DataError(
                f"path {resolved} escapes sandbox root {root}",
                code="data.path_traversal",
                context={"path": str(resolved), "root": str(root)},
            )
    return resolved


def _load_directory(
    path: Path,
    *,
    glob: str | None,
    name: str,
    sandbox_root: str | Path | None,
) -> Dataset:
    pattern = glob or "**/*"
    records: list[Record] = []
    for file in sorted(path.glob(pattern)):
        if not file.is_file() or file.suffix.lower() not in TEXT_SUFFIXES:
            continue
        checked = _check_sandbox(file, sandbox_root)
        records.append(
            Record(
                text=checked.read_text(errors="replace"),
                metadata={"source": str(checked), "filename": checked.name},
            )
        )
    if not records:
        raise DataError(
            f"no text files found under {path}",
            code="data.empty_source",
            context={"path": str(path), "glob": pattern},
        )
    return Dataset(records, name=name, source=str(path))


def _load_path(
    path: Path,
    *,
    text_field: str,
    glob: str | None,
    name: str | None,
    sandbox_root: str | Path | None,
) -> Dataset:
    path = _check_sandbox(path, sandbox_root)
    if not path.exists():
        raise DataError(f"data source not found: {path}", context={"path": str(path)})
    ds_name = name or path.stem
    if path.is_dir():
        return _load_directory(path, glob=glob, name=ds_name, sandbox_root=sandbox_root)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows = list(iter_jsonl(path))
        return Dataset.from_dicts(_ensure_dicts(rows, path), text_field=text_field, name=ds_name)
    if suffix == ".json":
        rows = read_json_file(path)
        if isinstance(rows, dict):
            rows = rows.get("records", [rows])
        return Dataset.from_dicts(_ensure_dicts(rows, path), text_field=text_field, name=ds_name)
    if suffix == ".csv":
        with path.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
        return Dataset.from_dicts(rows, text_field=text_field, name=ds_name)
    if suffix in TEXT_SUFFIXES:
        return Dataset(
            [
                Record(
                    text=path.read_text(errors="replace"),
                    metadata={"source": str(path), "filename": path.name},
                )
            ],
            name=ds_name,
            source=str(path),
        )
    raise DataError(
        f"unsupported file type {suffix!r}",
        code="data.type_unsupported",
        context={"path": str(path)},
    )


def _ensure_dicts(rows: Any, path: Path) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        raise DataError(f"{path} must contain a list of records", context={"path": str(path)})
    result: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        if isinstance(row, str):
            result.append({"text": row})
        elif isinstance(row, dict):
            result.append(row)
        else:
            raise DataError(
                f"record {i} in {path} is neither an object nor a string",
                code="data.record_invalid",
                context={"path": str(path), "index": i},
            )
    return result


def _load_url(url: str, *, text_field: str, name: str | None) -> Dataset:
    import httpx

    try:
        response = httpx.get(url, follow_redirects=True, timeout=30.0)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise DataError(
            f"failed to fetch {url}: {exc}",
            code="data.fetch_failed",
            context={"url": url},
            cause=exc,
        ) from exc
    content_type = response.headers.get("content-type", "")
    if "json" in content_type or url.endswith((".json", ".jsonl")):
        try:
            if url.endswith(".jsonl"):
                rows = [json.loads(line) for line in response.text.splitlines() if line.strip()]
            else:
                rows = response.json()
                if isinstance(rows, dict):
                    rows = rows.get("records", [rows])
        except (json.JSONDecodeError, ValueError) as exc:
            raise DataError(
                f"invalid JSON from {url}",
                code="data.parse_failed",
                context={"url": url},
                cause=exc,
            ) from exc
        return Dataset.from_dicts(
            _ensure_dicts(rows, Path(url)), text_field=text_field, name=name or "remote"
        )
    return Dataset(
        [Record(text=response.text, metadata={"source": url})],
        name=name or "remote",
        source=url,
    )

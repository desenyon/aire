"""The Dataset: an immutable, chainable collection of records.

All transformation methods return a *new* Dataset with lineage appended, so a
pipeline description doubles as its own provenance record.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from aire.core.errors import DataError
from aire.core.types import new_id as _new_id
from aire.data.types import (
    DatasetInfo,
    DatasetSplit,
    LineageEntry,
    QualityReport,
    Record,
)


def new_record_id() -> str:
    return _new_id("rec")


class Dataset:
    """An ordered, immutable set of :class:`Record` objects."""

    def __init__(
        self,
        records: list[Record],
        *,
        name: str = "dataset",
        source: str | None = None,
        lineage: list[LineageEntry] | None = None,
    ) -> None:
        self._records = tuple(records)
        self.name = name
        self.source = source
        self.lineage: tuple[LineageEntry, ...] = tuple(lineage or ())

    # -- basics ----------------------------------------------------------------------

    def __iter__(self) -> Iterator[Record]:
        return iter(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> Record:
        return self._records[index]

    @property
    def records(self) -> list[Record]:
        return list(self._records)

    @property
    def texts(self) -> list[str]:
        return [r.text for r in self._records]

    @property
    def version(self) -> str:
        """Content-addressed version: changes iff any record changes."""
        digest = hashlib.sha256()
        for record in self._records:
            digest.update(record.fingerprint().encode())
        return digest.hexdigest()[:12]

    def info(self) -> DatasetInfo:
        return DatasetInfo(
            name=self.name,
            count=len(self),
            version=self.version,
            source=self.source,
            lineage=list(self.lineage),
        )

    def describe(self) -> dict[str, Any]:
        return self.info().model_dump(mode="json")

    # -- chainable transforms -----------------------------------------------------------

    def _derive(self, records: list[Record], operation: str, **detail: Any) -> Dataset:
        entry = LineageEntry(operation=operation, detail={**detail, "count": len(records)})
        return Dataset(
            records,
            name=self.name,
            source=self.source,
            lineage=[*self.lineage, entry],
        )

    def map(self, fn: Callable[[Record], Record]) -> Dataset:
        """Transform each record."""
        return self._derive([fn(r) for r in self._records], "map", fn=getattr(fn, "__name__", "?"))

    def filter(self, predicate: Callable[[Record], bool]) -> Dataset:
        """Keep records where predicate(record) is true."""
        return self._derive(
            [r for r in self._records if predicate(r)],
            "filter",
            fn=getattr(predicate, "__name__", "?"),
        )

    def validate(
        self,
        *,
        min_length: int = 1,
        max_length: int | None = None,
        required_metadata: list[str] | None = None,
    ) -> Dataset:
        """Drop invalid records and raise if everything is dropped."""
        required = required_metadata or []

        def _valid(r: Record) -> bool:
            if len(r.text) < min_length:
                return False
            if max_length is not None and len(r.text) > max_length:
                return False
            return all(key in r.metadata for key in required)

        kept = [r for r in self._records if _valid(r)]
        if self._records and not kept:
            raise DataError(
                "validation removed every record",
                code="data.validation_empty",
                context={"min_length": min_length, "required_metadata": required},
            )
        return self._derive(kept, "validate", dropped=len(self._records) - len(kept))

    def deduplicate(self, *, key: Callable[[Record], str] | None = None) -> Dataset:
        """Remove exact duplicates by text fingerprint (or custom key)."""
        key_fn = key or (lambda r: r.fingerprint())
        seen: set[str] = set()
        unique: list[Record] = []
        for record in self._records:
            fingerprint = key_fn(record)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            unique.append(record)
        return self._derive(unique, "deduplicate", removed=len(self._records) - len(unique))

    def sample(self, n: int | None = None, *, frac: float | None = None, seed: int = 42) -> Dataset:
        """Deterministically sample records."""
        rng = random.Random(seed)  # noqa: S311 - sampling, not security
        count = n if n is not None else int(len(self._records) * (frac or 1.0))
        count = max(0, min(count, len(self._records)))
        picked = rng.sample(list(self._records), count)
        return self._derive(picked, "sample", n=count, seed=seed)

    def split(
        self,
        *,
        train: float = 0.8,
        validation: float = 0.1,
        test: float = 0.1,
        seed: int = 42,
    ) -> DatasetSplits:
        """Reproducibly split into train/validation/test datasets."""
        total = train + validation + test
        if abs(total - 1.0) > 1e-6:
            raise DataError(
                f"split fractions must sum to 1.0, got {total}",
                code="data.split_invalid",
                context={"train": train, "validation": validation, "test": test},
            )
        rng = random.Random(seed)  # noqa: S311 - splitting, not security
        shuffled = list(self._records)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(n * train)
        n_val = int(n * validation)
        meta = DatasetSplit(
            train_count=n_train,
            validation_count=n_val,
            test_count=n - n_train - n_val,
            seed=seed,
        )
        return DatasetSplits(
            train=self._derive(shuffled[:n_train], "split.train", seed=seed),
            validation=self._derive(
                shuffled[n_train : n_train + n_val], "split.validation", seed=seed
            ),
            test=self._derive(shuffled[n_train + n_val :], "split.test", seed=seed),
            info=meta,
        )

    def take(self, n: int) -> Dataset:
        return self._derive(list(self._records[:n]), "take", n=n)

    # -- analysis -----------------------------------------------------------------------

    def quality_report(self, *, pii: bool = True) -> QualityReport:
        """Compute a quality summary (counts, lengths, duplicates, PII suspects)."""
        lengths = [len(r.text) for r in self._records]
        fingerprints = [r.fingerprint() for r in self._records]
        duplicate = len(fingerprints) - len(set(fingerprints))
        pii_suspects = 0
        issues: list[str] = []
        if pii and self._records:
            from aire.safety.patterns import detect_pii

            pii_suspects = sum(1 for r in self._records if detect_pii(r.text))
            if pii_suspects:
                issues.append(f"{pii_suspects} record(s) contain suspected PII")
        empty = sum(1 for ln in lengths if ln == 0)
        if empty:
            issues.append(f"{empty} empty record(s)")
        return QualityReport(
            total=len(self._records),
            empty=empty,
            duplicate=duplicate,
            avg_length=(sum(lengths) / len(lengths)) if lengths else 0.0,
            min_length=min(lengths, default=0),
            max_length=max(lengths, default=0),
            pii_suspects=pii_suspects,
            issues=issues,
        )

    # -- persistence -----------------------------------------------------------------------

    def to_jsonl(self, path: str | Path) -> Path:
        from aire.core.serialization import write_jsonl

        return write_jsonl(path, self._records)

    def to_list(self) -> list[dict[str, Any]]:
        return [r.model_dump(mode="json") for r in self._records]

    @classmethod
    def from_texts(cls, texts: list[str], *, name: str = "dataset") -> Dataset:
        return cls([Record(text=t) for t in texts], name=name)

    @classmethod
    def from_dicts(
        cls, rows: list[dict[str, Any]], *, text_field: str = "text", name: str = "dataset"
    ) -> Dataset:
        records = []
        for row in rows:
            if text_field not in row:
                raise DataError(
                    f"row missing text field {text_field!r}",
                    code="data.field_missing",
                    context={"keys": sorted(row)},
                )
            # Rows written by Dataset.to_jsonl already carry Record shape;
            # preserve their id and metadata instead of nesting them.
            if isinstance(row.get("metadata"), dict) and set(row) <= {"id", text_field, "metadata"}:
                records.append(
                    Record(
                        id=str(row.get("id") or new_record_id()),
                        text=str(row[text_field]),
                        metadata=dict(row["metadata"]),
                    )
                )
                continue
            metadata = {k: v for k, v in row.items() if k != text_field}
            records.append(Record(text=str(row[text_field]), metadata=metadata))
        return cls(records, name=name)


class DatasetSplits:
    """Result of :meth:`Dataset.split`."""

    def __init__(
        self, *, train: Dataset, validation: Dataset, test: Dataset, info: DatasetSplit
    ) -> None:
        self.train = train
        self.validation = validation
        self.test = test
        self.info = info

    def describe(self) -> dict[str, Any]:
        result: dict[str, Any] = json.loads(self.info.model_dump_json())
        return result

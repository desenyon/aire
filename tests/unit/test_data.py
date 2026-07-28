"""Dataset operations, loaders and chunkers."""

from __future__ import annotations

from pathlib import Path

import pytest

from aire.core.errors import DataError
from aire.data import (
    Dataset,
    FixedChunker,
    Record,
    RecursiveChunker,
    SentenceChunker,
    get_chunker,
    load,
)


def _dataset() -> Dataset:
    return Dataset.from_texts(["alpha", "beta", "alpha", "", "gamma" * 100])


def test_validate_dedupe_chain() -> None:
    ds = _dataset().validate().deduplicate()
    assert ds.texts.count("alpha") == 1
    assert "" not in ds.texts
    assert any(entry.operation == "deduplicate" for entry in ds.lineage)


def test_validate_all_dropped_raises() -> None:
    with pytest.raises(DataError) as excinfo:
        Dataset.from_texts(["", ""]).validate()
    assert excinfo.value.code == "data.validation_empty"


def test_split_reproducible() -> None:
    ds = Dataset.from_texts([f"doc-{i}" for i in range(100)])
    a = ds.split(train=0.8, validation=0.1, test=0.1, seed=7)
    b = ds.split(train=0.8, validation=0.1, test=0.1, seed=7)
    assert len(a.train) == 80 and len(a.validation) == 10 and len(a.test) == 10
    assert a.train.texts == b.train.texts
    assert a.info.train_count == 80


def test_split_fractions_must_sum() -> None:
    with pytest.raises(DataError):
        Dataset.from_texts(["x"]).split(train=0.5, validation=0.3, test=0.3)


def test_sample_deterministic() -> None:
    ds = Dataset.from_texts([f"t{i}" for i in range(50)])
    assert ds.sample(n=5, seed=1).texts == ds.sample(n=5, seed=1).texts
    assert len(ds.sample(frac=0.2, seed=1)) == 10


def test_version_changes_with_content() -> None:
    assert Dataset.from_texts(["a"]).version != Dataset.from_texts(["b"]).version
    assert Dataset.from_texts(["a"]).version == Dataset.from_texts(["a"]).version


def test_quality_report_flags_pii() -> None:
    ds = Dataset.from_texts(["contact me at jane@example.com", "clean text"])
    report = ds.quality_report()
    assert report.pii_suspects == 1
    assert not report.ok


def test_jsonl_roundtrip(tmp_path: Path) -> None:
    ds = Dataset.from_dicts([{"text": "one", "tag": "x"}, {"text": "two"}])
    path = ds.to_jsonl(tmp_path / "out.jsonl")
    loaded = load(path)
    assert loaded.texts == ["one", "two"]
    assert loaded.records[0].metadata["tag"] == "x"


def test_load_directory(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("markdown doc")
    (tmp_path / "b.txt").write_text("text doc")
    (tmp_path / "c.bin").write_bytes(b"\x00")
    ds = load(tmp_path)
    assert len(ds) == 2
    assert all("source" in r.metadata for r in ds)


def test_load_csv(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    path.write_text("text,label\nhello,greet\nbye,farewell\n")
    ds = load(path)
    assert ds.records[1].metadata["label"] == "farewell"


def test_load_missing_file() -> None:
    with pytest.raises(DataError):
        load("/nonexistent/path/data.jsonl")


def test_load_sandbox_blocks_traversal(tmp_path: Path) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("classified")
    sandbox = tmp_path / "public"
    sandbox.mkdir()
    with pytest.raises(DataError) as excinfo:
        load(secret, sandbox_root=sandbox)
    assert excinfo.value.code == "data.path_traversal"


def test_load_from_memory() -> None:
    ds = load(["just a string", {"text": "structured"}])
    assert len(ds) == 2


def test_chunkers_cover_text() -> None:
    text = ("Sentence one is here. Sentence two follows it. " * 20).strip()
    for chunker in (
        FixedChunker(size=100, overlap=20),
        SentenceChunker(size=120),
        RecursiveChunker(size=100),
        get_chunker("semantic", size=120),
    ):
        chunks = chunker.chunk(text)
        assert chunks
        assert all(len(c.text) > 0 for c in chunks)
        reassembled = " ".join(c.text for c in chunks)
        assert "Sentence one" in reassembled and "Sentence two" in reassembled


def test_fixed_chunker_offsets_increase() -> None:
    chunks = FixedChunker(size=50, overlap=10).chunk("x" * 200)
    starts = [c.start for c in chunks]
    assert starts == sorted(starts)
    assert len(chunks) > 1


def test_record_fingerprint_stable() -> None:
    assert Record(text="same").fingerprint() == Record(text="same").fingerprint()

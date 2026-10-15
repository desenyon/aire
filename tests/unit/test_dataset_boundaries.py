import math

import pytest

from aire.core.errors import DataError
from aire.data.dataset import Dataset
from aire.data.types import LineageEntry, Record


def test_dataset_copies_constructor_inputs_and_all_record_accessors():
    record = Record(text="original", metadata={"labels": ["a"]})
    lineage = LineageEntry(operation="load", detail={"nested": ["original"]})
    dataset = Dataset([record], lineage=[lineage])
    version = dataset.version
    record.text = "changed"
    record.metadata["labels"].append("b")
    lineage.detail["nested"].append("changed")
    for snapshot in (dataset[0], dataset.records[0], next(iter(dataset))):
        snapshot.text = "changed again"
        snapshot.metadata["labels"].append("c")
    dataset.lineage[0].detail["nested"].append("again")
    assert dataset.texts == ["original"]
    assert dataset[0].metadata == {"labels": ["a"]}
    assert dataset.lineage[0].detail == {"nested": ["original"]}
    assert dataset.version == version


@pytest.mark.parametrize("operation", ["map", "filter", "deduplicate"])
def test_mutating_callback_cannot_change_source_dataset(operation):
    dataset = Dataset([Record(text="original", metadata={"nested": []})])

    def mutate(record):
        record.metadata["nested"].append("changed")
        record.text = "changed"
        if operation == "map":
            return record
        if operation == "filter":
            return True
        return record.text

    derived = getattr(dataset, operation)(
        *([mutate] if operation != "deduplicate" else []),
        **({"key": mutate} if operation == "deduplicate" else {}),
    )
    assert dataset[0].metadata == {"nested": []}
    assert dataset.texts == ["original"]
    if operation == "map":
        assert derived.texts == ["changed"]


def test_failed_transform_preserves_source():
    dataset = Dataset.from_texts(["original"])

    def fail(record):
        record.text = "corrupted"
        raise RuntimeError("transform failed")

    with pytest.raises(RuntimeError):
        dataset.map(fail)
    assert dataset.texts == ["original"]


def test_sampling_zero_and_reproducibility():
    dataset = Dataset.from_texts([str(i) for i in range(10)])
    assert len(dataset.sample(frac=0)) == 0
    assert len(dataset.sample(n=0)) == 0
    assert len(dataset.sample()) == 10
    assert dataset.sample(frac=0.3).texts == dataset.sample(n=3).texts
    assert len(dataset.sample(n=100)) == 10


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n": -1},
        {"n": 1.5},
        {"n": True},
        {"n": 1, "frac": 0.5},
        {"frac": -0.1},
        {"frac": 1.1},
        {"frac": math.nan},
        {"frac": math.inf},
    ],
)
def test_invalid_samples_are_structured_errors(kwargs):
    with pytest.raises(DataError) as error:
        Dataset.from_texts(["one"]).sample(**kwargs)
    assert error.value.code == "data.sample_invalid"


@pytest.mark.parametrize(
    "fractions", [(-0.2, 0.6, 0.6), (1.1, -0.1, 0), (math.nan, 0, 1), (math.inf, 0, 0), (0, 0, 0)]
)
def test_invalid_splits_fail_before_partitioning(fractions):
    with pytest.raises(DataError) as error:
        Dataset.from_texts(["a", "b"]).split(
            train=fractions[0], validation=fractions[1], test=fractions[2]
        )
    assert error.value.code == "data.split_invalid"


@pytest.mark.parametrize("size", [0, 1, 7, 10])
def test_split_conserves_records_and_matches_manifest(size):
    dataset = Dataset.from_texts([str(i) for i in range(size)])
    split = dataset.split(train=0.6, validation=0.2, test=0.2)
    partitions = [split.train, split.validation, split.test]
    assert [len(part) for part in partitions] == [
        split.info.train_count,
        split.info.validation_count,
        split.info.test_count,
    ]
    assert sorted(record.id for part in partitions for record in part) == sorted(
        record.id for record in dataset
    )

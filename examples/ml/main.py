"""Model creation through aire — runs fully offline with native estimators.

    python examples/ml/main.py

aire orchestrates the ML ecosystem: native estimators work with zero
dependencies, and the same code runs on scikit-learn (``sklearn:random_forest``)
or PyTorch (``torch:mlp``) once those extras are installed.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from aire import AI
from aire.data.dataset import Dataset
from aire.data.types import Record


def toy_dataset() -> Dataset:
    """Two separable clusters labeled low/high."""
    records: list[Record] = []
    for i in range(15):
        records.append(
            Record(
                text=f"short note {i}",
                metadata={"features": {"length": float(i), "code_ratio": 0.05}, "label": "note"},
            )
        )
    for i in range(15):
        records.append(
            Record(
                text=f"long technical document {i}",
                metadata={
                    "features": {"length": float(200 + i), "code_ratio": 0.6},
                    "label": "technical",
                },
            )
        )
    return Dataset(records, name="docs")


async def main() -> None:
    print("backends available:", AI.ml.backends())

    dataset = toy_dataset()

    # 1. Create + fit in one call (native centroid classifier, zero deps)
    estimator = await AI.ml.fit("simple:centroid", dataset)
    print("fit report:", estimator.report.model_dump(exclude={"feature_names"}))

    # 2. Evaluate on held-out-style data
    metrics = await estimator.evaluate(dataset)
    print("evaluation:", metrics)

    # 3. Predict on new, unlabeled records
    new_records = [
        Record(text="quick memo", metadata={"features": {"length": 4.0, "code_ratio": 0.02}}),
        Record(
            text="architecture deep dive",
            metadata={"features": {"length": 340.0, "code_ratio": 0.7}},
        ),
    ]
    for prediction in await estimator.predict(new_records):
        print(f"  predicted {prediction.value!r} for {prediction.record_id}")

    # 4. Persist + reload
    with tempfile.TemporaryDirectory() as tmp:
        path = estimator.save(Path(tmp) / "classifier.json")
        reloaded = AI.ml.create("simple:centroid").load(path)
        again = await reloaded.predict(new_records)
        print("reloaded model agrees:", [p.value for p in again])

    # 5. Same contract on other backends (when installed):
    #    est = await AI.ml.fit("sklearn:random_forest", dataset, n_estimators=50)
    #    est = await AI.ml.fit("torch:mlp", dataset, hidden=(32,), epochs=300)
    #    df  = AI.ml.to_frame(dataset)          # pandas bridge
    #    ds  = AI.ml.from_frame(df, target="label")

    # 6. Regression works too
    reg_data = Dataset(
        [
            Record(
                text=f"house {i}",
                metadata={
                    "features": {"sqft": float(500 + 100 * i)},
                    "price": 100_000 + 20_000 * i,
                },
            )
            for i in range(30)
        ],
        name="housing",
    )
    regressor = await AI.ml.fit("simple:linear_regression", reg_data, target="price", epochs=800)
    print("regression:", await regressor.evaluate(reg_data, target="price"))


if __name__ == "__main__":
    asyncio.run(main())

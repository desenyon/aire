"""Synthetic data generation driven by any aire model.

Generates evaluation QA pairs and augmentation records from seed documents,
validated through structured output so malformed generations are retried or
dropped rather than silently polluting datasets.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from aire.data.dataset import Dataset
from aire.data.types import Record
from aire.models.base import Model


class QAPair(BaseModel):
    question: str
    answer: str


class QABatch(BaseModel):
    pairs: list[QAPair] = Field(default_factory=list)


_QA_PROMPT = (
    "Generate {n} diverse question/answer pairs strictly grounded in the "
    "document below. Return JSON with a 'pairs' array of objects with "
    "'question' and 'answer' string fields.\n\nDocument:\n{document}"
)


class SyntheticGenerator:
    """Creates synthetic datasets from seed content using a model."""

    def __init__(self, model: Model) -> None:
        self.model = model

    async def qa_pairs(self, document: str, *, n: int = 5) -> list[QAPair]:
        """Generate n QA pairs grounded in a document."""
        batch = await self.model.ask_structured(
            _QA_PROMPT.format(n=n, document=document[:8000]), QABatch, retries=2
        )
        assert isinstance(batch, QABatch)
        return batch.pairs[:n]

    async def augment(self, dataset: Dataset, *, pairs_per_doc: int = 3) -> Dataset:
        """Turn a document dataset into a QA evaluation dataset."""
        records: list[Record] = []
        for record in dataset:
            pairs = await self.qa_pairs(record.text, n=pairs_per_doc)
            for pair in pairs:
                records.append(
                    Record(
                        text=pair.question,
                        metadata={
                            "expected": pair.answer,
                            "context": record.text[:2000],
                            "synthetic": True,
                            "source_record": record.id,
                        },
                    )
                )
        return Dataset(records, name=f"{dataset.name}-synthetic", source=dataset.source)

    def describe(self) -> dict[str, Any]:
        return {"kind": "synthetic_generator", "model": self.model.info.ref}

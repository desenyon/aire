"""Training orchestration, synthetic generation, multimodal conversions, vision/audio."""

from __future__ import annotations

from pathlib import Path

import pytest

from aire.core.content import ImageContent, TextContent
from aire.core.errors import NotFoundError
from aire.core.types import Capability
from aire.data import Dataset
from aire.models.builtin import EchoModel
from aire.models.types import ModelInfo
from aire.multimodal import ConversionRegistry, ModelConverter, describe_image
from aire.synthetic import SyntheticGenerator
from aire.training import FunctionTrainer, TrainingConfig
from aire.vision import VisionPipeline
from tests.conftest import arun


def test_function_trainer_epochs_and_checkpoints(tmp_path: Path) -> None:
    def step(epoch: int, dataset: Dataset, config: TrainingConfig, state: dict) -> tuple:
        return {"loss": 1.0 / (epoch + 1)}, {"seen": len(dataset)}

    trainer = FunctionTrainer(
        step,
        TrainingConfig(epochs=4, checkpoint_dir=str(tmp_path / "ckpts")),
    )
    result = arun(trainer.fit(Dataset.from_texts(["a", "b"])))
    assert result.epochs_completed == 4
    assert result.best_metric == 0.25
    assert len(result.checkpoints) == 4
    assert (tmp_path / "ckpts" / "checkpoint-epoch3.json").is_file()


def test_function_trainer_early_stopping() -> None:
    def step(epoch: int, dataset: Dataset, config: TrainingConfig, state: dict) -> tuple:
        return {"loss": 1.0 if epoch == 0 else 2.0}, {}

    trainer = FunctionTrainer(step, TrainingConfig(epochs=10, early_stopping_patience=2))
    result = arun(trainer.fit(Dataset.from_texts(["x"])))
    assert result.stopped_early
    assert result.epochs_completed == 3


def test_synthetic_qa_generation() -> None:
    generator = SyntheticGenerator(EchoModel())
    pairs = arun(generator.qa_pairs("Refunds are allowed within 30 days.", n=3))
    assert len(pairs) == 2  # offline stub emits two sample pairs
    assert all(p.question.startswith("mock-") for p in pairs)  # echo structured stub


def test_synthetic_augment_dataset() -> None:
    generator = SyntheticGenerator(EchoModel())
    augmented = arun(generator.augment(Dataset.from_texts(["doc one", "doc two"]), pairs_per_doc=2))
    assert len(augmented) == 4
    assert all(r.metadata["synthetic"] for r in augmented)


def test_conversion_registry() -> None:
    registry = ConversionRegistry()
    converter = ModelConverter(EchoModel(), "image", "text", prompt="describe")
    registry.register(converter)
    assert registry.get("image", "text") is converter
    with pytest.raises(NotFoundError):
        registry.get("video", "text")


def test_model_converter_delegates() -> None:
    converter = ModelConverter(EchoModel(), "image", "text", prompt="describe this")
    image = ImageContent(uri="https://example.com/x.png")
    result = arun(converter.convert(image))
    assert isinstance(result, TextContent)
    assert "describe this" in result.text


class _VisionEcho(EchoModel):
    @property
    def info(self) -> ModelInfo:
        return super().info.model_copy(
            update={"capabilities": [*super().info.capabilities, Capability.VISION_INPUT]}
        )


def test_vision_pipeline_requires_capability() -> None:
    with pytest.raises(NotFoundError):
        VisionPipeline(EchoModel())


def test_vision_pipeline_classify() -> None:
    pipeline = VisionPipeline(_VisionEcho())
    image = ImageContent(uri="https://example.com/cat.png")
    result = arun(pipeline.classify(image, ["cat", "dog"]))
    assert result.text  # echo returns the classification prompt
    assert result.model.startswith("mock:")


def test_describe_image_helper() -> None:
    result = arun(describe_image(_VisionEcho(), ImageContent(uri="https://x/y.png")))
    assert isinstance(result, TextContent)

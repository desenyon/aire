"""Training depth tests (offline)."""

from __future__ import annotations

from aire.data.dataset import Dataset
from aire.models.base import run_sync
from aire.training.distill import HFDistillTrainer, soft_kl_loss
from aire.training.foundation import FoundationModel, create_foundation
from aire.training.lm_trainer import create_lm_trainer
from aire.training.lora import create_lora


def test_foundation_toy_kind() -> None:
    fm = create_foundation("gpt2", n_layer=2, n_embd=64, n_head=2, vocab_size=128)
    desc = fm.describe()
    assert desc["kind"] == "foundation_toy_architecture"
    assert desc["pretrained"] is False


def test_foundation_catalog_mentions_from_pretrained() -> None:
    from aire.training.foundation import catalog

    cat = catalog()
    assert "from_pretrained" in cat
    assert "toy" in cat["honesty"]


def test_lora_dry_run_and_resume_api() -> None:
    trainer = create_lora("gpt2", dry_run=True, output_dir="./.tmp-lora-test")
    result = trainer.fit(["hello world", "another line"], epochs=2)
    assert result.epochs_completed == 2
    assert "resume" in trainer.describe()["methods"]
    resumed = trainer.resume(dataset=["hello again"], epochs=1, dry_run=True)
    assert resumed.epochs_completed == 1


def test_hf_distill_dry_run() -> None:
    kd = HFDistillTrainer(dry_run=True)
    out = kd.fit(["alpha beta", "gamma delta"], epochs=1)
    assert out["dry_run"] is True
    assert out["kind"] == "hf_distill_result"


def test_soft_kl_and_lm_toy() -> None:
    loss = soft_kl_loss([1.0, 2.0, 0.5], [1.1, 1.9, 0.4])
    assert loss >= 0.0
    trainer = create_lm_trainer(backend="toy", vocab_size=64)
    ds = Dataset.from_texts(["abc", "def"])
    result = run_sync(trainer.fit(ds))
    assert result.epochs_completed >= 1
    assert trainer.describe()["batch_size"] == 4


def test_foundation_model_class_has_from_pretrained() -> None:
    assert callable(FoundationModel.from_pretrained)

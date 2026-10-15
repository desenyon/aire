import math

import pytest

from aire.core.errors import ConfigurationError
from aire.training.distill import HFDistillTrainer

pytestmark = pytest.mark.optional


def test_real_distillation_updates_and_saves_student(tiny_lm, tmp_path):
    import torch

    trainer = HFDistillTrainer(
        str(tiny_lm), str(tiny_lm), output_dir=str(tmp_path / "student"), dry_run=False
    )
    trainer.prepare()
    before = [p.detach().clone() for p in trainer._student.parameters()]
    result = trainer.fit(["hello world", "another line"], epochs=1, batch_size=2, max_length=8)
    assert not result["dry_run"]
    assert math.isfinite(result["history"][0]["loss"])
    assert any(
        not torch.equal(a, b) for a, b in zip(before, trainer._student.parameters(), strict=True)
    )
    assert (tmp_path / "student" / "config.json").exists()


def test_distillation_does_not_claim_training_for_zero_epochs():
    with pytest.raises(ConfigurationError, match="positive"):
        HFDistillTrainer(dry_run=True).fit(["hello"], epochs=0)

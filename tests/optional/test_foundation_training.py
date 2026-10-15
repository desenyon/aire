import pytest

from aire.data.dataset import Dataset
from aire.models.base import run_sync
from aire.training.foundation import create_foundation
from aire.training.trainer import TrainingConfig

pytestmark = pytest.mark.optional


@pytest.mark.parametrize("family", ["gpt2", "llama", "mistral", "moe", "custom"])
def test_tiny_foundation_forward_and_real_training(family):
    torch = pytest.importorskip("torch")
    model = create_foundation(family, n_layer=1, n_embd=16, n_head=2, vocab_size=16, max_seq_len=16)
    logits, _ = model.architecture(torch.tensor([[1, 2, 3]]))
    assert tuple(logits.shape) == (1, 3, 16)
    before = [p.detach().clone() for p in model.architecture.parameters()]
    trainer = model.trainer(
        config=TrainingConfig(epochs=1), vocab_size=16, max_length=8, batch_size=2
    )
    result = run_sync(trainer.fit(Dataset.from_texts(["abc def", "def abc"])))
    assert result.epochs_completed == 1
    assert any(
        not torch.equal(a, b) for a, b in zip(before, model.architecture.parameters(), strict=True)
    )
    assert not model.pretrained

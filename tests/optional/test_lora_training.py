import importlib.util

import pytest

pytestmark = pytest.mark.optional


@pytest.mark.skipif(importlib.util.find_spec("peft") is None, reason="install peft profile")
def test_real_lora_fit_save_and_resume_without_download(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    from aire.training.lora import LoRAConfig, LoRATrainer

    base = tmp_path / "base"
    vocab = {"[UNK]": 0, "[PAD]": 1, "[EOS]": 2, "hello": 3, "world": 4, "another": 5, "line": 6}
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]"
    )
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(vocab),
            n_layer=1,
            n_head=2,
            n_embd=16,
            n_positions=32,
            bos_token_id=2,
            eos_token_id=2,
            pad_token_id=1,
        )
    )
    model.save_pretrained(base)
    fast.save_pretrained(base)
    trainer = LoRATrainer(
        str(base),
        config=LoRAConfig(target_modules=["c_attn"], r=2, dry_run=False),
        output_dir=str(tmp_path / "output"),
    )
    result = trainer.fit(["hello world", "another line"], epochs=1, batch_size=2, max_length=8)
    assert result.epochs_completed == 1
    assert (tmp_path / "output" / "adapter_config.json").exists()
    result = trainer.resume(
        dataset=["hello world", "another line"], epochs=2, batch_size=2, max_length=8
    )
    assert result.epochs_completed == 2

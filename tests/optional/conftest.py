import pytest


@pytest.fixture
def tiny_lm(tmp_path):
    pytest.importorskip("transformers")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

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
    path = tmp_path / "base"
    model.save_pretrained(path)
    fast.save_pretrained(path)
    return path

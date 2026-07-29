"""LoRA dry-run: validates dataset without Hugging Face Trainer (no peft required)."""

from __future__ import annotations

from aire.training.lora import LoRAConfig, LoRATrainer


def main() -> None:
    trainer = LoRATrainer(
        model_name="sshleifer/tiny-gpt2",
        config=LoRAConfig(dry_run=True, r=4),
    )
    result = trainer.fit(
        ["aire is an agent-first library.", "LoRA adapters fine-tune efficiently."],
        dry_run=True,
    )
    print("result:", result.model_dump(mode="json") if hasattr(result, "model_dump") else result)
    print("describe:", trainer.describe())
    print("Live LoRA: pip install 'aire[peft]' and set dry_run=False")


if __name__ == "__main__":
    main()

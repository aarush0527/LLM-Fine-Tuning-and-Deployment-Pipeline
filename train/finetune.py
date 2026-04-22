"""Minimal LoRA fine-tune script for causal language models.

Reads config from train/config.json and cleaned data from data/processed/.
Saves model + tokenizer artifacts to train/outputs/ and writes train_logs.json.

Usage:
    python -m train.finetune                          # uses default config
    python -m train.finetune --config train/config.json
"""

from __future__ import annotations

import json
import logging
import pathlib
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Final

from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

logger = logging.getLogger(__name__)

CONFIG_PATH: Final[pathlib.Path] = pathlib.Path("train/config.json")


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------
@dataclass
class FinetuneConfig:
    """Typed representation of train/config.json."""

    model_name: str = "distilgpt2"
    dataset_dir: str = "data/processed"
    output_dir: str = "train/outputs"
    seed: int = 42
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 2
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_seq_length: int = 256
    fp16: bool = False
    logging_steps: int = 10
    save_strategy: str = "epoch"
    lora: dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": True,
            "r": 8,
            "alpha": 16,
            "dropout": 0.05,
            "target_modules": ["c_attn", "c_proj"],
        }
    )

    @classmethod
    def from_json(cls, path: pathlib.Path) -> FinetuneConfig:
        """Load config from JSON file."""
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        return cls(**raw)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_splits(cfg: FinetuneConfig, tokenizer: AutoTokenizer) -> tuple[Dataset, Dataset]:
    """Load train/val .jsonl splits and tokenize them."""
    dataset_dir = pathlib.Path(cfg.dataset_dir)
    train_path = dataset_dir / "train.jsonl"
    val_path = dataset_dir / "val.jsonl"

    for p in (train_path, val_path):
        if not p.exists():
            print(f"[finetune] Missing split: {p}", file=sys.stderr)
            print("[finetune] Run `python -m data.clean` first.", file=sys.stderr)
            sys.exit(1)

    train_ds = load_dataset("json", data_files=str(train_path), split="train")
    val_ds = load_dataset("json", data_files=str(val_path), split="train")

    def tokenize(batch: dict[str, list]) -> dict[str, list]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=cfg.max_seq_length,
            padding=False,
        )

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    val_ds = val_ds.map(tokenize, batched=True, remove_columns=["text"])
    return train_ds, val_ds


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------
def build_model(cfg: FinetuneConfig) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load base model + tokenizer, apply LoRA if configured."""
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name)

    if cfg.lora.get("enabled", False):
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg.lora["r"],
            lora_alpha=cfg.lora["alpha"],
            lora_dropout=cfg.lora["dropout"],
            target_modules=cfg.lora["target_modules"],
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    return model, tokenizer


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(cfg: FinetuneConfig) -> dict[str, Any]:
    """Execute fine-tuning and return logs dict."""
    set_seed(cfg.seed)
    output_dir = pathlib.Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = build_model(cfg)
    train_ds, val_ds = load_splits(cfg, tokenizer)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        fp16=cfg.fp16,
        logging_steps=cfg.logging_steps,
        eval_strategy="epoch",
        save_strategy=cfg.save_strategy,
        seed=cfg.seed,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    start = time.time()
    train_result = trainer.train()
    elapsed = time.time() - start

    # Save final model + tokenizer
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))

    # Build train_logs
    logs: dict[str, Any] = {
        "config": asdict(cfg),
        "seed": cfg.seed,
        "model_name": cfg.model_name,
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_params": sum(p.numel() for p in model.parameters()),
        "train_loss": train_result.training_loss,
        "train_runtime_seconds": round(elapsed, 2),
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "history": [{k: v for k, v in entry.items()} for entry in trainer.state.log_history],
    }

    logs_path = output_dir / "train_logs.json"
    with open(logs_path, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2, default=str)

    print(f"\n[finetune] Training complete in {elapsed:.1f}s")
    print(f"[finetune] Train loss: {train_result.training_loss:.4f}")
    print(f"[finetune] Artifacts saved to {output_dir / 'final'}")
    print(f"[finetune] Logs written to {logs_path}")
    return logs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune a causal LM with LoRA")
    parser.add_argument("--config", type=pathlib.Path, default=CONFIG_PATH)
    args = parser.parse_args()

    if not args.config.exists():
        print(f"[finetune] Config not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    cfg = FinetuneConfig.from_json(args.config)
    train(cfg)


if __name__ == "__main__":
    main()

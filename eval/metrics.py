"""Evaluation metrics for the fine-tuned language model.

Computes:
  - Perplexity (language modeling fit)
  - ROUGE-1/2/L (summarization quality)
  - Token-level F1 (extraction quality)

Usage:
    python -m eval.metrics --model train/outputs/final --data data/processed/test.jsonl
"""

from __future__ import annotations

import json
import math
import pathlib
import sys
from collections import Counter
from typing import Any, Final

import torch
from datasets import load_dataset
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer

OUTPUT_PATH: Final[pathlib.Path] = pathlib.Path("eval/results.json")


# ---------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------
def compute_perplexity(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: list[str],
    max_length: int = 256,
    device: str = "cpu",
) -> float:
    """Compute perplexity on a list of texts using sliding-window evaluation."""
    model.eval()
    model.to(device)
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            encodings = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            ).to(device)
            input_ids = encodings["input_ids"]

            if input_ids.size(1) < 2:
                continue

            outputs = model(input_ids, labels=input_ids)
            seq_len = input_ids.size(1) - 1
            total_loss += outputs.loss.item() * seq_len
            total_tokens += seq_len

    if total_tokens == 0:
        return float("inf")
    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)


# ---------------------------------------------------------------------------
# ROUGE
# ---------------------------------------------------------------------------
def compute_rouge(
    predictions: list[str],
    references: list[str],
) -> dict[str, float]:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L F1 scores."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    totals: dict[str, float] = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for key in totals:
            totals[key] += scores[key].fmeasure

    n = max(len(predictions), 1)
    return {k: round(v / n, 4) for k, v in totals.items()}


# ---------------------------------------------------------------------------
# Token F1
# ---------------------------------------------------------------------------
def compute_token_f1(prediction: str, reference: str) -> float:
    """Compute token-level F1 between two strings."""
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()

    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)

    common = sum((pred_counts & ref_counts).values())
    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    return round(2 * precision * recall / (precision + recall), 4)


def compute_avg_f1(predictions: list[str], references: list[str]) -> float:
    """Compute average token F1 over paired lists."""
    scores = [compute_token_f1(p, r) for p, r in zip(predictions, references)]
    return round(sum(scores) / max(len(scores), 1), 4)


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------
def evaluate(
    model_path: str,
    data_path: str,
    max_length: int = 256,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run all metrics and return results dict."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_path)

    # Load test data
    ds = load_dataset("json", data_files=data_path, split="train")
    texts = ds["text"]

    # Perplexity
    ppl = compute_perplexity(model, tokenizer, texts, max_length=max_length, device=device)

    # Generate predictions for ROUGE/F1 (use first 50 chars as prompt, rest as reference)
    predictions: list[str] = []
    references: list[str] = []

    model.eval()
    with torch.no_grad():
        for text in texts[:100]:  # cap at 100 samples for speed
            if len(text) < 60:
                continue
            prompt = text[:50]
            reference = text[50:]
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            output_ids = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            generated = tokenizer.decode(
                output_ids[0][inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )
            predictions.append(generated)
            references.append(reference)

    rouge = compute_rouge(predictions, references)
    avg_f1 = compute_avg_f1(predictions, references)

    results: dict[str, Any] = {
        "model_path": model_path,
        "data_path": data_path,
        "num_samples": len(texts),
        "perplexity": round(ppl, 2),
        "rouge1": rouge["rouge1"],
        "rouge2": rouge["rouge2"],
        "rougeL": rouge["rougeL"],
        "token_f1": avg_f1,
    }

    # Write results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 50)
    print("  EVALUATION RESULTS")
    print("=" * 50)
    for k, v in results.items():
        print(f"  {k:<20} {v}")
    print("=" * 50 + "\n")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model")
    parser.add_argument("--model", type=str, default="train/outputs/final")
    parser.add_argument("--data", type=str, default="data/processed/test.jsonl")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    for p in (args.model, args.data):
        if not pathlib.Path(p).exists():
            print(f"[eval] Path not found: {p}", file=sys.stderr)
            sys.exit(1)

    evaluate(args.model, args.data, device=args.device)


if __name__ == "__main__":
    main()

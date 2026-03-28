"""Generate a deliberately dirty dataset for demonstrating data cleaning.

Produces raw/dirty_data.jsonl with realistic noise:
  - Duplicates
  - HTML fragments
  - Broken unicode
  - PII-like patterns (fake SSNs, emails, phones)
  - Overly long lines
  - Boilerplate / legal text
"""

from __future__ import annotations

import json
import pathlib
import random
from typing import Final

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED: Final[int] = 42
N_CLEAN: Final[int] = 500
N_DUPLICATES: Final[int] = 60
OUTPUT_DIR: Final[pathlib.Path] = pathlib.Path(__file__).resolve().parent / "raw"

# ---------------------------------------------------------------------------
# Noise generators
# ---------------------------------------------------------------------------

CLEAN_TEXTS: list[str] = [
    "Machine learning models require careful evaluation to ensure they generalize well.",
    "Gradient descent is an optimization algorithm used to minimize the loss function.",
    "Transformers use self-attention to capture long-range dependencies in sequences.",
    "Fine-tuning a pre-trained model on domain data improves task-specific performance.",
    "Data preprocessing is critical: garbage in, garbage out applies to every ML pipeline.",
    "Regularization techniques like dropout and weight decay help prevent overfitting.",
    "The bias-variance tradeoff is fundamental to understanding model generalization.",
    "Cross-validation provides a robust estimate of model performance on unseen data.",
    "Feature engineering transforms raw inputs into representations the model can learn from.",
    "Batch normalization accelerates training and provides a mild regularizing effect.",
    "Learning rate schedules can improve convergence speed and final model quality.",
    "Tokenization converts raw text into numerical ids that language models consume.",
    "Beam search decoding balances output quality with computational cost.",
    "Low-rank adaptation (LoRA) enables parameter-efficient fine-tuning of large models.",
    "Quantization reduces model size and inference latency with minimal accuracy loss.",
    "The softmax function converts logits into a probability distribution over classes.",
    "Attention mechanisms allow models to weigh the importance of different input tokens.",
    "Encoder-decoder architectures are used for sequence-to-sequence tasks like translation.",
    "Causal language modeling predicts the next token given all previous tokens.",
    "Masked language modeling randomly hides tokens and trains the model to reconstruct them.",
    "Retrieval-augmented generation combines a retriever with a generator for open-domain QA.",
    "Knowledge distillation transfers knowledge from a large teacher to a smaller student model.",
    "Mixed-precision training uses float16 to speed up training while keeping float32 weights.",
    "Contrastive learning teaches models to distinguish between similar and dissimilar pairs.",
    "Prompt engineering designs input templates that steer pre-trained model behavior.",
]

HTML_FRAGMENTS: list[str] = [
    "<div class='content'><p>This is an important paragraph.</p></div>",
    "<html><body><h1>Page Title</h1><p>Some text here.</p></body></html>",
    "Click <a href='http://example.com'>here</a> for more &amp; details.",
    "<span style='color:red'>Warning:</span> <b>deprecated</b> API endpoint.",
    "<table><tr><td>Model</td><td>Accuracy</td></tr></table>",
]

BROKEN_UNICODE: list[str] = [
    "The model\x80s accuracy improved significantly after fine-tuning.",
    "We observed a 15% improvement in F1\x92score on the test set.",
    "Training converged after 10 epochs\x85 results look promising.",
    "The\xc0\xc1 tokenizer handled edge cases poorly at first.",
    "Perplexity dropped from 45\xff to 12 after domain adaptation.",
]

PII_PATTERNS: list[str] = [
    "Contact John Smith at john.smith@example.com for more details.",
    "Call us at 555-123-4567 for support regarding your account.",
    "SSN: 123-45-6789 was found in the training data and should be removed.",
    "Patient record for Jane Doe, DOB 03/15/1985, MRN: 0012345.",
    "Credit card ending in 4242 was used for the transaction on 2024-01-15.",
    "Send documents to 1234 Elm Street, Springfield, IL 62704.",
    "Employee ID E-99182 belongs to user maria.garcia@corp-internal.net.",
]

LONG_LINES: list[str] = [
    " ".join(["This is a very long line that keeps going."] * 80),
    " ".join(["Repeated filler content for stress testing the length filter."] * 60),
]

BOILERPLATE: list[str] = [
    (
        "DISCLAIMER: This document is provided as-is without warranty of any kind. "
        "All rights reserved. Unauthorized reproduction is prohibited. "
        "Copyright 2024 Acme Corp. Terms and conditions apply."
    ),
    (
        "By using this service you agree to our Terms of Service and Privacy Policy. "
        "We may collect and process personal data as described in our data processing agreement. "
        "For questions contact legal@example.com."
    ),
    (
        "This email and any attachments are confidential and intended solely for the addressee. "
        "If you have received this in error, please notify the sender immediately. "
        "Any unauthorized copying or distribution is strictly prohibited."
    ),
]


def _make_clean_records(n: int, rng: random.Random) -> list[dict]:
    """Generate n records from the clean text pool."""
    return [
        {"text": rng.choice(CLEAN_TEXTS), "source": "curated", "label": "clean"} for _ in range(n)
    ]


def _make_duplicates(records: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Clone n existing records (exact duplicates)."""
    return [dict(rng.choice(records)) for _ in range(n)]


def _make_html_records(rng: random.Random) -> list[dict]:
    return [
        {"text": rng.choice(HTML_FRAGMENTS), "source": "web_scrape", "label": "html"}
        for _ in range(20)
    ]


def _make_unicode_records(rng: random.Random) -> list[dict]:
    return [
        {"text": rng.choice(BROKEN_UNICODE), "source": "ocr_pipeline", "label": "unicode"}
        for _ in range(15)
    ]


def _make_pii_records(rng: random.Random) -> list[dict]:
    return [
        {"text": rng.choice(PII_PATTERNS), "source": "mixed", "label": "pii"} for _ in range(25)
    ]


def _make_long_records(rng: random.Random) -> list[dict]:
    return [
        {"text": rng.choice(LONG_LINES), "source": "bulk_import", "label": "long"}
        for _ in range(10)
    ]


def _make_boilerplate_records(rng: random.Random) -> list[dict]:
    return [
        {"text": rng.choice(BOILERPLATE), "source": "template", "label": "boilerplate"}
        for _ in range(15)
    ]


def generate(seed: int = SEED) -> pathlib.Path:
    """Generate dirty_data.jsonl and return its path."""
    rng = random.Random(seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records = _make_clean_records(N_CLEAN, rng)
    records += _make_duplicates(records, N_DUPLICATES, rng)
    records += _make_html_records(rng)
    records += _make_unicode_records(rng)
    records += _make_pii_records(rng)
    records += _make_long_records(rng)
    records += _make_boilerplate_records(rng)

    rng.shuffle(records)

    out_path = OUTPUT_DIR / "dirty_data.jsonl"
    with open(out_path, "w", encoding="utf-8", errors="replace") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[generate] Wrote {len(records)} records -> {out_path}")
    return out_path


if __name__ == "__main__":
    generate()

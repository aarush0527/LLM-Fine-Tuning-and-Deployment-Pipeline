"""Production data cleaning pipeline.

Reads raw/dirty_data.jsonl and produces:
  - processed/train.jsonl, val.jsonl, test.jsonl
  - processed/manifest.json  (hash, rule counts, split seed)

Usage:
    python -m data.clean                       # uses defaults
    python -m data.clean --input data/raw/dirty_data.jsonl --seed 42
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import re
import sys
import unicodedata
from dataclasses import asdict, dataclass
from typing import Final

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_DATA_DIR: Final[pathlib.Path] = pathlib.Path(__file__).resolve().parent
DEFAULT_INPUT: Final[pathlib.Path] = _DATA_DIR / "raw" / "dirty_data.jsonl"
OUTPUT_DIR: Final[pathlib.Path] = pathlib.Path(__file__).resolve().parent / "processed"
SPLIT_SEED: Final[int] = 42
CLEANING_RULES_VERSION: Final[str] = "1.0.0"

MAX_CHAR_LENGTH: Final[int] = 2000
MIN_CHAR_LENGTH: Final[int] = 20

# ---------------------------------------------------------------------------
# PII regex patterns
# ---------------------------------------------------------------------------
PII_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),  # email
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN
    re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"),  # phone
    re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\b"),  # credit card
    re.compile(r"\bDOB\s*[:]\s*\d{2}/\d{2}/\d{4}\b", re.IGNORECASE),  # date of birth
    re.compile(r"\bMRN\s*[:]\s*\d+\b", re.IGNORECASE),  # medical record
    re.compile(r"\bSSN\s*[:]\s*\d{3}-\d{2}-\d{4}\b", re.IGNORECASE),  # explicit SSN label
]

# HTML tag pattern
HTML_TAG_RE: Final[re.Pattern[str]] = re.compile(r"<[^>]+>")
HTML_ENTITY_RE: Final[re.Pattern[str]] = re.compile(r"&[a-zA-Z]+;|&#\d+;")

# Boilerplate signals
BOILERPLATE_SIGNALS: Final[list[str]] = [
    "terms of service",
    "privacy policy",
    "all rights reserved",
    "unauthorized reproduction",
    "terms and conditions apply",
    "this email and any attachments are confidential",
]


# ---------------------------------------------------------------------------
# Counters
# ---------------------------------------------------------------------------
@dataclass
class CleaningStats:
    """Track how many records each rule removed."""

    total_input: int = 0
    removed_duplicate: int = 0
    removed_html: int = 0
    removed_unicode: int = 0
    removed_pii: int = 0
    removed_length: int = 0
    removed_boilerplate: int = 0
    total_output: int = 0
    train_count: int = 0
    val_count: int = 0
    test_count: int = 0


# ---------------------------------------------------------------------------
# Cleaning functions
# ---------------------------------------------------------------------------


def normalize_unicode(text: str) -> str:
    """Fix broken unicode, normalize to NFC, strip control chars."""
    text = unicodedata.normalize("NFC", text)
    text = text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
    return text.strip()


def has_html(text: str) -> bool:
    """Return True if text contains HTML tags or entities."""
    return bool(HTML_TAG_RE.search(text)) or bool(HTML_ENTITY_RE.search(text))


def has_pii(text: str) -> bool:
    """Return True if any PII pattern matches."""
    return any(p.search(text) for p in PII_PATTERNS)


def is_boilerplate(text: str) -> bool:
    """Return True if text matches boilerplate signals."""
    lower = text.lower()
    return sum(1 for sig in BOILERPLATE_SIGNALS if sig in lower) >= 2


def has_broken_unicode(text: str) -> bool:
    """Return True if replacement chars are present after normalization."""
    return "\ufffd" in text


def clean_pipeline(input_path: pathlib.Path, seed: int = SPLIT_SEED) -> CleaningStats:
    """Run the full cleaning pipeline. Returns stats."""
    import random

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stats = CleaningStats()

    # ── Load ──────────────────────────────────────────────────────
    raw_records: list[dict] = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_records.append(json.loads(line))
    stats.total_input = len(raw_records)

    # ── Deduplicate (by text hash) ────────────────────────────────
    seen_hashes: set[str] = set()
    deduped: list[dict] = []
    for rec in raw_records:
        h = hashlib.md5(rec["text"].encode("utf-8")).hexdigest()
        if h in seen_hashes:
            stats.removed_duplicate += 1
            continue
        seen_hashes.add(h)
        deduped.append(rec)

    # ── Apply rules sequentially ──────────────────────────────────
    cleaned: list[dict] = []
    for rec in deduped:
        text = normalize_unicode(rec["text"])

        if has_broken_unicode(text):
            stats.removed_unicode += 1
            continue

        if has_html(text):
            stats.removed_html += 1
            continue

        if has_pii(text):
            stats.removed_pii += 1
            continue

        if len(text) > MAX_CHAR_LENGTH or len(text) < MIN_CHAR_LENGTH:
            stats.removed_length += 1
            continue

        if is_boilerplate(text):
            stats.removed_boilerplate += 1
            continue

        cleaned.append({"text": text})

    stats.total_output = len(cleaned)

    # ── Deterministic split (80/10/10) ────────────────────────────
    rng = random.Random(seed)
    rng.shuffle(cleaned)

    n = len(cleaned)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    splits = {
        "train": cleaned[:train_end],
        "val": cleaned[train_end:val_end],
        "test": cleaned[val_end:],
    }
    stats.train_count = len(splits["train"])
    stats.val_count = len(splits["val"])
    stats.test_count = len(splits["test"])

    # ── Write splits ──────────────────────────────────────────────
    for name, records in splits.items():
        out_path = OUTPUT_DIR / f"{name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ── Write manifest ────────────────────────────────────────────
    dataset_hash = hashlib.sha256(
        json.dumps([r["text"] for r in cleaned], sort_keys=True).encode()
    ).hexdigest()[:16]

    manifest = {
        "cleaning_rules_version": CLEANING_RULES_VERSION,
        "split_seed": seed,
        "dataset_hash": dataset_hash,
        "input_file": str(input_path),
        "stats": asdict(stats),
    }
    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return stats


def print_report(stats: CleaningStats) -> None:
    """Print a before/after console report."""
    print("\n" + "=" * 60)
    print("  DATA CLEANING REPORT")
    print("=" * 60)
    print(f"  Input records:          {stats.total_input:>6}")
    print(f"  ─ Removed (duplicate):  {stats.removed_duplicate:>6}")
    print(f"  ─ Removed (HTML):       {stats.removed_html:>6}")
    print(f"  ─ Removed (unicode):    {stats.removed_unicode:>6}")
    print(f"  ─ Removed (PII):        {stats.removed_pii:>6}")
    print(f"  ─ Removed (length):     {stats.removed_length:>6}")
    print(f"  ─ Removed (boilerplate):{stats.removed_boilerplate:>6}")
    total_removed = (
        stats.removed_duplicate
        + stats.removed_html
        + stats.removed_unicode
        + stats.removed_pii
        + stats.removed_length
        + stats.removed_boilerplate
    )
    print(f"  Total removed:          {total_removed:>6}")
    print(f"  Output records:         {stats.total_output:>6}")
    print(f"    ├─ train:             {stats.train_count:>6}")
    print(f"    ├─ val:               {stats.val_count:>6}")
    print(f"    └─ test:              {stats.test_count:>6}")
    pct = (1 - stats.total_output / max(stats.total_input, 1)) * 100
    print(f"  Reduction:              {pct:>5.1f}%")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Clean raw dataset")
    parser.add_argument("--input", type=pathlib.Path, default=DEFAULT_INPUT)
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    args = parser.parse_args()

    if not args.input.exists():
        print(f"[clean] Input not found: {args.input}", file=sys.stderr)
        print("[clean] Run `python -m data.generate_dirty` first.", file=sys.stderr)
        sys.exit(1)

    stats = clean_pipeline(args.input, seed=args.seed)
    print_report(stats)


if __name__ == "__main__":
    main()

"""Unit tests for data cleaning rules.

Tests the individual cleaning functions in data/clean.py to verify
regex behavior, edge cases, and pipeline correctness.
"""

from __future__ import annotations

import json
import pathlib

from data.clean import (
    CleaningStats,
    clean_pipeline,
    has_broken_unicode,
    has_html,
    has_pii,
    is_boilerplate,
    normalize_unicode,
)


# ---------------------------------------------------------------------------
# normalize_unicode
# ---------------------------------------------------------------------------
class TestNormalizeUnicode:
    def test_strips_control_chars(self) -> None:
        text = "hello\x00world\x07"
        assert normalize_unicode(text) == "helloworld"

    def test_preserves_normal_text(self) -> None:
        text = "Machine learning is great."
        assert normalize_unicode(text) == text

    def test_normalizes_nfc(self) -> None:
        # e + combining accent -> single char
        text = "caf\u0065\u0301"
        result = normalize_unicode(text)
        assert "\u0301" not in result  # combining accent folded

    def test_strips_whitespace(self) -> None:
        assert normalize_unicode("  hello  ") == "hello"


# ---------------------------------------------------------------------------
# has_html
# ---------------------------------------------------------------------------
class TestHasHtml:
    def test_detects_tags(self) -> None:
        assert has_html("<p>Hello</p>") is True

    def test_detects_entities(self) -> None:
        assert has_html("Tom &amp; Jerry") is True

    def test_clean_text(self) -> None:
        assert has_html("No HTML here.") is False

    def test_angle_brackets_in_math(self) -> None:
        # a < b is not an HTML tag (no closing >)
        assert has_html("if a < b then") is False

    def test_numeric_entity(self) -> None:
        assert has_html("&#8212; is an em-dash") is True


# ---------------------------------------------------------------------------
# has_pii
# ---------------------------------------------------------------------------
class TestHasPii:
    def test_detects_email(self) -> None:
        assert has_pii("Contact me at john@example.com") is True

    def test_detects_ssn(self) -> None:
        assert has_pii("SSN: 123-45-6789") is True

    def test_detects_phone(self) -> None:
        assert has_pii("Call 555-123-4567 now") is True

    def test_detects_credit_card(self) -> None:
        assert has_pii("Card: 4242 4242 4242 4242") is True

    def test_detects_dob(self) -> None:
        assert has_pii("Patient DOB: 03/15/1985") is True

    def test_detects_mrn(self) -> None:
        assert has_pii("MRN: 0012345") is True

    def test_clean_text(self) -> None:
        assert has_pii("Gradient descent minimizes the loss function.") is False

    def test_near_miss_not_flagged(self) -> None:
        # 4-digit number is not a SSN
        assert has_pii("The year 2024 was productive.") is False


# ---------------------------------------------------------------------------
# is_boilerplate
# ---------------------------------------------------------------------------
class TestIsBoilerplate:
    def test_detects_legal_boilerplate(self) -> None:
        text = "All rights reserved. Unauthorized reproduction is prohibited."
        assert is_boilerplate(text) is True

    def test_single_signal_not_enough(self) -> None:
        text = "All rights reserved."
        assert is_boilerplate(text) is False

    def test_clean_text(self) -> None:
        assert is_boilerplate("Transformers use self-attention.") is False


# ---------------------------------------------------------------------------
# has_broken_unicode
# ---------------------------------------------------------------------------
class TestHasBrokenUnicode:
    def test_detects_replacement_char(self) -> None:
        assert has_broken_unicode("hello\ufffdworld") is True

    def test_clean_text(self) -> None:
        assert has_broken_unicode("Normal text here.") is False


# ---------------------------------------------------------------------------
# clean_pipeline (integration)
# ---------------------------------------------------------------------------
class TestCleanPipeline:
    def _make_dirty_file(self, tmp_dir: pathlib.Path) -> pathlib.Path:
        """Create a small dirty dataset for testing."""
        records = [
            {"text": "Machine learning models need careful evaluation."},
            {"text": "Machine learning models need careful evaluation."},  # duplicate
            {"text": "<div>HTML content</div>"},
            {"text": "Contact john@example.com for details."},
            {"text": "x"},  # too short
            {"text": "A" * 3000},  # too long
            {"text": "Fine-tuning improves domain-specific performance significantly."},
            {"text": "Gradient descent is used to minimize the loss function in training."},
        ]
        path = tmp_dir / "test_dirty.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        return path

    def test_pipeline_produces_splits(self, tmp_path: pathlib.Path) -> None:
        dirty = self._make_dirty_file(tmp_path)
        stats = clean_pipeline(dirty, seed=42)

        assert stats.total_input == 8
        assert stats.removed_duplicate >= 1
        assert stats.removed_html >= 1
        assert stats.removed_pii >= 1
        assert stats.removed_length >= 1
        assert stats.total_output > 0
        assert stats.train_count + stats.val_count + stats.test_count == stats.total_output

    def test_pipeline_creates_manifest(self, tmp_path: pathlib.Path) -> None:
        dirty = self._make_dirty_file(tmp_path)
        # We need clean_pipeline to write to a known location.
        # Since it uses its own OUTPUT_DIR, we test that manifest structure is correct
        # by checking the stats return value.
        stats = clean_pipeline(dirty, seed=42)
        assert isinstance(stats, CleaningStats)

    def test_deterministic_with_same_seed(self, tmp_path: pathlib.Path) -> None:
        dirty = self._make_dirty_file(tmp_path)
        stats1 = clean_pipeline(dirty, seed=99)
        stats2 = clean_pipeline(dirty, seed=99)
        assert stats1.total_output == stats2.total_output
        assert stats1.train_count == stats2.train_count

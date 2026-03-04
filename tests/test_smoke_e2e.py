"""
End-to-end smoke tests for the full CEFR experiment pipeline.

Exercises Exp 0 (majority), Exp 1 (TF-IDF+LR), and Exp 5 (hybrid essay)
using only synthetic in-memory data.

Requirements: scikit-learn, numpy  (no torch / transformers / network needed).
"""

import os
import tempfile

import numpy as np
import pytest

from src.config import CEFR_LEVELS, ID2LABEL, LABEL2ID
from src.data_utils import load_jsonl, save_jsonl, stratified_split


# ---------------------------------------------------------------------------
# Shared fixture: 180 synthetic samples (30 × 6 CEFR classes)
# ---------------------------------------------------------------------------

def _make_synthetic_data(n_per_class: int = 30):
    """
    Return (texts, labels) with *n_per_class* samples per CEFR level.
    Texts are short but distinct enough for TF-IDF to separate classes.
    """
    texts, labels = [], []
    for level in CEFR_LEVELS:
        for j in range(n_per_class):
            text = (
                f"This is a synthetic {level} proficiency sample number {j}. "
                f"The level marker is {level} and the index is {j}."
            )
            texts.append(text)
            labels.append(LABEL2ID[level])
    return texts, labels


@pytest.fixture(scope="module")
def splits():
    """Stratified 80/10/10 split of the synthetic dataset."""
    texts, labels = _make_synthetic_data(30)
    return stratified_split(texts, labels, seed=42)


# ---------------------------------------------------------------------------
# Exp 0 – Majority baseline
# ---------------------------------------------------------------------------

class TestEndToEndExp0:
    def test_returns_valid_metrics(self, splits):
        (tr_t, tr_l), (va_t, va_l), (te_t, te_l) = splits
        from src.run_experiments import run_exp0

        r = run_exp0(tr_l, te_l, len(te_t), track="sentence")
        assert 0.0 <= r.accuracy <= 1.0
        assert 0.0 <= r.macro_f1 <= 1.0
        assert -1.0 <= r.qwk <= 1.0
        assert r.note.startswith("majority=")

    def test_majority_level_is_valid_cefr(self, splits):
        (tr_t, tr_l), _, (te_t, te_l) = splits
        from src.run_experiments import run_exp0

        r = run_exp0(tr_l, te_l, len(te_t), track="essay")
        majority_level = r.note.replace("majority=", "")
        assert majority_level in CEFR_LEVELS


# ---------------------------------------------------------------------------
# Exp 1 – TF-IDF + Logistic Regression
# ---------------------------------------------------------------------------

class TestEndToEndExp1:
    def test_returns_valid_metrics(self, splits):
        (tr_t, tr_l), _, (te_t, te_l) = splits
        from src.run_experiments import run_exp1

        r = run_exp1(tr_t, tr_l, te_t, te_l, track="sentence")
        assert 0.0 <= r.accuracy <= 1.0
        assert r.name == "Exp 1 – TF-IDF+LR"

    def test_beats_majority_on_synthetic_data(self, splits):
        """TF-IDF+LR should match or outperform the majority baseline on distinct synthetic texts."""
        (tr_t, tr_l), _, (te_t, te_l) = splits
        from src.run_experiments import run_exp0, run_exp1

        r0 = run_exp0(tr_l, te_l, len(te_t), track="sentence")
        r1 = run_exp1(tr_t, tr_l, te_t, te_l, track="sentence")
        assert r1.accuracy >= r0.accuracy

    def test_latency_is_positive(self, splits):
        (tr_t, tr_l), _, (te_t, te_l) = splits
        from src.run_experiments import run_exp1

        r = run_exp1(tr_t, tr_l, te_t, te_l, track="sentence")
        assert r.latency >= 0.0


# ---------------------------------------------------------------------------
# Exp 5 – Hybrid essay (sentence classifier + aggregation)
# ---------------------------------------------------------------------------

class TestEndToEndExp5:
    def test_majority_vote(self, splits):
        (tr_t, tr_l), _, (te_t, te_l) = splits
        from src.run_experiments import run_exp5

        r = run_exp5(tr_t, tr_l, te_t, te_l, track="essay", aggregation="majority_vote")
        assert 0.0 <= r.accuracy <= 1.0
        assert "Exp 5" in r.name
        assert "majority_vote" in r.name

    def test_mean_prob(self, splits):
        (tr_t, tr_l), _, (te_t, te_l) = splits
        from src.run_experiments import run_exp5

        r = run_exp5(tr_t, tr_l, te_t, te_l, track="essay", aggregation="mean_prob")
        assert 0.0 <= r.accuracy <= 1.0

    def test_weighted_vote(self, splits):
        (tr_t, tr_l), _, (te_t, te_l) = splits
        from src.run_experiments import run_exp5

        r = run_exp5(tr_t, tr_l, te_t, te_l, track="essay", aggregation="weighted_vote")
        assert 0.0 <= r.accuracy <= 1.0


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

class TestComparisonTable:
    def test_full_cpu_pipeline_prints_table(self, splits, capsys):
        """Run Exp 0, 1, 5 and verify the comparison table is printed."""
        (tr_t, tr_l), _, (te_t, te_l) = splits
        from src.run_experiments import (
            print_comparison_table,
            run_exp0,
            run_exp1,
            run_exp5,
        )

        results = [
            run_exp0(tr_l, te_l, len(te_t), track="sentence"),
            run_exp1(tr_t, tr_l, te_t, te_l, track="sentence"),
            run_exp5(tr_t, tr_l, te_t, te_l, track="essay", aggregation="majority_vote"),
        ]
        print_comparison_table(results)
        out = capsys.readouterr().out
        assert "Exp 0" in out
        assert "Exp 1" in out
        assert "Exp 5" in out
        assert "Acc" in out
        assert "F1" in out
        assert "QWK" in out

    def test_all_results_have_valid_metrics(self, splits):
        (tr_t, tr_l), _, (te_t, te_l) = splits
        from src.run_experiments import run_exp0, run_exp1

        results = [
            run_exp0(tr_l, te_l, len(te_t), track="sentence"),
            run_exp1(tr_t, tr_l, te_t, te_l, track="sentence"),
        ]
        for r in results:
            assert 0.0 <= r.accuracy <= 1.0
            assert 0.0 <= r.macro_f1 <= 1.0
            assert -1.0 <= r.qwk <= 1.0


# ---------------------------------------------------------------------------
# JSONL data-dir flow (used by --data_dir in run_experiments)
# ---------------------------------------------------------------------------

class TestJsonlDataFlow:
    """Tests the prepare_data → run_experiments JSONL data-dir flow."""

    def _write_jsonl_splits(self, tmpdir, track, splits):
        (tr_t, tr_l), (va_t, va_l), (te_t, te_l) = splits
        for split_name, (texts, labels) in [
            ("train", (tr_t, tr_l)),
            ("dev", (va_t, va_l)),
            ("test", (te_t, te_l)),
        ]:
            records = [
                {"text": t, "label": ID2LABEL[l], "n_tokens": len(t.split())}
                for t, l in zip(texts, labels)
            ]
            save_jsonl(records, os.path.join(tmpdir, track, f"{split_name}.jsonl"))

    def test_load_splits_from_jsonl(self, splits):
        from src.run_experiments import _load_splits_from_jsonl

        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_jsonl_splits(tmpdir, "sentence", splits)
            (tr_t, tr_l), (va_t, va_l), (te_t, te_l) = _load_splits_from_jsonl(
                tmpdir, "sentence"
            )

        (orig_tr_t, orig_tr_l), (orig_va_t, orig_va_l), (orig_te_t, orig_te_l) = splits
        assert len(tr_t) == len(orig_tr_t)
        assert len(va_t) == len(orig_va_t)
        assert len(te_t) == len(orig_te_t)
        assert tr_l == orig_tr_l
        assert all(l in range(6) for l in tr_l)

    def test_missing_jsonl_raises_file_not_found(self, splits):
        from src.run_experiments import _load_splits_from_jsonl

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError, match="train.jsonl"):
                _load_splits_from_jsonl(tmpdir, "sentence")

    def test_exp0_exp1_via_jsonl(self, splits):
        """Full pipeline: write JSONL → load via _load_splits_from_jsonl → run Exp 0, 1."""
        from src.run_experiments import _load_splits_from_jsonl, run_exp0, run_exp1

        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_jsonl_splits(tmpdir, "sentence", splits)
            (tr_t, tr_l), (va_t, va_l), (te_t, te_l) = _load_splits_from_jsonl(
                tmpdir, "sentence"
            )

        r0 = run_exp0(tr_l, te_l, len(te_t), track="sentence")
        r1 = run_exp1(tr_t, tr_l, te_t, te_l, track="sentence")

        assert 0.0 <= r0.accuracy <= 1.0
        assert 0.0 <= r1.accuracy <= 1.0

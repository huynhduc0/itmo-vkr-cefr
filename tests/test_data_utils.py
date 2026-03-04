"""
Unit tests for data utilities.
"""

import pytest

from src.config import CEFR_LEVELS, LABEL2ID
from src.data_utils import (
    get_label_distribution,
    normalize_label,
    remove_duplicates,
    set_seed,
    stratified_split,
)


class TestNormalizeLabel:
    def test_valid_labels(self):
        for level in CEFR_LEVELS:
            assert normalize_label(level) == level

    def test_lowercase(self):
        assert normalize_label("a1") == "A1"
        assert normalize_label("b2") == "B2"
        assert normalize_label("c1") == "C1"

    def test_whitespace_stripped(self):
        assert normalize_label("  B1  ") == "B1"

    def test_invalid_label(self):
        assert normalize_label("D1") is None
        assert normalize_label("unknown") is None
        assert normalize_label("") is None

    def test_none_input(self):
        assert normalize_label(None) is None


class TestStratifiedSplit:
    @pytest.fixture
    def sample_data(self):
        texts = [f"sample text {i}" for i in range(120)]
        labels = [i % 6 for i in range(120)]
        return texts, labels

    def test_split_sizes(self, sample_data):
        texts, labels = sample_data
        (tr_t, tr_l), (va_t, va_l), (te_t, te_l) = stratified_split(
            texts, labels, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
        )
        total = len(tr_t) + len(va_t) + len(te_t)
        assert total == len(texts)

    def test_split_no_overlap(self, sample_data):
        texts, labels = sample_data
        (tr_t, _), (va_t, _), (te_t, _) = stratified_split(texts, labels)
        tr_set = set(tr_t)
        va_set = set(va_t)
        te_set = set(te_t)
        assert len(tr_set & va_set) == 0
        assert len(tr_set & te_set) == 0
        assert len(va_set & te_set) == 0

    def test_split_stratification(self, sample_data):
        texts, labels = sample_data
        (tr_t, tr_l), (va_t, va_l), (te_t, te_l) = stratified_split(texts, labels)
        for label_id in range(6):
            assert label_id in tr_l
            assert label_id in va_l
            assert label_id in te_l

    def test_reproducibility(self, sample_data):
        texts, labels = sample_data
        result1 = stratified_split(texts, labels, seed=42)
        result2 = stratified_split(texts, labels, seed=42)
        assert result1[0][0] == result2[0][0]
        assert result1[1][0] == result2[1][0]

    def test_invalid_ratios(self, sample_data):
        texts, labels = sample_data
        with pytest.raises(AssertionError):
            stratified_split(texts, labels, train_ratio=0.9, val_ratio=0.1, test_ratio=0.1)


class TestGetLabelDistribution:
    def test_counts(self):
        labels = [0, 1, 2, 0, 1, 0]
        dist = get_label_distribution(labels)
        assert dist["A1"] == 3
        assert dist["A2"] == 2
        assert dist["B1"] == 1
        assert dist["B2"] == 0
        assert dist["C1"] == 0
        assert dist["C2"] == 0

    def test_all_levels_present(self):
        labels = [0]
        dist = get_label_distribution(labels)
        for level in CEFR_LEVELS:
            assert level in dist


class TestSetSeed:
    def test_set_seed_runs(self):
        set_seed(42)
        set_seed(0)


class TestRemoveDuplicates:
    def test_removes_exact_duplicates(self):
        texts = ["hello", "world", "hello"]
        labels = [0, 1, 0]
        out_t, out_l = remove_duplicates(texts, labels)
        assert out_t == ["hello", "world"]
        assert out_l == [0, 1]

    def test_same_text_different_label_kept(self):
        texts = ["hello", "hello"]
        labels = [0, 1]
        out_t, out_l = remove_duplicates(texts, labels)
        assert len(out_t) == 2

    def test_preserves_order(self):
        texts = ["c", "a", "b", "a"]
        labels = [2, 0, 1, 0]
        out_t, out_l = remove_duplicates(texts, labels)
        assert out_t == ["c", "a", "b"]
        assert out_l == [2, 0, 1]

    def test_no_duplicates_unchanged(self):
        texts = ["x", "y", "z"]
        labels = [0, 1, 2]
        out_t, out_l = remove_duplicates(texts, labels)
        assert out_t == texts
        assert out_l == labels

    def test_empty_input(self):
        out_t, out_l = remove_duplicates([], [])
        assert out_t == []
        assert out_l == []

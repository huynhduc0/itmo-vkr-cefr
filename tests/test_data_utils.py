"""
Unit tests for data utilities.
"""

import json
import os
import tempfile

import pytest

from src.config import CEFR_LEVELS, LABEL2ID
from src.data_utils import (
    build_token_counts,
    count_tokens,
    filter_by_length,
    filter_min_class_size,
    get_label_distribution,
    load_jsonl,
    normalize_label,
    normalize_text,
    remove_duplicates,
    save_jsonl,
    set_seed,
    stratified_split,
)


# ---------------------------------------------------------------------------
# Mock tokenizer for tests that count tokens without requiring `transformers`
# ---------------------------------------------------------------------------

class _MockTokenizer:
    """Splits on whitespace and adds 2 for [CLS]/[SEP] — no model download."""

    def __call__(self, text, **kwargs):
        tokens = text.split()
        return {"input_ids": list(range(len(tokens) + 2))}


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


# ---------------------------------------------------------------------------
# normalize_text
# ---------------------------------------------------------------------------

class TestNormalizeText:
    def test_strips_leading_trailing_whitespace(self):
        assert normalize_text("  hello world  ") == "hello world"

    def test_collapses_internal_whitespace(self):
        assert normalize_text("hello   world") == "hello world"

    def test_collapses_tabs_and_newlines(self):
        assert normalize_text("hello\t\nworld") == "hello world"

    def test_preserves_casing(self):
        assert normalize_text("Hello World") == "Hello World"

    def test_empty_string(self):
        assert normalize_text("") == ""

    def test_only_whitespace(self):
        assert normalize_text("   ") == ""

    def test_no_change_needed(self):
        text = "This is a clean sentence."
        assert normalize_text(text) == text


# ---------------------------------------------------------------------------
# count_tokens / build_token_counts
# ---------------------------------------------------------------------------

class TestCountTokens:
    def test_basic(self):
        tok = _MockTokenizer()
        # "hello world" -> 2 word tokens + 2 special = 4
        assert count_tokens("hello world", tok) == 4

    def test_single_word(self):
        tok = _MockTokenizer()
        assert count_tokens("hello", tok) == 3  # 1 word + 2 special

    def test_empty_string(self):
        tok = _MockTokenizer()
        # 0 words + 2 special tokens
        assert count_tokens("", tok) == 2

    def test_build_token_counts_length(self):
        tok = _MockTokenizer()
        texts = ["hello world", "this is a test", "ok"]
        counts = build_token_counts(texts, tok)
        assert len(counts) == len(texts)

    def test_build_token_counts_values(self):
        tok = _MockTokenizer()
        texts = ["one", "one two", "one two three"]
        counts = build_token_counts(texts, tok)
        assert counts == [3, 4, 5]


# ---------------------------------------------------------------------------
# filter_by_length
# ---------------------------------------------------------------------------

class TestFilterByLength:
    @pytest.fixture
    def sample(self):
        texts = ["a", "b", "c", "d", "e"]
        labels = [0, 1, 2, 3, 4]
        n_tokens = [5, 10, 20, 64, 128]
        return texts, labels, n_tokens

    def test_min_only(self, sample):
        texts, labels, n_tokens = sample
        ft, fl, fn = filter_by_length(texts, labels, n_tokens, min_tokens=10)
        assert 5 not in fn
        assert all(n >= 10 for n in fn)

    def test_max_only(self, sample):
        texts, labels, n_tokens = sample
        ft, fl, fn = filter_by_length(texts, labels, n_tokens, max_tokens=20)
        assert all(n <= 20 for n in fn)

    def test_min_and_max(self, sample):
        texts, labels, n_tokens = sample
        ft, fl, fn = filter_by_length(texts, labels, n_tokens, min_tokens=10, max_tokens=64)
        assert all(10 <= n <= 64 for n in fn)
        assert len(ft) == 3  # n=10, 20, 64

    def test_no_bounds(self, sample):
        texts, labels, n_tokens = sample
        ft, fl, fn = filter_by_length(texts, labels, n_tokens)
        assert len(ft) == len(texts)

    def test_empty_result(self, sample):
        texts, labels, n_tokens = sample
        ft, fl, fn = filter_by_length(texts, labels, n_tokens, min_tokens=999)
        assert ft == []
        assert fl == []
        assert fn == []

    def test_parallel_lists_consistent(self, sample):
        texts, labels, n_tokens = sample
        ft, fl, fn = filter_by_length(texts, labels, n_tokens, min_tokens=10, max_tokens=64)
        assert len(ft) == len(fl) == len(fn)


# ---------------------------------------------------------------------------
# filter_min_class_size
# ---------------------------------------------------------------------------

class TestFilterMinClassSize:
    def test_drops_small_class(self):
        # class 0 has 1 sample, class 1 has 5 samples; min=2 → drop class 0
        texts = ["a"] + ["b"] * 5
        labels = [0] + [1] * 5
        n_tokens = [10] * 6
        ft, fl, fn = filter_min_class_size(texts, labels, n_tokens, min_samples=2)
        assert 0 not in fl
        assert all(l == 1 for l in fl)

    def test_keeps_all_large_enough(self):
        texts = ["a"] * 3 + ["b"] * 3
        labels = [0] * 3 + [1] * 3
        n_tokens = [10] * 6
        ft, fl, fn = filter_min_class_size(texts, labels, n_tokens, min_samples=3)
        assert len(ft) == 6

    def test_empty_result_when_all_too_small(self):
        texts = ["a", "b"]
        labels = [0, 1]
        n_tokens = [10, 10]
        ft, fl, fn = filter_min_class_size(texts, labels, n_tokens, min_samples=5)
        assert ft == []
        assert fl == []
        assert fn == []

    def test_parallel_lists_consistent(self):
        texts = ["a"] * 10 + ["b"] * 10
        labels = [0] * 10 + [1] * 10
        n_tokens = list(range(20))
        ft, fl, fn = filter_min_class_size(texts, labels, n_tokens, min_samples=5)
        assert len(ft) == len(fl) == len(fn)


# ---------------------------------------------------------------------------
# save_jsonl / load_jsonl
# ---------------------------------------------------------------------------

class TestJsonl:
    def test_roundtrip(self):
        records = [
            {"text": "Hello world.", "label": "B1", "n_tokens": 5},
            {"text": "Another sentence.", "label": "A2", "n_tokens": 3},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.jsonl")
            save_jsonl(records, path)
            loaded = load_jsonl(path)
        assert loaded == records

    def test_unicode_preserved(self):
        records = [{"text": "Héllo wörld", "label": "C1", "n_tokens": 4}]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "unicode.jsonl")
            save_jsonl(records, path)
            loaded = load_jsonl(path)
        assert loaded[0]["text"] == "Héllo wörld"

    def test_creates_parent_dirs(self):
        records = [{"x": 1}]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "nested", "deep", "out.jsonl")
            save_jsonl(records, path)
            assert os.path.exists(path)

    def test_empty_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "empty.jsonl")
            save_jsonl([], path)
            loaded = load_jsonl(path)
        assert loaded == []


# ---------------------------------------------------------------------------
# Language presets (config.py)
# ---------------------------------------------------------------------------

class TestLanguagePresets:
    def test_english_preset_exists(self):
        from src.config import LANGUAGE_PRESETS
        assert "en" in LANGUAGE_PRESETS

    def test_russian_preset_exists(self):
        from src.config import LANGUAGE_PRESETS
        assert "ru" in LANGUAGE_PRESETS

    def test_english_preset_uses_roberta(self):
        from src.config import LANGUAGE_PRESETS
        assert LANGUAGE_PRESETS["en"]["tokenizer"] == "roberta-base"

    def test_russian_preset_uses_xlm_roberta(self):
        from src.config import LANGUAGE_PRESETS
        assert LANGUAGE_PRESETS["ru"]["tokenizer"] == "xlm-roberta-base"

    def test_presets_have_required_keys(self):
        from src.config import LANGUAGE_PRESETS
        required = {"dataset_name", "tokenizer", "text_column", "label_column"}
        for lang, preset in LANGUAGE_PRESETS.items():
            assert required == set(preset.keys()), f"Preset '{lang}' is missing keys"


# ---------------------------------------------------------------------------
# load_dataset error handling
# ---------------------------------------------------------------------------

class TestLoadDatasetErrorHandling:
    def test_dataset_not_found_raises_value_error(self, monkeypatch):
        """DatasetNotFoundError should be re-raised as a descriptive ValueError."""
        import sys
        from unittest.mock import MagicMock

        class _FakeDatasetNotFoundError(Exception):
            pass

        mock_ds_module = MagicMock()
        mock_ds_module.load_dataset.side_effect = _FakeDatasetNotFoundError(
            "Dataset 'bad/dataset' doesn't exist on the Hub or cannot be accessed."
        )
        monkeypatch.setitem(sys.modules, "datasets", mock_ds_module)

        from src.data_utils import load_dataset

        with pytest.raises(ValueError, match="not found on the HuggingFace Hub"):
            load_dataset("bad/dataset")

    def test_dataset_not_found_error_message_mentions_dataset_name(self, monkeypatch):
        import sys
        from unittest.mock import MagicMock

        class _FakeDatasetNotFoundError(Exception):
            pass

        mock_ds_module = MagicMock()
        mock_ds_module.load_dataset.side_effect = _FakeDatasetNotFoundError(
            "Dataset 'UniversalCEFR/cefr_sp_ru' doesn't exist"
        )
        monkeypatch.setitem(sys.modules, "datasets", mock_ds_module)

        from src.data_utils import load_dataset

        with pytest.raises(ValueError, match="UniversalCEFR/cefr_sp_ru"):
            load_dataset("UniversalCEFR/cefr_sp_ru")

    def test_dataset_not_found_error_message_suggests_dataset_override(self, monkeypatch):
        import sys
        from unittest.mock import MagicMock

        class _FakeDatasetNotFoundError(Exception):
            pass

        mock_ds_module = MagicMock()
        mock_ds_module.load_dataset.side_effect = _FakeDatasetNotFoundError(
            "Dataset 'UniversalCEFR/cefr_sp_ru' doesn't exist"
        )
        monkeypatch.setitem(sys.modules, "datasets", mock_ds_module)

        from src.data_utils import load_dataset

        with pytest.raises(ValueError, match="--dataset"):
            load_dataset("UniversalCEFR/cefr_sp_ru")

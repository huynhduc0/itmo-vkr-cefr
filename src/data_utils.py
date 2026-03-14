"""
Data loading and preprocessing utilities for CEFR classification.
"""

import json
import os
import random
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.config import (
    CEFR_LEVELS,
    DATA_PREP_CONFIG,
    DATASET_CONFIG,
    ID2LABEL,
    LABEL2ID,
    RANDOM_SEED,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def normalize_label(label: str) -> Optional[str]:
    """Normalize a raw label string to a valid CEFR level or None."""
    if label is None:
        return None
    label = str(label).strip().upper()
    if label in LABEL2ID:
        return label
    return None


def load_dataset(
    dataset_name: str = DATASET_CONFIG["dataset_name"],
    text_column: str = DATASET_CONFIG["text_column"],
    label_column: str = DATASET_CONFIG["label_column"],
    split: str = "train",
) -> Tuple[List[str], List[int]]:
    """
    Load and preprocess a HuggingFace dataset for CEFR classification.

    Returns:
        texts: list of text strings
        labels: list of integer label ids

    Raises:
        ValueError: if the dataset is not found on the HuggingFace Hub.
    """
    from datasets import load_dataset as hf_load_dataset

    try:
        dataset = hf_load_dataset(dataset_name, split=split)
    except Exception as exc:
        # Provide a helpful message when the dataset does not exist or is
        # not accessible, e.g. when using an unsupported language preset.
        exc_name = type(exc).__name__
        if "DatasetNotFoundError" in exc_name or "not found" in str(exc).lower():
            raise ValueError(
                f"Dataset '{dataset_name}' was not found on the HuggingFace Hub "
                f"or cannot be accessed.\n"
                f"  • Check that the dataset path is correct.\n"
                f"  • If the dataset is private, ensure HF_TOKEN is set.\n"
                f"  • If using a language preset whose default dataset does not "
                f"yet exist, supply a valid dataset path via --dataset <hf_path>.\n"
                f"Original error: {exc}"
            ) from exc
        raise

    texts, labels = [], []
    for sample in dataset:
        text = sample.get(text_column, "")
        raw_label = sample.get(label_column, None)
        if not text or not text.strip():
            continue
        label = normalize_label(raw_label)
        if label is None:
            continue
        texts.append(normalize_text(text))
        labels.append(LABEL2ID[label])
    return texts, labels


def stratified_split(
    texts: List[str],
    labels: List[int],
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    seed: int = RANDOM_SEED,
) -> Tuple[
    Tuple[List[str], List[int]],
    Tuple[List[str], List[int]],
    Tuple[List[str], List[int]],
]:
    """
    Perform stratified train/validation/test split.

    Returns:
        (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels)
    """
    from sklearn.model_selection import train_test_split

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
        "Ratios must sum to 1.0"
    )

    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts,
        labels,
        test_size=(val_ratio + test_ratio),
        stratify=labels,
        random_state=seed,
    )

    val_fraction = val_ratio / (val_ratio + test_ratio)
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts,
        temp_labels,
        test_size=(1.0 - val_fraction),
        stratify=temp_labels,
        random_state=seed,
    )

    return (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels)


def load_and_split_dataset(
    dataset_name: str = DATASET_CONFIG["dataset_name"],
    text_column: str = DATASET_CONFIG["text_column"],
    label_column: str = DATASET_CONFIG["label_column"],
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    seed: int = RANDOM_SEED,
    deduplicate: bool = True,
) -> Tuple[
    Tuple[List[str], List[int]],
    Tuple[List[str], List[int]],
    Tuple[List[str], List[int]],
]:
    """
    Load dataset and perform stratified split in one step.

    Args:
        deduplicate: if True, remove duplicate (text, label) pairs before splitting.
    """
    texts, labels = load_dataset(
        dataset_name=dataset_name,
        text_column=text_column,
        label_column=label_column,
    )
    if deduplicate:
        texts, labels = remove_duplicates(texts, labels)
    return stratified_split(
        texts,
        labels,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )


def get_label_distribution(labels: List[int]) -> Dict[str, int]:
    """Return a dict mapping CEFR level names to sample counts."""
    dist: Dict[str, int] = {level: 0 for level in CEFR_LEVELS}
    for label_id in labels:
        dist[ID2LABEL[label_id]] += 1
    return dist


def remove_duplicates(
    texts: List[str],
    labels: List[int],
) -> Tuple[List[str], List[int]]:
    """
    Remove duplicate (text, label) pairs while preserving order.

    Returns:
        Deduplicated (texts, labels) lists.
    """
    seen = set()
    out_texts: List[str] = []
    out_labels: List[int] = []
    for text, label in zip(texts, labels):
        key = (text, label)
        if key not in seen:
            seen.add(key)
            out_texts.append(text)
            out_labels.append(label)
    return out_texts, out_labels


def load_multiple_datasets(
    dataset_names: List[str],
    text_column: str = DATASET_CONFIG["text_column"],
    label_column: str = DATASET_CONFIG["label_column"],
) -> Tuple[List[str], List[int]]:
    """
    Load and concatenate multiple HuggingFace datasets.

    Useful for domain transfer experiments where data comes from
    different UniversalCEFR subcorpora.

    Returns:
        Combined (texts, labels) across all datasets.
    """
    all_texts: List[str] = []
    all_labels: List[int] = []
    for name in dataset_names:
        texts, labels = load_dataset(
            dataset_name=name,
            text_column=text_column,
            label_column=label_column,
        )
        all_texts.extend(texts)
        all_labels.extend(labels)
    return all_texts, all_labels


# ---------------------------------------------------------------------------
# Text normalisation (DATA_PREP.md §4)
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """
    Normalise text for CEFR classification.

    Operations applied:
    * Strip leading/trailing whitespace.
    * Collapse any internal sequence of whitespace characters to a single space.
    * Original casing is preserved (no lowercasing).

    Args:
        text: raw input string

    Returns:
        Normalised string.
    """
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


# ---------------------------------------------------------------------------
# Token counting (DATA_PREP.md §5)
# ---------------------------------------------------------------------------

def count_tokens(text: str, tokenizer) -> int:
    """
    Count the number of tokens produced by *tokenizer* for *text*.

    Args:
        text: input string
        tokenizer: any callable that accepts a string and returns a dict with
                   an ``"input_ids"`` key (e.g. a HuggingFace fast tokenizer).

    Returns:
        Integer token count including special tokens ([CLS], [SEP], etc.).
    """
    return len(tokenizer(text)["input_ids"])


def build_token_counts(
    texts: List[str],
    tokenizer,
) -> List[int]:
    """
    Compute token counts for every text in *texts*.

    Args:
        texts: list of input strings
        tokenizer: tokenizer callable (see :func:`count_tokens`)

    Returns:
        List of integer token counts, one per input text.
    """
    return [count_tokens(t, tokenizer) for t in texts]


# ---------------------------------------------------------------------------
# Length-based filtering (DATA_PREP.md §5)
# ---------------------------------------------------------------------------

def filter_by_length(
    texts: List[str],
    labels: List[int],
    n_tokens_list: List[int],
    min_tokens: Optional[int] = None,
    max_tokens: Optional[int] = None,
) -> Tuple[List[str], List[int], List[int]]:
    """
    Retain only samples whose token count falls within [min_tokens, max_tokens].

    Args:
        texts: input text strings
        labels: corresponding integer label ids
        n_tokens_list: pre-computed token counts per text
        min_tokens: inclusive lower bound (None = no lower bound)
        max_tokens: inclusive upper bound (None = no upper bound)

    Returns:
        Filtered (texts, labels, n_tokens_list) triple.
    """
    out_t, out_l, out_n = [], [], []
    for text, label, n in zip(texts, labels, n_tokens_list):
        if min_tokens is not None and n < min_tokens:
            continue
        if max_tokens is not None and n > max_tokens:
            continue
        out_t.append(text)
        out_l.append(label)
        out_n.append(n)
    return out_t, out_l, out_n


# ---------------------------------------------------------------------------
# Minimum class-size filter (DATA_PREP.md §8)
# ---------------------------------------------------------------------------

def filter_min_class_size(
    texts: List[str],
    labels: List[int],
    n_tokens_list: List[int],
    min_samples: int = DATA_PREP_CONFIG["min_class_samples"],
) -> Tuple[List[str], List[int], List[int]]:
    """
    Drop CEFR classes that have fewer than *min_samples* samples.

    The filter is applied independently per track (sentence / essay).

    Args:
        texts: input text strings
        labels: corresponding integer label ids
        n_tokens_list: pre-computed token counts per text
        min_samples: minimum number of samples required to keep a class

    Returns:
        Filtered (texts, labels, n_tokens_list) triple.
    """
    counts = Counter(labels)
    keep = {label for label, count in counts.items() if count >= min_samples}
    out_t, out_l, out_n = [], [], []
    for text, label, n in zip(texts, labels, n_tokens_list):
        if label in keep:
            out_t.append(text)
            out_l.append(label)
            out_n.append(n)
    return out_t, out_l, out_n


# ---------------------------------------------------------------------------
# JSONL I/O (DATA_PREP.md §9)
# ---------------------------------------------------------------------------

def save_jsonl(data: List[Dict[str, Any]], path: str) -> None:
    """
    Save a list of dicts to a JSONL file (one JSON object per line).

    Parent directories are created if they do not exist.

    Args:
        data: list of serialisable dicts
        path: output file path
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for item in data:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    Load a JSONL file as a list of dicts.

    Args:
        path: input file path

    Returns:
        List of parsed dicts, one per non-empty line.
    """
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _split_to_records(
    texts: List[str],
    labels: List[int],
    n_tokens_list: List[int],
) -> List[Dict[str, Any]]:
    """Convert parallel lists to the JSONL record format."""
    return [
        {"text": t, "label": ID2LABEL[l], "n_tokens": n}
        for t, l, n in zip(texts, labels, n_tokens_list)
    ]


# ---------------------------------------------------------------------------
# High-level track preparation (DATA_PREP.md §5–9)
# ---------------------------------------------------------------------------

def load_and_prepare_tracks(
    dataset_name: str = DATASET_CONFIG["dataset_name"],
    text_column: str = DATASET_CONFIG["text_column"],
    label_column: str = DATASET_CONFIG["label_column"],
    tokenizer=None,
    tokenizer_name: str = DATA_PREP_CONFIG["tokenizer"],
    sentence_min_tokens: int = DATA_PREP_CONFIG["sentence_min_tokens"],
    sentence_max_tokens: int = DATA_PREP_CONFIG["sentence_max_tokens"],
    essay_min_tokens: int = DATA_PREP_CONFIG["essay_min_tokens"],
    min_class_samples: int = DATA_PREP_CONFIG["min_class_samples"],
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    seed: int = RANDOM_SEED,
    output_dir: Optional[str] = None,
) -> Dict[str, Tuple]:
    """
    Load a CEFR dataset and produce sentence-level and essay-level splits.

    Pipeline per track:
    1. Load & normalise text; validate labels.
    2. Count tokens with *tokenizer* (loaded from *tokenizer_name* if None).
    3. Deduplicate by (text, label).
    4. Filter by token length to obtain sentence / essay subsets.
    5. Drop CEFR classes with fewer than *min_class_samples* samples.
    6. Stratified 80/10/10 split.
    7. Optionally save as JSONL under *output_dir*/{sentence,essay}/{train,dev,test}.jsonl.

    Args:
        dataset_name: HuggingFace dataset identifier
        text_column: field name for the text
        label_column: field name for the CEFR label
        tokenizer: pre-loaded tokenizer instance; loaded from *tokenizer_name* if None
        tokenizer_name: model name used to load the tokenizer when *tokenizer* is None
        sentence_min_tokens: inclusive lower token bound for sentence track
        sentence_max_tokens: inclusive upper token bound for sentence track
        essay_min_tokens: inclusive lower token bound for essay track
        min_class_samples: minimum samples per class; classes below are dropped
        train_ratio / val_ratio / test_ratio: split proportions (must sum to 1)
        seed: random seed for reproducibility
        output_dir: if given, JSONL splits are saved to this directory

    Returns:
        Dict with keys ``"sentence"`` and ``"essay"``.  Each value is a tuple::

            (
              (train_texts, train_labels, train_n_tokens),
              (val_texts,   val_labels,   val_n_tokens),
              (test_texts,  test_labels,  test_n_tokens),
            )
    """
    from sklearn.model_selection import train_test_split

    set_seed(seed)

    # ---- load raw data -------------------------------------------------------
    texts_raw, labels = load_dataset(
        dataset_name=dataset_name,
        text_column=text_column,
        label_column=label_column,
    )

    # ---- tokenise ------------------------------------------------------------
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    n_tokens_all = build_token_counts(texts_raw, tokenizer)

    # ---- deduplicate ---------------------------------------------------------
    # Dedup on (text, label) before any length filtering so that duplicates
    # that land in different tracks are also eliminated.
    seen: set = set()
    texts_clean, labels_clean, n_tokens_clean = [], [], []
    for t, l, n in zip(texts_raw, labels, n_tokens_all):
        key = (t, l)
        if key not in seen:
            seen.add(key)
            texts_clean.append(t)
            labels_clean.append(l)
            n_tokens_clean.append(n)

    # ---- build tracks --------------------------------------------------------
    def _prepare_track(
        t_min: Optional[int],
        t_max: Optional[int],
    ) -> Tuple:
        t_texts, t_labels, t_ntoks = filter_by_length(
            texts_clean, labels_clean, n_tokens_clean,
            min_tokens=t_min, max_tokens=t_max,
        )
        t_texts, t_labels, t_ntoks = filter_min_class_size(
            t_texts, t_labels, t_ntoks, min_samples=min_class_samples,
        )
        if not t_texts:
            empty: List = []
            return (empty, empty, empty), (empty, empty, empty), (empty, empty, empty)

        # stratified split on texts + labels; carry n_tokens alongside
        tr_t, tmp_t, tr_l, tmp_l, tr_n, tmp_n = train_test_split(
            t_texts, t_labels, t_ntoks,
            test_size=(val_ratio + test_ratio),
            stratify=t_labels,
            random_state=seed,
        )
        val_frac = val_ratio / (val_ratio + test_ratio)
        va_t, te_t, va_l, te_l, va_n, te_n = train_test_split(
            tmp_t, tmp_l, tmp_n,
            test_size=(1.0 - val_frac),
            stratify=tmp_l,
            random_state=seed,
        )
        return (tr_t, tr_l, tr_n), (va_t, va_l, va_n), (te_t, te_l, te_n)

    sentence_splits = _prepare_track(sentence_min_tokens, sentence_max_tokens)
    essay_splits = _prepare_track(essay_min_tokens, None)

    # ---- optional JSONL output -----------------------------------------------
    if output_dir is not None:
        for track_name, splits in (
            ("sentence", sentence_splits),
            ("essay", essay_splits),
        ):
            track_dir = os.path.join(output_dir, track_name)
            for split_name, (s_texts, s_labels, s_ntoks) in zip(
                ("train", "dev", "test"), splits
            ):
                if not s_texts:
                    continue
                records = _split_to_records(s_texts, s_labels, s_ntoks)
                save_jsonl(records, os.path.join(track_dir, f"{split_name}.jsonl"))

    return {"sentence": sentence_splits, "essay": essay_splits}

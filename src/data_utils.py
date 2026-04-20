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
    SUPPORTED_LANGUAGES,
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
        raise RuntimeError(
            "Failed to download/load dataset from HuggingFace Hub. "
            "Check network/proxy access (HTTPS to huggingface.co) and auth token if needed. "
            f"dataset={dataset_name}, split={split}. Original error: {exc}"
        ) from exc
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


def load_dataset_records(
    dataset_name: str = DATASET_CONFIG["dataset_name"],
    split: str = "train",
) -> List[Dict[str, Any]]:
    """
    Load raw HuggingFace dataset records without CEFR/text preprocessing.

    This is used by multilingual builders that need access to metadata such as
    language columns before normalizing and filtering the examples.
    """
    from datasets import load_dataset as hf_load_dataset

    try:
        dataset = hf_load_dataset(dataset_name, split=split)
    except Exception as exc:
        exc_name = type(exc).__name__
        if "DatasetNotFoundError" in exc_name or "not found" in str(exc).lower():
            raise ValueError(
                f"Dataset '{dataset_name}' was not found on the HuggingFace Hub "
                f"or cannot be accessed.\n"
                f"  • Check that the dataset path is correct.\n"
                f"  • If the dataset is private, ensure HF_TOKEN is set.\n"
                f"Original error: {exc}"
            ) from exc
        raise RuntimeError(
            "Failed to download/load dataset from HuggingFace Hub. "
            "Check network/proxy access (HTTPS to huggingface.co) and auth token if needed. "
            f"dataset={dataset_name}, split={split}. Original error: {exc}"
        ) from exc
    return [dict(sample) for sample in dataset]


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


def detect_text_language(text: str) -> str:
    """
    Detect the language of *text* using ``langdetect``.

    Raises:
        ImportError: if langdetect is not installed.
        ValueError: if language detection fails.
    """
    try:
        from langdetect import DetectorFactory, LangDetectException, detect
    except ImportError as exc:
        raise ImportError(
            "langdetect is required for automatic language detection. "
            "Install it with `pip install langdetect`."
        ) from exc

    DetectorFactory.seed = RANDOM_SEED
    try:
        return detect(text).lower()
    except LangDetectException as exc:
        raise ValueError(f"Language detection failed for text: {text[:80]!r}") from exc


def resolve_record_language(
    record: Dict[str, Any],
    dataset_language: Optional[str] = None,
    language_keys: Tuple[str, ...] = ("lang", "language"),
    use_langdetect: bool = False,
) -> Optional[str]:
    """
    Resolve a language code for one dataset record.

    Priority:
    1. Explicit per-record metadata (`lang`, `language`, ...).
    2. Dataset-level language hint.
    3. Optional `langdetect` fallback.
    """
    for key in language_keys:
        value = record.get(key)
        if value is None:
            continue
        value = str(value).strip().lower()
        if value:
            return value

    if dataset_language:
        return dataset_language.lower()

    if use_langdetect:
        text = record.get("text")
        if text and str(text).strip():
            return detect_text_language(str(text))

    return None


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


def _prepare_tracks_from_arrays(
    texts_raw: List[str],
    labels: List[int],
    tokenizer=None,
    tokenizer_name: str = DATA_PREP_CONFIG["tokenizer"],
    sentence_min_tokens: int = DATA_PREP_CONFIG["sentence_min_tokens"],
    sentence_max_tokens: int = DATA_PREP_CONFIG["sentence_max_tokens"],
    essay_min_tokens: int = DATA_PREP_CONFIG["essay_min_tokens"],
    min_class_samples: int = DATA_PREP_CONFIG["min_class_samples"],
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    seed: int = RANDOM_SEED,
    output_dir: Optional[str] = None,
) -> Dict[str, Tuple]:
    """Shared implementation for single-language and combined-corpus builders."""
    from sklearn.model_selection import train_test_split

    set_seed(seed)

    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    n_tokens_all = build_token_counts(texts_raw, tokenizer)

    seen: set = set()
    texts_clean, labels_clean, n_tokens_clean = [], [], []
    for t, l, n in zip(texts_raw, labels, n_tokens_all):
        key = (t, l)
        if key not in seen:
            seen.add(key)
            texts_clean.append(t)
            labels_clean.append(l)
            n_tokens_clean.append(n)

    def _prepare_track(
        t_min: Optional[int],
        t_max: Optional[int],
    ) -> Tuple:
        t_texts, t_labels, t_ntoks = filter_by_length(
            texts_clean,
            labels_clean,
            n_tokens_clean,
            min_tokens=t_min,
            max_tokens=t_max,
        )
        t_texts, t_labels, t_ntoks = filter_min_class_size(
            t_texts,
            t_labels,
            t_ntoks,
            min_samples=min_class_samples,
        )
        if not t_texts:
            empty: List = []
            return (empty, empty, empty), (empty, empty, empty), (empty, empty, empty)

        tr_t, tmp_t, tr_l, tmp_l, tr_n, tmp_n = train_test_split(
            t_texts,
            t_labels,
            t_ntoks,
            test_size=(val_ratio + test_ratio),
            stratify=t_labels,
            random_state=seed,
        )
        val_frac = val_ratio / (val_ratio + test_ratio)
        va_t, te_t, va_l, te_l, va_n, te_n = train_test_split(
            tmp_t,
            tmp_l,
            tmp_n,
            test_size=(1.0 - val_frac),
            stratify=tmp_l,
            random_state=seed,
        )
        return (tr_t, tr_l, tr_n), (va_t, va_l, va_n), (te_t, te_l, te_n)

    sentence_splits = _prepare_track(sentence_min_tokens, sentence_max_tokens)
    essay_splits = _prepare_track(essay_min_tokens, None)

    if output_dir is not None:
        for track_name, splits in (
            ("sentence", sentence_splits),
            ("essay", essay_splits),
        ):
            track_dir = os.path.join(output_dir, track_name)
            for split_name, (s_texts, s_labels, s_ntoks) in zip(
                ("train", "dev", "test"),
                splits,
            ):
                if not s_texts:
                    continue
                records = _split_to_records(s_texts, s_labels, s_ntoks)
                save_jsonl(records, os.path.join(track_dir, f"{split_name}.jsonl"))

    return {"sentence": sentence_splits, "essay": essay_splits}


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
    texts_raw, labels = load_dataset(
        dataset_name=dataset_name,
        text_column=text_column,
        label_column=label_column,
    )
    return _prepare_tracks_from_arrays(
        texts_raw=texts_raw,
        labels=labels,
        tokenizer=tokenizer,
        tokenizer_name=tokenizer_name,
        sentence_min_tokens=sentence_min_tokens,
        sentence_max_tokens=sentence_max_tokens,
        essay_min_tokens=essay_min_tokens,
        min_class_samples=min_class_samples,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        output_dir=output_dir,
    )


def load_and_prepare_multilingual_tracks(
    dataset_specs: List[Dict[str, Any]],
    text_column: str = DATASET_CONFIG["text_column"],
    label_column: str = DATASET_CONFIG["label_column"],
    tokenizer=None,
    tokenizer_name: str = DATA_PREP_CONFIG["tokenizer"],
    sentence_min_tokens: int = DATA_PREP_CONFIG["sentence_min_tokens"],
    sentence_max_tokens: int = DATA_PREP_CONFIG["sentence_max_tokens"],
    essay_min_tokens: int = DATA_PREP_CONFIG["essay_min_tokens"],
    min_class_samples: int = DATA_PREP_CONFIG["min_class_samples"],
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    seed: int = RANDOM_SEED,
    output_dir: Optional[str] = None,
    use_langdetect: bool = False,
    language_keys: Tuple[str, ...] = ("lang", "language"),
    allowed_languages: Tuple[str, ...] = SUPPORTED_LANGUAGES,
) -> Dict[str, Dict[str, Tuple]]:
    """
    Build per-language sentence/essay splits from a combined multilingual corpus.

    Each item in ``dataset_specs`` must contain at least ``dataset_name`` and may
    optionally contain:
    * ``text_column``
    * ``label_column``
    * ``language``   – dataset-level language hint
    * ``split``      – defaults to ``train``
    """
    set_seed(seed)

    per_language: Dict[str, Dict[str, List[Any]]] = {
        lang: {"texts": [], "labels": []}
        for lang in allowed_languages
    }

    for spec in dataset_specs:
        dataset_name = spec["dataset_name"]
        split = spec.get("split", "train")
        spec_text_column = spec.get("text_column", text_column)
        spec_label_column = spec.get("label_column", label_column)
        dataset_language = spec.get("language")

        for record in load_dataset_records(dataset_name=dataset_name, split=split):
            text = record.get(spec_text_column, "")
            raw_label = record.get(spec_label_column)
            if not text or not str(text).strip():
                continue

            label = normalize_label(raw_label)
            if label is None:
                continue

            resolved_language = resolve_record_language(
                record,
                dataset_language=dataset_language,
                language_keys=language_keys,
                use_langdetect=use_langdetect,
            )
            if resolved_language not in per_language:
                continue

            per_language[resolved_language]["texts"].append(normalize_text(str(text)))
            per_language[resolved_language]["labels"].append(LABEL2ID[label])

    multilingual_tracks: Dict[str, Dict[str, Tuple]] = {}
    for language, payload in per_language.items():
        lang_output = os.path.join(output_dir, language) if output_dir else None
        multilingual_tracks[language] = _prepare_tracks_from_arrays(
            texts_raw=payload["texts"],
            labels=payload["labels"],
            tokenizer=tokenizer,
            tokenizer_name=tokenizer_name,
            sentence_min_tokens=sentence_min_tokens,
            sentence_max_tokens=sentence_max_tokens,
            essay_min_tokens=essay_min_tokens,
            min_class_samples=min_class_samples,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            output_dir=lang_output,
        )

    return multilingual_tracks

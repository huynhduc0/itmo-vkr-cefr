"""
Data loading and preprocessing utilities for CEFR classification.
"""

import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.config import (
    CEFR_LEVELS,
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
    """
    from datasets import load_dataset as hf_load_dataset

    dataset = hf_load_dataset(dataset_name, split=split)
    texts, labels = [], []
    for sample in dataset:
        text = sample.get(text_column, "")
        raw_label = sample.get(label_column, None)
        if not text or not text.strip():
            continue
        label = normalize_label(raw_label)
        if label is None:
            continue
        texts.append(text.strip())
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
) -> Tuple[
    Tuple[List[str], List[int]],
    Tuple[List[str], List[int]],
    Tuple[List[str], List[int]],
]:
    """
    Load dataset and perform stratified split in one step.
    """
    texts, labels = load_dataset(
        dataset_name=dataset_name,
        text_column=text_column,
        label_column=label_column,
    )
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

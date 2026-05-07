"""
Evaluation utilities: accuracy, macro-F1, QWK, confusion matrix, error analysis.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
)

from src.config import CEFR_LEVELS, ID2LABEL, LABEL2ID


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
) -> Dict[str, float]:
    """
    Compute accuracy, macro-F1, quadratic weighted kappa, and MAE.

    Args:
        y_true: ground-truth label ids
        y_pred: predicted label ids

    Returns:
        Dictionary with keys 'accuracy', 'macro_f1', 'qwk', 'mae'.
    """
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    if len(set(y_true) | set(y_pred)) < 2:
        qwk = 0.0
    else:
        qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
        if np.isnan(qwk):
            qwk = 0.0
    mae = mean_absolute_error(y_true, y_pred)
    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "qwk": float(qwk),
        "mae": float(mae),
    }


def bootstrap_ci(
    y_true: List[int],
    y_pred: List[int],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Compute bootstrap 95% confidence-interval half-widths for QWK and Macro-F1.

    Performs *n_bootstrap* resamplings with replacement and returns the
    half-width of the symmetric percentile CI for each metric, so results
    can be reported as ``{metric} ± {ci}``.

    Args:
        y_true:      ground-truth label ids
        y_pred:      predicted label ids
        n_bootstrap: number of bootstrap resamples (default 1000)
        confidence:  CI level (default 0.95)
        seed:        numpy random seed for reproducibility

    Returns:
        Dict with keys ``qwk_ci`` and ``macro_f1_ci`` (half-widths).
    """
    rng = np.random.default_rng(seed)
    yt = np.array(y_true)
    yp = np.array(y_pred)
    n = len(yt)

    qwk_samples: List[float] = []
    f1_samples: List[float] = []

    alpha = (1.0 - confidence) / 2.0

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        m = compute_metrics(yt[idx].tolist(), yp[idx].tolist())
        qwk_samples.append(m["qwk"])
        f1_samples.append(m["macro_f1"])

    qwk_lo = float(np.quantile(qwk_samples, alpha))
    qwk_hi = float(np.quantile(qwk_samples, 1.0 - alpha))
    f1_lo = float(np.quantile(f1_samples, alpha))
    f1_hi = float(np.quantile(f1_samples, 1.0 - alpha))

    return {
        "qwk_ci": (qwk_hi - qwk_lo) / 2.0,
        "macro_f1_ci": (f1_hi - f1_lo) / 2.0,
    }


def compute_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    labels: Optional[List[int]] = None,
) -> np.ndarray:
    """
    Compute confusion matrix for CEFR levels.

    Args:
        y_true: ground-truth label ids
        y_pred: predicted label ids
        labels: optional list of label ids to include (defaults to all 6)

    Returns:
        Confusion matrix as a numpy array.
    """
    if labels is None:
        labels = list(range(len(CEFR_LEVELS)))
    return confusion_matrix(y_true, y_pred, labels=labels)


def adjacent_confusion_analysis(
    y_true: List[int],
    y_pred: List[int],
) -> Dict[str, int]:
    """
    Count adjacent-level confusions: A1↔A2, A2↔B1, B1↔B2, B2↔C1, C1↔C2.

    Returns:
        Dict mapping adjacent pair string to count of confused samples.
    """
    adjacent_pairs = [
        ("A1", "A2"),
        ("A2", "B1"),
        ("B1", "B2"),
        ("B2", "C1"),
        ("C1", "C2"),
    ]
    counts: Dict[str, int] = {}
    for level_a, level_b in adjacent_pairs:
        id_a = LABEL2ID[level_a]
        id_b = LABEL2ID[level_b]
        pair_key = f"{level_a}↔{level_b}"
        count = sum(
            1
            for t, p in zip(y_true, y_pred)
            if (t == id_a and p == id_b) or (t == id_b and p == id_a)
        )
        counts[pair_key] = count
    return counts


def print_evaluation_report(
    y_true: List[int],
    y_pred: List[int],
    model_name: str = "Model",
) -> None:
    """Print a formatted evaluation report to stdout."""
    metrics = compute_metrics(y_true, y_pred)
    cm = compute_confusion_matrix(y_true, y_pred)
    adj = adjacent_confusion_analysis(y_true, y_pred)

    print(f"\n{'='*50}")
    print(f"Evaluation Report: {model_name}")
    print(f"{'='*50}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Macro-F1  : {metrics['macro_f1']:.4f}")
    print(f"  QWK       : {metrics['qwk']:.4f}")
    print(f"  MAE       : {metrics['mae']:.4f}")

    print("\nConfusion Matrix (rows=true, cols=pred):")
    header = "      " + "  ".join(f"{l:>3}" for l in CEFR_LEVELS)
    print(header)
    for i, level in enumerate(CEFR_LEVELS):
        row = "  ".join(f"{cm[i, j]:>3}" for j in range(len(CEFR_LEVELS)))
        print(f"  {level}: {row}")

    print("\nAdjacent-level confusions:")
    for pair, count in adj.items():
        print(f"  {pair}: {count}")
    print(f"{'='*50}\n")

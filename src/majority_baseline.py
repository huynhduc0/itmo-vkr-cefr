"""
Majority class baseline for CEFR classification (Exp 0).

Predicts the most frequent class in the training set for every sample.
Serves as a sanity-check lower bound.
"""

from collections import Counter
from typing import List

import numpy as np

from src.config import CEFR_LEVELS, ID2LABEL


class MajorityClassifier:
    """
    Predicts the most frequent CEFR level seen during training.
    """

    def __init__(self):
        self.majority_label: int = -1

    def fit(self, labels: List[int]) -> "MajorityClassifier":
        """
        Determine the majority class from training labels.

        Args:
            labels: list of integer label ids

        Returns:
            self
        """
        if not labels:
            raise ValueError("Cannot fit on empty label list.")
        counter = Counter(labels)
        self.majority_label = counter.most_common(1)[0][0]
        return self

    def predict(self, n: int) -> np.ndarray:
        """
        Predict the majority class for n samples.

        Args:
            n: number of samples to predict

        Returns:
            Array of length n filled with the majority label id.
        """
        if self.majority_label == -1:
            raise RuntimeError("MajorityClassifier is not fitted yet. Call fit() first.")
        return np.full(n, self.majority_label, dtype=int)

    @property
    def majority_level(self) -> str:
        """Return the majority CEFR level name."""
        if self.majority_label == -1:
            raise RuntimeError("MajorityClassifier is not fitted yet.")
        return ID2LABEL[self.majority_label]

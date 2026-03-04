"""
Hybrid long-text strategy for essay-level CEFR classification (Exp 5).

Approach:
1. Split the essay into individual sentences.
2. Classify each sentence using a pre-trained sentence-level classifier.
3. Aggregate per-sentence predictions into an essay-level prediction via
   mean probability pooling or majority/weighted voting.
"""

import re
from typing import Callable, List, Optional

import numpy as np

from src.config import CEFR_LEVELS, LABEL2ID


_NUM_LABELS = len(LABEL2ID)

# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

_SENT_SPLIT_RE = re.compile(
    r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s"
)


def split_into_sentences(text: str) -> List[str]:
    """
    Split a text into sentences using a lightweight regex heuristic.

    Tries to use NLTK's Punkt tokenizer when available, falls back to a
    simple regex split so that there are no hard dependencies at import time.

    Args:
        text: input essay or paragraph string

    Returns:
        List of non-empty sentence strings.
    """
    try:
        import nltk

        try:
            sent_detector = nltk.data.load("tokenizers/punkt_tab/english.pickle")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
            sent_detector = nltk.data.load("tokenizers/punkt_tab/english.pickle")
        sentences = sent_detector.tokenize(text.strip())
    except Exception:
        sentences = _SENT_SPLIT_RE.split(text.strip())

    return [s.strip() for s in sentences if s.strip()]


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def aggregate_mean_prob(probs: np.ndarray) -> int:
    """
    Average sentence-level class probabilities and return the argmax.

    Args:
        probs: (num_sentences, num_labels) probability array

    Returns:
        Predicted label id.
    """
    mean_probs = probs.mean(axis=0)
    return int(np.argmax(mean_probs))


def aggregate_majority_vote(label_ids: List[int]) -> int:
    """
    Return the most frequent predicted label id.

    Args:
        label_ids: list of per-sentence predicted label ids

    Returns:
        Most common label id (ties broken by lowest id).
    """
    from collections import Counter

    if not label_ids:
        return 0
    return Counter(label_ids).most_common(1)[0][0]


def aggregate_weighted_vote(label_ids: List[int], weights: List[float]) -> int:
    """
    Weighted vote over sentence-level predictions.

    Args:
        label_ids: list of per-sentence predicted label ids
        weights: weight for each sentence (e.g., sentence length)

    Returns:
        Label id with the highest weighted vote.
    """
    scores = np.zeros(_NUM_LABELS)
    for label, weight in zip(label_ids, weights):
        scores[label] += weight
    return int(np.argmax(scores))


# ---------------------------------------------------------------------------
# HybridEssayClassifier
# ---------------------------------------------------------------------------

class HybridEssayClassifier:
    """
    Classifies essays by aggregating sentence-level classifier outputs.

    The sentence-level classifier is provided as a callable:
        predict_fn(texts: List[str]) -> np.ndarray of label ids

    For probability-based aggregation supply:
        predict_proba_fn(texts: List[str]) -> np.ndarray (N, num_labels)
    """

    def __init__(
        self,
        predict_fn: Callable[[List[str]], np.ndarray],
        predict_proba_fn: Optional[Callable[[List[str]], np.ndarray]] = None,
        aggregation: str = "mean_prob",
    ):
        """
        Args:
            predict_fn: function mapping a list of texts to label id array
            predict_proba_fn: function mapping a list of texts to probability
                              matrix (N, num_labels); required for 'mean_prob'
            aggregation: one of 'mean_prob', 'majority_vote', 'weighted_vote'
        """
        if aggregation not in ("mean_prob", "majority_vote", "weighted_vote"):
            raise ValueError(
                f"Unknown aggregation '{aggregation}'. "
                "Choose from: mean_prob, majority_vote, weighted_vote."
            )
        if aggregation == "mean_prob" and predict_proba_fn is None:
            raise ValueError(
                "predict_proba_fn is required for 'mean_prob' aggregation."
            )
        self.predict_fn = predict_fn
        self.predict_proba_fn = predict_proba_fn
        self.aggregation = aggregation

    def predict_one(self, essay: str) -> int:
        """
        Predict the CEFR level of a single essay.

        Returns:
            Predicted label id.
        """
        sentences = split_into_sentences(essay)
        if not sentences:
            return 0

        if self.aggregation == "mean_prob":
            probs = self.predict_proba_fn(sentences)
            return aggregate_mean_prob(probs)

        label_ids = self.predict_fn(sentences).tolist()

        if self.aggregation == "weighted_vote":
            weights = [len(s) for s in sentences]
            return aggregate_weighted_vote(label_ids, weights)

        return aggregate_majority_vote(label_ids)

    def predict(self, essays: List[str]) -> np.ndarray:
        """
        Predict CEFR levels for a list of essays.

        Returns:
            Array of predicted label ids.
        """
        return np.array([self.predict_one(essay) for essay in essays])

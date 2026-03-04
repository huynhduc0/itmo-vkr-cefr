"""
Unit tests for the hybrid essay classifier.
"""

import numpy as np
import pytest

from src.hybrid_essay import (
    HybridEssayClassifier,
    aggregate_majority_vote,
    aggregate_mean_prob,
    aggregate_weighted_vote,
    split_into_sentences,
)


class TestSplitIntoSentences:
    def test_single_sentence(self):
        sents = split_into_sentences("Hello world.")
        assert len(sents) == 1
        assert sents[0] == "Hello world."

    def test_multiple_sentences(self):
        text = "I went to the store. I bought some milk. It was cold outside."
        sents = split_into_sentences(text)
        assert len(sents) >= 2

    def test_empty_string(self):
        sents = split_into_sentences("")
        assert sents == []

    def test_whitespace_only(self):
        sents = split_into_sentences("   ")
        assert sents == []

    def test_no_empty_sentences(self):
        text = "First sentence. Second sentence. Third sentence."
        sents = split_into_sentences(text)
        for s in sents:
            assert s.strip() != ""


class TestAggregateMeanProb:
    def test_basic(self):
        probs = np.array([
            [0.1, 0.7, 0.1, 0.05, 0.025, 0.025],
            [0.05, 0.8, 0.1, 0.025, 0.0125, 0.0125],
        ])
        pred = aggregate_mean_prob(probs)
        assert pred == 1

    def test_uniform_returns_label_zero(self):
        probs = np.ones((3, 6)) / 6
        pred = aggregate_mean_prob(probs)
        assert pred == 0

    def test_single_sentence(self):
        probs = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
        pred = aggregate_mean_prob(probs)
        assert pred == 3


class TestAggregateMajorityVote:
    def test_clear_majority(self):
        assert aggregate_majority_vote([2, 2, 2, 1, 3]) == 2

    def test_single_element(self):
        assert aggregate_majority_vote([4]) == 4

    def test_empty_returns_zero(self):
        assert aggregate_majority_vote([]) == 0


class TestAggregateWeightedVote:
    def test_higher_weight_wins(self):
        label_ids = [0, 1]
        weights = [1.0, 10.0]
        assert aggregate_weighted_vote(label_ids, weights) == 1

    def test_equal_weights_like_majority(self):
        label_ids = [2, 2, 3]
        weights = [1.0, 1.0, 1.0]
        assert aggregate_weighted_vote(label_ids, weights) == 2


class TestHybridEssayClassifier:
    @pytest.fixture
    def simple_classifier(self):
        """Return a dummy HybridEssayClassifier backed by a constant predictor."""
        def predict_fn(texts):
            return np.array([1] * len(texts))

        def predict_proba_fn(texts):
            probs = np.zeros((len(texts), 6))
            probs[:, 1] = 1.0
            return probs

        return predict_fn, predict_proba_fn

    def test_mean_prob_aggregation(self, simple_classifier):
        predict_fn, predict_proba_fn = simple_classifier
        clf = HybridEssayClassifier(
            predict_fn=predict_fn,
            predict_proba_fn=predict_proba_fn,
            aggregation="mean_prob",
        )
        result = clf.predict_one("I went to school. The teacher was kind.")
        assert result == 1

    def test_majority_vote_aggregation(self, simple_classifier):
        predict_fn, _ = simple_classifier
        clf = HybridEssayClassifier(predict_fn=predict_fn, aggregation="majority_vote")
        result = clf.predict_one("Short text here.")
        assert result == 1

    def test_weighted_vote_aggregation(self, simple_classifier):
        predict_fn, _ = simple_classifier
        clf = HybridEssayClassifier(predict_fn=predict_fn, aggregation="weighted_vote")
        result = clf.predict_one("This is the essay. It has some sentences.")
        assert result == 1

    def test_predict_batch(self, simple_classifier):
        predict_fn, predict_proba_fn = simple_classifier
        clf = HybridEssayClassifier(
            predict_fn=predict_fn,
            predict_proba_fn=predict_proba_fn,
            aggregation="mean_prob",
        )
        essays = [
            "First essay. Has multiple sentences.",
            "Second essay. Also multiple sentences.",
        ]
        preds = clf.predict(essays)
        assert isinstance(preds, np.ndarray)
        assert len(preds) == 2

    def test_invalid_aggregation_raises(self, simple_classifier):
        predict_fn, _ = simple_classifier
        with pytest.raises(ValueError):
            HybridEssayClassifier(predict_fn=predict_fn, aggregation="unknown")

    def test_mean_prob_without_proba_fn_raises(self, simple_classifier):
        predict_fn, _ = simple_classifier
        with pytest.raises(ValueError):
            HybridEssayClassifier(predict_fn=predict_fn, aggregation="mean_prob")

    def test_empty_essay_returns_label_zero(self, simple_classifier):
        predict_fn, _ = simple_classifier
        clf = HybridEssayClassifier(predict_fn=predict_fn, aggregation="majority_vote")
        result = clf.predict_one("")
        assert result == 0

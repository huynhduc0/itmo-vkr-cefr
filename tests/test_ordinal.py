"""
Unit tests for ordinal classification utilities (CORAL).
"""

import numpy as np
import pytest

from src.ordinal_classifier import coral_predict


class TestCoralPredict:
    def test_accepts_torch_tensor_input(self):
        torch = pytest.importorskip("torch")
        logits = torch.tensor([[10.0, -10.0, 10.0, -10.0, 10.0]])
        preds = coral_predict(logits)
        assert isinstance(preds, np.ndarray)
        assert preds[0] == 3

    def test_threshold_at_zero_counts_as_exceeded(self):
        # sigmoid(0) == 0.5 and function uses >= 0.5
        logits = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]])
        preds = coral_predict(logits)
        assert preds[0] == 5

    def test_all_thresholds_exceeded(self):
        # large positive logits -> all probs >= 0.5 -> label = K-1 = 5
        logits = np.array([[10.0, 10.0, 10.0, 10.0, 10.0]])
        preds = coral_predict(logits)
        assert preds[0] == 5

    def test_no_threshold_exceeded(self):
        # large negative logits -> no prob >= 0.5 -> label = 0
        logits = np.array([[-10.0, -10.0, -10.0, -10.0, -10.0]])
        preds = coral_predict(logits)
        assert preds[0] == 0

    def test_partial_thresholds(self):
        # First 2 exceeded, last 3 not -> label = 2
        logits = np.array([[5.0, 5.0, -5.0, -5.0, -5.0]])
        preds = coral_predict(logits)
        assert preds[0] == 2

    def test_batch_output_shape(self):
        logits = np.zeros((4, 5))
        preds = coral_predict(logits)
        assert preds.shape == (4,)

    def test_output_type(self):
        logits = np.array([[1.0, -1.0, 1.0, -1.0, 1.0]])
        preds = coral_predict(logits)
        assert preds.dtype in (np.int32, np.int64, int)

    def test_batch_mixed_predictions(self):
        logits = np.array([
            [10.0, 10.0, 10.0, 10.0, 10.0],     # -> 5
            [-10.0, -10.0, -10.0, -10.0, -10.0],  # -> 0
            [10.0, -10.0, 10.0, -10.0, 10.0],   # -> 3
        ])
        preds = coral_predict(logits)
        assert preds.tolist() == [5, 0, 3]

    def test_valid_label_range(self):
        rng = np.random.default_rng(0)
        logits = rng.standard_normal((20, 5))
        preds = coral_predict(logits)
        assert np.all(preds >= 0)
        assert np.all(preds <= 5)

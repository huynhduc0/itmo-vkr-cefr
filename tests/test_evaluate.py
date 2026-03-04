"""
Unit tests for evaluation utilities.
"""

import numpy as np
import pytest

from src.evaluate import (
    adjacent_confusion_analysis,
    compute_confusion_matrix,
    compute_metrics,
    print_evaluation_report,
)


@pytest.fixture
def perfect_predictions():
    y_true = list(range(6)) * 10
    y_pred = y_true.copy()
    return y_true, y_pred


@pytest.fixture
def imperfect_predictions():
    y_true = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    y_pred = [0, 1, 1, 2, 2, 1, 3, 4, 4, 3, 5, 4]
    return y_true, y_pred


class TestComputeMetrics:
    def test_perfect_accuracy(self, perfect_predictions):
        y_true, y_pred = perfect_predictions
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["accuracy"] == pytest.approx(1.0)
        assert metrics["macro_f1"] == pytest.approx(1.0)
        assert metrics["qwk"] == pytest.approx(1.0)

    def test_metric_keys(self, imperfect_predictions):
        y_true, y_pred = imperfect_predictions
        metrics = compute_metrics(y_true, y_pred)
        assert "accuracy" in metrics
        assert "macro_f1" in metrics
        assert "qwk" in metrics

    def test_metric_ranges(self, imperfect_predictions):
        y_true, y_pred = imperfect_predictions
        metrics = compute_metrics(y_true, y_pred)
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["macro_f1"] <= 1.0
        assert -1.0 <= metrics["qwk"] <= 1.0

    def test_returns_floats(self, imperfect_predictions):
        y_true, y_pred = imperfect_predictions
        metrics = compute_metrics(y_true, y_pred)
        for v in metrics.values():
            assert isinstance(v, float)


class TestComputeConfusionMatrix:
    def test_shape(self, perfect_predictions):
        y_true, y_pred = perfect_predictions
        cm = compute_confusion_matrix(y_true, y_pred)
        assert cm.shape == (6, 6)

    def test_diagonal_for_perfect(self, perfect_predictions):
        y_true, y_pred = perfect_predictions
        cm = compute_confusion_matrix(y_true, y_pred)
        assert np.all(np.diag(cm) > 0)
        off_diag = cm - np.diag(np.diag(cm))
        assert np.all(off_diag == 0)

    def test_returns_ndarray(self, imperfect_predictions):
        y_true, y_pred = imperfect_predictions
        cm = compute_confusion_matrix(y_true, y_pred)
        assert isinstance(cm, np.ndarray)


class TestAdjacentConfusionAnalysis:
    def test_keys_present(self, imperfect_predictions):
        y_true, y_pred = imperfect_predictions
        adj = adjacent_confusion_analysis(y_true, y_pred)
        expected_keys = {"A1↔A2", "A2↔B1", "B1↔B2", "B2↔C1", "C1↔C2"}
        assert set(adj.keys()) == expected_keys

    def test_no_adjacent_confusion(self, perfect_predictions):
        y_true, y_pred = perfect_predictions
        adj = adjacent_confusion_analysis(y_true, y_pred)
        for count in adj.values():
            assert count == 0

    def test_counts_both_directions(self):
        y_true = [0, 1]
        y_pred = [1, 0]
        adj = adjacent_confusion_analysis(y_true, y_pred)
        assert adj["A1↔A2"] == 2

    def test_non_negative_counts(self, imperfect_predictions):
        y_true, y_pred = imperfect_predictions
        adj = adjacent_confusion_analysis(y_true, y_pred)
        for count in adj.values():
            assert count >= 0


class TestPrintEvaluationReport:
    def test_runs_without_error(self, imperfect_predictions, capsys):
        y_true, y_pred = imperfect_predictions
        print_evaluation_report(y_true, y_pred, model_name="Test Model")
        captured = capsys.readouterr()
        assert "Test Model" in captured.out
        assert "Accuracy" in captured.out
        assert "Macro-F1" in captured.out
        assert "QWK" in captured.out

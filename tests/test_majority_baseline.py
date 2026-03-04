"""
Unit tests for MajorityClassifier.
"""

import numpy as np
import pytest

from src.majority_baseline import MajorityClassifier


class TestMajorityClassifier:
    def test_fit_and_predict(self):
        clf = MajorityClassifier()
        labels = [0, 0, 0, 1, 2, 2]
        clf.fit(labels)
        preds = clf.predict(4)
        assert np.all(preds == 0)

    def test_majority_level_name(self):
        clf = MajorityClassifier()
        clf.fit([2, 2, 2, 3])
        assert clf.majority_level == "B1"

    def test_predict_length(self):
        clf = MajorityClassifier()
        clf.fit([1, 1, 2])
        preds = clf.predict(10)
        assert len(preds) == 10

    def test_predict_returns_ndarray(self):
        clf = MajorityClassifier()
        clf.fit([3])
        preds = clf.predict(5)
        assert isinstance(preds, np.ndarray)

    def test_fit_empty_raises(self):
        clf = MajorityClassifier()
        with pytest.raises(ValueError):
            clf.fit([])

    def test_predict_before_fit_raises(self):
        clf = MajorityClassifier()
        with pytest.raises(RuntimeError):
            clf.predict(3)

    def test_majority_level_before_fit_raises(self):
        clf = MajorityClassifier()
        with pytest.raises(RuntimeError):
            _ = clf.majority_level

    def test_tie_broken_by_most_common(self):
        clf = MajorityClassifier()
        labels = [0, 0, 1, 1, 1, 2]
        clf.fit(labels)
        preds = clf.predict(1)
        assert preds[0] == 1

    def test_single_class(self):
        clf = MajorityClassifier()
        clf.fit([5, 5, 5])
        assert clf.majority_level == "C2"
        preds = clf.predict(3)
        assert np.all(preds == 5)

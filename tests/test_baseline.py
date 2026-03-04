"""
Unit tests for the TF-IDF + Logistic Regression baseline.
"""

import numpy as np
import pytest

from src.baseline_tfidf import build_tfidf_pipeline, predict_baseline, train_baseline
from src.config import CEFR_LEVELS, LABEL2ID


@pytest.fixture
def small_dataset():
    """Generate a small synthetic CEFR dataset for fast testing."""
    texts = []
    labels = []
    level_texts = {
        "A1": ["I am a student.", "This is a cat.", "Hello world."],
        "A2": ["I like to play football.", "She has a big house.", "We eat dinner."],
        "B1": [
            "The weather is quite nice today.",
            "I enjoy reading books in my free time.",
            "They decided to go for a walk.",
        ],
        "B2": [
            "The government implemented new policies.",
            "She presented her research findings.",
            "The economy has been growing steadily.",
        ],
        "C1": [
            "The phenomenon has been extensively documented.",
            "His arguments are well-substantiated.",
            "The implications are far-reaching.",
        ],
        "C2": [
            "The epistemological underpinnings of the theory are contested.",
            "Her nuanced analysis elucidates the paradox.",
            "The synthesis of disparate evidence is compelling.",
        ],
    }
    for level, level_label_texts in level_texts.items():
        for text in level_label_texts:
            texts.append(text)
            labels.append(LABEL2ID[level])
    return texts, labels


class TestBuildTfidfPipeline:
    def test_pipeline_creation(self):
        pipeline = build_tfidf_pipeline()
        assert pipeline is not None
        assert "features" in pipeline.named_steps
        assert "clf" in pipeline.named_steps

    def test_pipeline_has_word_and_char_features(self):
        pipeline = build_tfidf_pipeline()
        feature_union = pipeline.named_steps["features"]
        transformer_names = [name for name, _ in feature_union.transformer_list]
        assert "word" in transformer_names
        assert "char" in transformer_names


class TestTrainBaseline:
    def test_train_and_predict(self, small_dataset):
        texts, labels = small_dataset
        pipeline = train_baseline(texts, labels)
        preds = predict_baseline(pipeline, texts)
        assert len(preds) == len(texts)
        assert all(p in range(len(CEFR_LEVELS)) for p in preds)

    def test_predict_returns_array(self, small_dataset):
        texts, labels = small_dataset
        pipeline = train_baseline(texts, labels)
        preds = predict_baseline(pipeline, texts[:3])
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (3,)

    def test_train_with_custom_params(self, small_dataset):
        texts, labels = small_dataset
        pipeline = train_baseline(
            texts,
            labels,
            lr_max_iter=100,
            lr_C=0.5,
        )
        preds = predict_baseline(pipeline, texts)
        assert len(preds) == len(texts)

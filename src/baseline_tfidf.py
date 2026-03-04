"""
TF-IDF + Logistic Regression baseline for CEFR classification.
"""

from typing import List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from src.config import LABEL2ID, TFIDF_CONFIG


def build_tfidf_pipeline(
    word_ngram_range: Tuple[int, int] = TFIDF_CONFIG["word_ngram_range"],
    char_ngram_range: Tuple[int, int] = TFIDF_CONFIG["char_ngram_range"],
    max_features: Optional[int] = TFIDF_CONFIG["max_features"],
    lr_max_iter: int = TFIDF_CONFIG["lr_max_iter"],
    lr_C: float = TFIDF_CONFIG["lr_C"],
    random_state: int = 42,
) -> Pipeline:
    """
    Build a TF-IDF + Logistic Regression pipeline that combines
    word n-grams (1–2) and character n-grams (3–5).
    """
    word_tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=word_ngram_range,
        max_features=max_features,
        sublinear_tf=True,
    )
    char_tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=char_ngram_range,
        max_features=max_features,
        sublinear_tf=True,
    )
    combined = FeatureUnion([
        ("word", word_tfidf),
        ("char", char_tfidf),
    ])
    clf = LogisticRegression(
        max_iter=lr_max_iter,
        C=lr_C,
        random_state=random_state,
        solver="lbfgs",
    )
    return Pipeline([
        ("features", combined),
        ("clf", clf),
    ])


def train_baseline(
    train_texts: List[str],
    train_labels: List[int],
    **kwargs,
) -> Pipeline:
    """
    Train the TF-IDF + LR baseline model.

    Returns:
        Fitted sklearn Pipeline.
    """
    pipeline = build_tfidf_pipeline(**kwargs)
    pipeline.fit(train_texts, train_labels)
    return pipeline


def predict_baseline(
    pipeline: Pipeline,
    texts: List[str],
) -> np.ndarray:
    """
    Generate predictions using a trained baseline pipeline.

    Returns:
        Array of predicted label ids.
    """
    return pipeline.predict(texts)

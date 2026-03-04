"""
Training script for TF-IDF + Logistic Regression baseline.

Usage:
    python -m src.train_baseline [--dataset DATASET] [--output OUTPUT]
"""

import argparse
import os
import pickle

from src.baseline_tfidf import predict_baseline, train_baseline
from src.config import DATASET_CONFIG, RANDOM_SEED
from src.data_utils import load_and_split_dataset, set_seed
from src.evaluate import print_evaluation_report


def parse_args():
    parser = argparse.ArgumentParser(description="Train TF-IDF + LR baseline")
    parser.add_argument(
        "--dataset",
        default=DATASET_CONFIG["dataset_name"],
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--text_column",
        default=DATASET_CONFIG["text_column"],
        help="Name of the text column",
    )
    parser.add_argument(
        "--label_column",
        default=DATASET_CONFIG["label_column"],
        help="Name of the label column",
    )
    parser.add_argument(
        "--output",
        default="checkpoints/baseline",
        help="Directory to save model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    print(f"Loading dataset: {args.dataset}")
    (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels) = (
        load_and_split_dataset(
            dataset_name=args.dataset,
            text_column=args.text_column,
            label_column=args.label_column,
            seed=args.seed,
        )
    )
    print(f"Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")

    print("Training baseline model...")
    pipeline = train_baseline(train_texts, train_labels)

    print("Evaluating on validation set...")
    val_preds = predict_baseline(pipeline, val_texts)
    print_evaluation_report(val_labels, val_preds.tolist(), model_name="TF-IDF+LR (Val)")

    print("Evaluating on test set...")
    test_preds = predict_baseline(pipeline, test_texts)
    print_evaluation_report(test_labels, test_preds.tolist(), model_name="TF-IDF+LR (Test)")

    os.makedirs(args.output, exist_ok=True)
    model_path = os.path.join(args.output, "tfidf_lr_pipeline.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()

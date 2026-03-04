"""
Training script for transformer (RoBERTa / DeBERTa) fine-tuning.

Usage:
    python -m src.train_transformer [--model MODEL] [--task sentence|essay]
"""

import argparse
import os

from src.config import DATASET_CONFIG, RANDOM_SEED, TRANSFORMER_CONFIG
from src.data_utils import load_and_split_dataset, set_seed
from src.evaluate import print_evaluation_report
from src.transformer_classifier import predict_transformer, train_transformer


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune transformer for CEFR")
    parser.add_argument(
        "--model",
        default=TRANSFORMER_CONFIG["sentence_model"],
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--task",
        choices=["sentence", "essay"],
        default="sentence",
        help="Classification task type",
    )
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
        default="checkpoints/transformer",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Maximum token length (default depends on task)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=TRANSFORMER_CONFIG["num_epochs"],
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=TRANSFORMER_CONFIG["batch_size"],
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=TRANSFORMER_CONFIG["learning_rate"],
        help="Learning rate",
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

    if args.max_length is None:
        args.max_length = (
            TRANSFORMER_CONFIG["max_length_sentence"]
            if args.task == "sentence"
            else TRANSFORMER_CONFIG["max_length_essay"]
        )

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

    print(f"Fine-tuning {args.model} (task={args.task}, max_length={args.max_length})")
    trainer, tokenizer = train_transformer(
        model_name=args.model,
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        output_dir=args.output,
        max_length=args.max_length,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
    )

    print("Evaluating on test set...")
    model = trainer.model
    test_preds = predict_transformer(
        model,
        tokenizer,
        test_texts,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    print_evaluation_report(
        test_labels,
        test_preds.tolist(),
        model_name=f"{args.model} ({args.task})",
    )

    final_dir = os.path.join(args.output, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Model saved to {final_dir}")


if __name__ == "__main__":
    main()

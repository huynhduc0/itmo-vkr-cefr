"""
Training script for LLM + LoRA / QLoRA for CEFR classification.

Usage:
    python -m src.train_llm [--model MODEL] [--task sentence|essay]
"""

import argparse
import os

from src.config import DATASET_CONFIG, LLM_CONFIG, RANDOM_SEED
from src.data_utils import load_and_split_dataset, set_seed
from src.evaluate import print_evaluation_report
from src.llm_lora import predict_llm, train_llm_lora


def parse_args():
    parser = argparse.ArgumentParser(description="Train LLM+LoRA for CEFR")
    parser.add_argument(
        "--model",
        default=LLM_CONFIG["base_model"],
        help="HuggingFace LLM model name",
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
        default="checkpoints/llm_lora",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=LLM_CONFIG["num_epochs"],
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=LLM_CONFIG["batch_size"],
        help="Per-device batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=LLM_CONFIG["learning_rate"],
        help="Learning rate",
    )
    parser.add_argument(
        "--no_4bit",
        action="store_true",
        help="Disable 4-bit quantization",
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

    print(f"Training LLM+LoRA ({args.model}, task={args.task})")
    trainer, model, tokenizer = train_llm_lora(
        base_model_name=args.model,
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        output_dir=args.output,
        task=args.task,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
    )

    print("Evaluating on test set...")
    test_preds = predict_llm(model, tokenizer, test_texts, task=args.task)
    valid_mask = test_preds != -1
    if valid_mask.sum() < len(test_preds):
        print(
            f"Warning: {(~valid_mask).sum()} samples had unparseable LLM output "
            f"and are excluded from metrics."
        )
    if valid_mask.sum() > 0:
        print_evaluation_report(
            [test_labels[i] for i in range(len(test_labels)) if valid_mask[i]],
            test_preds[valid_mask].tolist(),
            model_name=f"{args.model} LoRA ({args.task})",
        )

    final_dir = os.path.join(args.output, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Model saved to {final_dir}")


if __name__ == "__main__":
    main()

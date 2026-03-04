"""
Standalone data preparation script for CEFR classification (DATA_PREP.md).

Loads a UniversalCEFR English dataset, applies text normalisation, computes
token lengths with a RoBERTa tokenizer, creates sentence-level and essay-level
track splits, and saves them as JSONL files.

Usage:
    python -m src.prepare_data [OPTIONS]

Examples:
    # Prepare both tracks from the default dataset:
    python -m src.prepare_data --output data/

    # Use a different dataset (e.g. for domain transfer):
    python -m src.prepare_data \
        --dataset UniversalCEFR/cefr_sp_en \
        --output  data/cefr_sp_en/

Output structure::

    <output>/
      sentence/
        train.jsonl
        dev.jsonl
        test.jsonl
      essay/
        train.jsonl
        dev.jsonl
        test.jsonl

Each JSONL line:
    {"text": "...", "label": "B2", "n_tokens": 42}
"""

import argparse
import json
import os

from src.config import DATA_PREP_CONFIG, DATASET_CONFIG, ID2LABEL, RANDOM_SEED
from src.data_utils import (
    get_label_distribution,
    load_and_prepare_tracks,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare CEFR sentence/essay track splits and save as JSONL"
    )
    parser.add_argument(
        "--dataset",
        default=DATASET_CONFIG["dataset_name"],
        help="HuggingFace dataset name (default: %(default)s)",
    )
    parser.add_argument(
        "--text_column",
        default=DATASET_CONFIG["text_column"],
        help="Dataset field containing the text (default: %(default)s)",
    )
    parser.add_argument(
        "--label_column",
        default=DATASET_CONFIG["label_column"],
        help="Dataset field containing the CEFR label (default: %(default)s)",
    )
    parser.add_argument(
        "--tokenizer",
        default=DATA_PREP_CONFIG["tokenizer"],
        help="HuggingFace tokenizer for token counting (default: %(default)s)",
    )
    parser.add_argument(
        "--sent_min",
        type=int,
        default=DATA_PREP_CONFIG["sentence_min_tokens"],
        help="Minimum tokens for sentence track (default: %(default)s)",
    )
    parser.add_argument(
        "--sent_max",
        type=int,
        default=DATA_PREP_CONFIG["sentence_max_tokens"],
        help="Maximum tokens for sentence track (default: %(default)s)",
    )
    parser.add_argument(
        "--essay_min",
        type=int,
        default=DATA_PREP_CONFIG["essay_min_tokens"],
        help="Minimum tokens for essay track (default: %(default)s)",
    )
    parser.add_argument(
        "--min_class",
        type=int,
        default=DATA_PREP_CONFIG["min_class_samples"],
        help="Drop classes with fewer than this many samples (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default="data",
        help="Output directory for JSONL splits (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed (default: %(default)s)",
    )
    return parser.parse_args()


def _print_track_summary(
    track_name: str,
    splits,
) -> None:
    split_names = ("train", "dev", "test")
    print(f"\n  {track_name.capitalize()} track:")
    for name, (texts, labels, n_toks) in zip(split_names, splits):
        if not texts:
            print(f"    {name}: 0 samples (empty — check class-size filter)")
            continue
        dist = get_label_distribution(labels)
        dist_str = "  ".join(f"{k}:{v}" for k, v in dist.items() if v > 0)
        avg_len = sum(n_toks) / len(n_toks)
        print(
            f"    {name:5}: {len(texts):5} samples  avg_tokens={avg_len:5.1f}  [{dist_str}]"
        )


def main():
    args = parse_args()
    set_seed(args.seed)

    print(f"Dataset    : {args.dataset}")
    print(f"Tokenizer  : {args.tokenizer}")
    print(f"Sent range : {args.sent_min} – {args.sent_max} tokens")
    print(f"Essay range: ≥ {args.essay_min} tokens")
    print(f"Min class  : {args.min_class} samples")
    print(f"Output dir : {args.output}")
    print(f"Seed       : {args.seed}")

    tracks = load_and_prepare_tracks(
        dataset_name=args.dataset,
        text_column=args.text_column,
        label_column=args.label_column,
        tokenizer_name=args.tokenizer,
        sentence_min_tokens=args.sent_min,
        sentence_max_tokens=args.sent_max,
        essay_min_tokens=args.essay_min,
        min_class_samples=args.min_class,
        seed=args.seed,
        output_dir=args.output,
    )

    print("\nSplit summary:")
    for track_name, splits in tracks.items():
        _print_track_summary(track_name, splits)

    print(f"\nJSONL files written to: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()

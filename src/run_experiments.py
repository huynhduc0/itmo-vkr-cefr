"""
Unified experiment runner for CEFR level classification (Exp 0 – Exp 6).

Runs the selected experiments and prints a comparison table of:
  Accuracy | Macro-F1 | QWK | Inference latency (s/sample)

Usage:
    python -m src.run_experiments --task sentence --exps 0 1 2
    python -m src.run_experiments --task essay   --exps 0 1 2 5
    python -m src.run_experiments --task sentence --exps 6 \
        --train_dataset UniversalCEFR/cefr_sp_en \
        --eval_dataset  UniversalCEFR/cefr_sp_de   # cross-corpus
"""

import argparse
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.config import DATASET_CONFIG, RANDOM_SEED, TRANSFORMER_CONFIG
from src.data_utils import (
    load_and_split_dataset,
    load_multiple_datasets,
    remove_duplicates,
    set_seed,
    stratified_split,
)
from src.evaluate import compute_metrics, print_evaluation_report


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ExperimentResult:
    name: str
    track: str
    accuracy: float = 0.0
    macro_f1: float = 0.0
    qwk: float = 0.0
    latency: float = 0.0
    note: str = ""


# ---------------------------------------------------------------------------
# Experiment helpers
# ---------------------------------------------------------------------------

def _time_predict(predict_fn, texts: List[str]) -> Tuple[np.ndarray, float]:
    """Return (predictions, seconds_per_sample)."""
    start = time.perf_counter()
    preds = predict_fn(texts)
    elapsed = time.perf_counter() - start
    per_sample = elapsed / max(len(texts), 1)
    return preds, per_sample


# ---------------------------------------------------------------------------
# Exp 0 – Majority baseline
# ---------------------------------------------------------------------------

def run_exp0(
    train_labels: List[int],
    test_labels: List[int],
    test_size: int,
    track: str,
) -> ExperimentResult:
    from src.majority_baseline import MajorityClassifier

    clf = MajorityClassifier()
    clf.fit(train_labels)

    preds, latency = _time_predict(lambda texts: clf.predict(len(texts)), [""] * test_size)
    metrics = compute_metrics(test_labels, preds.tolist())
    return ExperimentResult(
        name="Exp 0 – Majority",
        track=track,
        latency=latency,
        note=f"majority={clf.majority_level}",
        **metrics,
    )


# ---------------------------------------------------------------------------
# Exp 1 – TF-IDF + Logistic Regression
# ---------------------------------------------------------------------------

def run_exp1(
    train_texts: List[str],
    train_labels: List[int],
    test_texts: List[str],
    test_labels: List[int],
    track: str,
) -> ExperimentResult:
    from src.baseline_tfidf import predict_baseline, train_baseline

    pipeline = train_baseline(train_texts, train_labels)
    preds, latency = _time_predict(pipeline.predict, test_texts)
    metrics = compute_metrics(test_labels, preds.tolist())
    return ExperimentResult(name="Exp 1 – TF-IDF+LR", track=track, latency=latency, **metrics)


# ---------------------------------------------------------------------------
# Exp 2 – Transformer encoder (RoBERTa)
# ---------------------------------------------------------------------------

def run_exp2(
    train_texts: List[str],
    train_labels: List[int],
    val_texts: List[str],
    val_labels: List[int],
    test_texts: List[str],
    test_labels: List[int],
    track: str,
    model_name: Optional[str] = None,
    max_length: Optional[int] = None,
    num_epochs: int = TRANSFORMER_CONFIG["num_epochs"],
    batch_size: int = TRANSFORMER_CONFIG["batch_size"],
    seed: int = RANDOM_SEED,
) -> ExperimentResult:
    from src.transformer_classifier import predict_transformer, train_transformer

    if model_name is None:
        model_name = (
            TRANSFORMER_CONFIG["sentence_model"]
            if track == "sentence"
            else TRANSFORMER_CONFIG["essay_model"]
        )
    if max_length is None:
        max_length = (
            TRANSFORMER_CONFIG["max_length_sentence"]
            if track == "sentence"
            else TRANSFORMER_CONFIG["max_length_essay"]
        )

    trainer, tokenizer = train_transformer(
        model_name=model_name,
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        max_length=max_length,
        num_epochs=num_epochs,
        batch_size=batch_size,
        seed=seed,
    )
    model = trainer.model

    def _pred(texts):
        return predict_transformer(model, tokenizer, texts, max_length=max_length, batch_size=batch_size)

    preds, latency = _time_predict(_pred, test_texts)
    metrics = compute_metrics(test_labels, preds.tolist())
    return ExperimentResult(
        name=f"Exp 2 – Transformer ({model_name})",
        track=track,
        latency=latency,
        **metrics,
    )


# ---------------------------------------------------------------------------
# Exp 3 – Ordinal classification (CORAL)
# ---------------------------------------------------------------------------

def run_exp3(
    train_texts: List[str],
    train_labels: List[int],
    val_texts: List[str],
    val_labels: List[int],
    test_texts: List[str],
    test_labels: List[int],
    track: str,
    model_name: Optional[str] = None,
    max_length: Optional[int] = None,
    num_epochs: int = TRANSFORMER_CONFIG["num_epochs"],
    batch_size: int = TRANSFORMER_CONFIG["batch_size"],
    seed: int = RANDOM_SEED,
) -> ExperimentResult:
    from src.ordinal_classifier import predict_ordinal, train_ordinal

    if model_name is None:
        model_name = TRANSFORMER_CONFIG["sentence_model"]
    if max_length is None:
        max_length = (
            TRANSFORMER_CONFIG["max_length_sentence"]
            if track == "sentence"
            else TRANSFORMER_CONFIG["max_length_essay"]
        )

    model, tokenizer = train_ordinal(
        model_name=model_name,
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        max_length=max_length,
        num_epochs=num_epochs,
        batch_size=batch_size,
        seed=seed,
    )

    def _pred(texts):
        return predict_ordinal(model, tokenizer, texts, max_length=max_length, batch_size=batch_size)

    preds, latency = _time_predict(_pred, test_texts)
    metrics = compute_metrics(test_labels, preds.tolist())
    return ExperimentResult(
        name=f"Exp 3 – Ordinal CORAL ({model_name})",
        track=track,
        latency=latency,
        **metrics,
    )


# ---------------------------------------------------------------------------
# Exp 4 – LLM + LoRA / QLoRA
# ---------------------------------------------------------------------------

def run_exp4(
    train_texts: List[str],
    train_labels: List[int],
    val_texts: List[str],
    val_labels: List[int],
    test_texts: List[str],
    test_labels: List[int],
    track: str,
    seed: int = RANDOM_SEED,
) -> ExperimentResult:
    from src.llm_lora import predict_llm, train_llm_lora

    trainer, model, tokenizer = train_llm_lora(
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        task=track,
        seed=seed,
    )

    def _pred(texts):
        return predict_llm(model, tokenizer, texts, task=track)

    preds, latency = _time_predict(_pred, test_texts)
    valid_mask = preds != -1
    if valid_mask.sum() == 0:
        return ExperimentResult(
            name="Exp 4 – LLM+LoRA", track=track, note="no valid predictions"
        )
    filtered_true = [test_labels[i] for i in range(len(test_labels)) if valid_mask[i]]
    metrics = compute_metrics(filtered_true, preds[valid_mask].tolist())
    return ExperimentResult(
        name="Exp 4 – LLM+LoRA",
        track=track,
        latency=latency,
        note=f"{valid_mask.sum()}/{len(test_texts)} valid",
        **metrics,
    )


# ---------------------------------------------------------------------------
# Exp 5 – Hybrid long-text (sentence classifier + aggregation)
# ---------------------------------------------------------------------------

def run_exp5(
    train_texts: List[str],
    train_labels: List[int],
    test_texts: List[str],
    test_labels: List[int],
    track: str = "essay",
    aggregation: str = "mean_prob",
    seed: int = RANDOM_SEED,
) -> ExperimentResult:
    """
    Train a sentence-level classifier (TF-IDF+LR for speed), then apply
    it with per-sentence aggregation to essay-level test texts.
    """
    from src.baseline_tfidf import train_baseline
    from src.hybrid_essay import HybridEssayClassifier

    pipeline = train_baseline(train_texts, train_labels)
    num_labels = len(set(train_labels + test_labels))

    def _predict_fn(texts):
        return pipeline.predict(texts)

    def _predict_proba_fn(texts):
        return pipeline.predict_proba(texts)

    hybrid = HybridEssayClassifier(
        predict_fn=_predict_fn,
        predict_proba_fn=_predict_proba_fn,
        aggregation=aggregation,
    )

    preds, latency = _time_predict(hybrid.predict, test_texts)
    metrics = compute_metrics(test_labels, preds.tolist())
    return ExperimentResult(
        name=f"Exp 5 – Hybrid ({aggregation})",
        track=track,
        latency=latency,
        **metrics,
    )


# ---------------------------------------------------------------------------
# Exp 6 – Domain transfer
# ---------------------------------------------------------------------------

def run_exp6(
    train_dataset: str,
    eval_dataset: str,
    text_column: str = DATASET_CONFIG["text_column"],
    label_column: str = DATASET_CONFIG["label_column"],
    track: str = "sentence",
    seed: int = RANDOM_SEED,
) -> ExperimentResult:
    """
    Train on train_dataset, evaluate on eval_dataset (cross-corpus transfer).
    Uses TF-IDF+LR for speed; swap in any predictor as needed.
    """
    from src.baseline_tfidf import predict_baseline, train_baseline
    from src.data_utils import load_dataset

    print(f"Loading train corpus: {train_dataset}")
    train_texts, train_labels = load_dataset(
        dataset_name=train_dataset,
        text_column=text_column,
        label_column=label_column,
    )
    train_texts, train_labels = remove_duplicates(train_texts, train_labels)

    print(f"Loading eval corpus: {eval_dataset}")
    eval_texts, eval_labels = load_dataset(
        dataset_name=eval_dataset,
        text_column=text_column,
        label_column=label_column,
    )
    eval_texts, eval_labels = remove_duplicates(eval_texts, eval_labels)

    pipeline = train_baseline(train_texts, train_labels)

    preds, latency = _time_predict(pipeline.predict, eval_texts)
    metrics = compute_metrics(eval_labels, preds.tolist())
    return ExperimentResult(
        name="Exp 6 – Domain Transfer (TF-IDF+LR)",
        track=track,
        latency=latency,
        note=f"train={train_dataset} → eval={eval_dataset}",
        **metrics,
    )


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def print_comparison_table(results: List[ExperimentResult]) -> None:
    """Print a formatted comparison table of all experiment results."""
    header = (
        f"{'Experiment':<45} {'Track':<10} "
        f"{'Acc':>6} {'F1':>6} {'QWK':>6} {'Lat(ms)':>9} {'Note'}"
    )
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for r in results:
        print(
            f"{r.name:<45} {r.track:<10} "
            f"{r.accuracy:>6.4f} {r.macro_f1:>6.4f} {r.qwk:>6.4f} "
            f"{r.latency * 1000:>8.2f}ms  {r.note}"
        )
    print(sep)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Run CEFR classification experiments")
    parser.add_argument(
        "--task",
        choices=["sentence", "essay"],
        default="sentence",
        help="Classification track",
    )
    parser.add_argument(
        "--exps",
        nargs="+",
        type=int,
        default=[0, 1],
        help="Experiment ids to run (0–6)",
    )
    parser.add_argument(
        "--dataset",
        default=DATASET_CONFIG["dataset_name"],
        help="Primary HuggingFace dataset",
    )
    parser.add_argument(
        "--train_dataset",
        default=None,
        help="Train dataset for Exp 6 (domain transfer)",
    )
    parser.add_argument(
        "--eval_dataset",
        default=None,
        help="Eval dataset for Exp 6 (domain transfer)",
    )
    parser.add_argument(
        "--text_column",
        default=DATASET_CONFIG["text_column"],
    )
    parser.add_argument(
        "--label_column",
        default=DATASET_CONFIG["label_column"],
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=TRANSFORMER_CONFIG["num_epochs"],
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=TRANSFORMER_CONFIG["batch_size"],
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # Load data (shared across Exp 0–5)
    if any(e in args.exps for e in [0, 1, 2, 3, 4, 5]):
        print(f"Loading dataset: {args.dataset}")
        (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels) = (
            load_and_split_dataset(
                dataset_name=args.dataset,
                text_column=args.text_column,
                label_column=args.label_column,
                seed=args.seed,
                deduplicate=True,
            )
        )
        print(
            f"Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}"
        )

    results: List[ExperimentResult] = []

    if 0 in args.exps:
        print("\n--- Exp 0: Majority baseline ---")
        r = run_exp0(train_labels, test_labels, len(test_texts), track=args.task)
        results.append(r)
        print(f"  Majority class: {r.note}, Accuracy: {r.accuracy:.4f}")

    if 1 in args.exps:
        print("\n--- Exp 1: TF-IDF + LR ---")
        r = run_exp1(train_texts, train_labels, test_texts, test_labels, track=args.task)
        results.append(r)

    if 2 in args.exps:
        print("\n--- Exp 2: Transformer (RoBERTa) ---")
        r = run_exp2(
            train_texts, train_labels,
            val_texts, val_labels,
            test_texts, test_labels,
            track=args.task,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        results.append(r)

    if 3 in args.exps:
        print("\n--- Exp 3: Ordinal CORAL ---")
        r = run_exp3(
            train_texts, train_labels,
            val_texts, val_labels,
            test_texts, test_labels,
            track=args.task,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        results.append(r)

    if 4 in args.exps:
        print("\n--- Exp 4: LLM + LoRA ---")
        r = run_exp4(
            train_texts, train_labels,
            val_texts, val_labels,
            test_texts, test_labels,
            track=args.task,
            seed=args.seed,
        )
        results.append(r)

    if 5 in args.exps:
        print("\n--- Exp 5: Hybrid essay (sentence agg) ---")
        r = run_exp5(
            train_texts, train_labels,
            test_texts, test_labels,
            track=args.task,
            seed=args.seed,
        )
        results.append(r)

    if 6 in args.exps:
        print("\n--- Exp 6: Domain transfer ---")
        train_ds = args.train_dataset or args.dataset
        eval_ds = args.eval_dataset or args.dataset
        r = run_exp6(
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            text_column=args.text_column,
            label_column=args.label_column,
            track=args.task,
            seed=args.seed,
        )
        results.append(r)

    print_comparison_table(results)


if __name__ == "__main__":
    main()

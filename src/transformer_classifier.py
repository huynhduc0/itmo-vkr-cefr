"""
Transformer encoder (RoBERTa / DeBERTa) fine-tuning for CEFR classification.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from src.config import ID2LABEL, LABEL2ID, RANDOM_SEED, TRANSFORMER_CONFIG
from src.evaluate import compute_metrics


class CEFRTransformerDataset:
    """PyTorch dataset for CEFR classification with a tokenizer."""

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer,
        max_length: int = 128,
    ):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict:
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def build_transformer_model(
    model_name: str,
    num_labels: int = 6,
):
    """
    Load a pretrained transformer model for sequence classification.
    """
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    return model


def get_training_args(
    output_dir: str,
    num_epochs: int = TRANSFORMER_CONFIG["num_epochs"],
    batch_size: int = TRANSFORMER_CONFIG["batch_size"],
    learning_rate: float = TRANSFORMER_CONFIG["learning_rate"],
    weight_decay: float = TRANSFORMER_CONFIG["weight_decay"],
    warmup_ratio: float = TRANSFORMER_CONFIG["warmup_ratio"],
    seed: int = RANDOM_SEED,
):
    """Build HuggingFace TrainingArguments.

    Mixed precision is auto-detected from the GPU:
    - sm_80+ (A100, H100): bf16=True  (native, fastest)
    - sm_60–sm_75 (P100, T4): fp16=True  (AMP via GradScaler)
    - CPU: no mixed precision

    IMPORTANT: Do NOT set the ACCELERATE_MIXED_PRECISION env var alongside
    fp16/bf16 in TrainingArguments. The env var causes Accelerate to cast all
    model parameters to fp16, after which GradScaler sees fp16 params/grads
    and raises "Attempting to unscale FP16 gradients".
    TrainingArguments(fp16=True) handles AMP correctly: params stay fp32 and
    only the forward pass runs under fp16 autocast.
    """
    import torch
    from transformers import TrainingArguments

    # DeBERTa-v3 uses weight-tied embeddings whose gradients come out as fp16
    # even in standard AMP, causing GradScaler to raise
    # "Attempting to unscale FP16 gradients". Run in fp32 — DeBERTa-v3-base
    # (184 M params) is fast enough on T4/P100 without mixed precision.
    use_fp16 = False
    use_bf16 = False

    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_qwk",
        greater_is_better=True,
        seed=seed,
        logging_steps=50,
        fp16=use_fp16,
        bf16=use_bf16,
    )


def compute_transformer_metrics(eval_pred) -> Dict[str, float]:
    """Compute report-aligned metrics during transformer training."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return compute_metrics(labels.tolist(), predictions.tolist())


def train_transformer(
    model_name: str,
    train_texts: List[str],
    train_labels: List[int],
    val_texts: List[str],
    val_labels: List[int],
    output_dir: str = "checkpoints/transformer",
    max_length: int = TRANSFORMER_CONFIG["max_length_sentence"],
    num_epochs: int = TRANSFORMER_CONFIG["num_epochs"],
    batch_size: int = TRANSFORMER_CONFIG["batch_size"],
    learning_rate: float = TRANSFORMER_CONFIG["learning_rate"],
    seed: int = RANDOM_SEED,
) -> Tuple:
    """
    Fine-tune a transformer model for CEFR classification.

    Returns:
        (trainer, tokenizer)
    """
    import torch
    from transformers import AutoTokenizer, Trainer

    from src.data_utils import set_seed

    set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = build_transformer_model(model_name)

    train_dataset = CEFRTransformerDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = CEFRTransformerDataset(val_texts, val_labels, tokenizer, max_length)

    training_args = get_training_args(
        output_dir=output_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_transformer_metrics,
    )
    trainer.train()
    return trainer, tokenizer


def predict_transformer(
    model,
    tokenizer,
    texts: List[str],
    max_length: int = TRANSFORMER_CONFIG["max_length_sentence"],
    batch_size: int = TRANSFORMER_CONFIG["batch_size"],
) -> np.ndarray:
    """
    Run inference with a fine-tuned transformer.

    Returns:
        Array of predicted label ids.
    """
    import torch

    model.eval()
    all_preds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        all_preds.extend(preds.tolist())
    return np.array(all_preds)

"""
Transformer encoder (RoBERTa / DeBERTa) fine-tuning for CEFR classification.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.config import ID2LABEL, LABEL2ID, RANDOM_SEED, TRANSFORMER_CONFIG


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
    """Build HuggingFace TrainingArguments."""
    from transformers import TrainingArguments

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
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        seed=seed,
        logging_steps=50,
        fp16=False,
    )


def compute_transformer_metrics(eval_pred) -> Dict[str, float]:
    """Compute accuracy and macro-F1 during transformer training."""
    from sklearn.metrics import accuracy_score, f1_score

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, average="macro", zero_division=0)
    return {"accuracy": accuracy, "macro_f1": macro_f1}


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

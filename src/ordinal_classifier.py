"""
Ordinal classification for CEFR levels using a CORAL-style cumulative link head
on top of a transformer encoder (Exp 3).

Reference: Cao et al., "Rank Consistent Ordinal Regression for Neural Networks",
           Pattern Recognition Letters, 2020.

The CORAL head applies K-1 binary output neurons that share the same weight
vector but have independent bias terms, then sums the threshold exceedances to
obtain the predicted ordinal label.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from src.config import ID2LABEL, LABEL2ID, RANDOM_SEED, TRANSFORMER_CONFIG

NUM_LABELS = len(LABEL2ID)


# ---------------------------------------------------------------------------
# PyTorch model
# ---------------------------------------------------------------------------

def _build_ordinal_model(model_name: str, num_labels: int = NUM_LABELS):
    """
    Build a transformer encoder with a CORAL ordinal regression head.

    The backbone is any AutoModel-compatible encoder.  On top of the [CLS]
    pooled representation we add a linear layer with (num_labels - 1) outputs
    that share a weight column but have independent biases.
    """
    import torch
    import torch.nn as nn
    from transformers import AutoModel

    class CoralOrdinalModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(model_name)
            hidden_size = self.encoder.config.hidden_size
            self.fc = nn.Linear(hidden_size, 1, bias=False)
            self.bias = nn.Parameter(torch.zeros(num_labels - 1))

        def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            pooled = outputs.last_hidden_state[:, 0, :]
            logits = self.fc(pooled) + self.bias
            loss = None
            if labels is not None:
                loss = coral_loss(logits, labels, num_labels)
            return (loss, logits) if loss is not None else logits

    return CoralOrdinalModel()


def coral_loss(logits, labels, num_labels: int = NUM_LABELS):
    """
    Compute CORAL loss: sum of binary cross-entropies over K-1 thresholds.

    Args:
        logits: (batch, K-1) raw logits
        labels: (batch,) integer class labels 0..K-1
        num_labels: total number of CEFR classes K

    Returns:
        Scalar loss.
    """
    import torch
    import torch.nn.functional as F

    # Build binary targets: targets[i, k] = 1 if labels[i] > k else 0
    targets = torch.zeros_like(logits)
    for k in range(num_labels - 1):
        targets[:, k] = (labels > k).float()
    return F.binary_cross_entropy_with_logits(logits, targets)


def coral_predict(logits) -> np.ndarray:
    """
    Convert CORAL logits to ordinal label predictions.

    Prediction = number of thresholds for which sigmoid(logit) >= 0.5

    Works with both numpy arrays and torch tensors.
    """
    if isinstance(logits, np.ndarray):
        probs = 1.0 / (1.0 + np.exp(-logits))
    else:
        import torch
        probs = torch.sigmoid(logits).detach().cpu().numpy()
    return (probs >= 0.5).sum(axis=1).astype(int)


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------

class OrdinalDataset:
    """PyTorch dataset for ordinal CEFR classification."""

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int):
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
        import torch

        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_ordinal(
    model_name: str,
    train_texts: List[str],
    train_labels: List[int],
    val_texts: List[str],
    val_labels: List[int],
    output_dir: str = "checkpoints/ordinal",
    max_length: int = TRANSFORMER_CONFIG["max_length_sentence"],
    num_epochs: int = TRANSFORMER_CONFIG["num_epochs"],
    batch_size: int = TRANSFORMER_CONFIG["batch_size"],
    learning_rate: float = TRANSFORMER_CONFIG["learning_rate"],
    seed: int = RANDOM_SEED,
) -> Tuple:
    """
    Fine-tune a transformer with a CORAL ordinal head for CEFR classification.

    Returns:
        (model, tokenizer)
    """
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    from src.data_utils import set_seed

    set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = _build_ordinal_model(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_dataset = OrdinalDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = OrdinalDataset(val_texts, val_labels, tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_val_acc = -1.0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            loss, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        # Validation
        val_preds = _predict_ordinal_batched(model, val_loader, device)
        val_acc = float((np.array(val_preds) == np.array(val_labels)).mean())
        avg_loss = total_loss / max(len(train_loader), 1)
        print(
            f"Epoch {epoch + 1}/{num_epochs}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            import os
            os.makedirs(output_dir, exist_ok=True)
            torch.save(model.state_dict(), f"{output_dir}/best_model.pt")

    # Restore best
    model.load_state_dict(torch.load(f"{output_dir}/best_model.pt", map_location=device))
    return model, tokenizer


def _predict_ordinal_batched(model, data_loader, device) -> List[int]:
    """Run batched inference and return predicted label ids."""
    import torch

    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            preds = coral_predict(logits)
            all_preds.extend(preds.tolist())
    return all_preds


def predict_ordinal(
    model,
    tokenizer,
    texts: List[str],
    max_length: int = TRANSFORMER_CONFIG["max_length_sentence"],
    batch_size: int = TRANSFORMER_CONFIG["batch_size"],
) -> np.ndarray:
    """
    Run inference using a trained CORAL ordinal model.

    Returns:
        Array of predicted label ids.
    """
    import torch
    from torch.utils.data import DataLoader

    dataset = OrdinalDataset(texts, [0] * len(texts), tokenizer, max_length)
    loader = DataLoader(dataset, batch_size=batch_size)
    device = next(model.parameters()).device
    return np.array(_predict_ordinal_batched(model, loader, device))

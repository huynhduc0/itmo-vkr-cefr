"""
LLM + LoRA / QLoRA training for CEFR classification.
"""

import re
from typing import List, Optional, Tuple

import numpy as np

from src.config import (
    CEFR_LEVELS,
    ESSAY_PROMPT,
    LABEL2ID,
    LLM_CONFIG,
    RANDOM_SEED,
    SENTENCE_PROMPT,
)


def build_bnb_config(
    use_4bit: bool = LLM_CONFIG["use_4bit"],
    bnb_4bit_quant_type: str = LLM_CONFIG["bnb_4bit_quant_type"],
    bnb_4bit_compute_dtype: str = LLM_CONFIG["bnb_4bit_compute_dtype"],
):
    """Build BitsAndBytesConfig for 4-bit quantization."""
    import torch
    from transformers import BitsAndBytesConfig

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    compute_dtype = dtype_map.get(bnb_4bit_compute_dtype, torch.float16)
    return BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def load_llm(
    base_model_name: str = LLM_CONFIG["base_model"],
    use_4bit: bool = LLM_CONFIG["use_4bit"],
):
    """
    Load an LLM with optional 4-bit quantization.

    Returns:
        (model, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_4bit:
        bnb_config = build_bnb_config()
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
        )
    return model, tokenizer


def build_lora_config(
    r: int = LLM_CONFIG["lora_r"],
    lora_alpha: int = LLM_CONFIG["lora_alpha"],
    lora_dropout: float = LLM_CONFIG["lora_dropout"],
    target_modules: Optional[List[str]] = None,
):
    """Build LoRA configuration."""
    from peft import LoraConfig

    if target_modules is None:
        target_modules = LLM_CONFIG["lora_target_modules"]

    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        task_type="CAUSAL_LM",
    )


def apply_lora(model, lora_config=None):
    """Apply LoRA adapters to the model."""
    from peft import get_peft_model, prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(model)
    if lora_config is None:
        lora_config = build_lora_config()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def format_prompt(
    text: str,
    task: str = "sentence",
) -> str:
    """
    Format a classification prompt for the LLM.

    Args:
        text: input sentence or essay
        task: 'sentence' or 'essay'

    Returns:
        Formatted prompt string.
    """
    if task == "sentence":
        return SENTENCE_PROMPT.format(text=text)
    return ESSAY_PROMPT.format(text=text)


def format_sft_example(
    text: str,
    label: int,
    task: str = "sentence",
) -> str:
    """
    Format a supervised fine-tuning example as instruction + answer.

    Args:
        text: input text
        label: integer label id
        task: 'sentence' or 'essay'

    Returns:
        Full training string: prompt + '\n' + CEFR level.
    """
    from src.config import ID2LABEL

    prompt = format_prompt(text, task)
    answer = ID2LABEL[label]
    return f"{prompt}\n{answer}"


def extract_predicted_label(output_text: str) -> Optional[str]:
    """
    Post-process LLM output to extract CEFR level label.

    Returns:
        CEFR label string (e.g. 'B2') or None if not found.
    """
    for level in CEFR_LEVELS:
        pattern = rf"\b{level}\b"
        if re.search(pattern, output_text, re.IGNORECASE):
            return level
    return None


def build_sft_dataset(
    texts: List[str],
    labels: List[int],
    tokenizer,
    task: str = "sentence",
    max_length: int = LLM_CONFIG["max_length"],
):
    """
    Build a HuggingFace Dataset suitable for SFT training.
    """
    import torch
    from torch.utils.data import Dataset

    class SFTDataset(Dataset):
        def __init__(self):
            self.examples = [
                format_sft_example(t, l, task) for t, l in zip(texts, labels)
            ]
            self.encodings = tokenizer(
                self.examples,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, idx):
            item = {k: v[idx] for k, v in self.encodings.items()}
            item["labels"] = item["input_ids"].clone()
            return item

    return SFTDataset()


def train_llm_lora(
    base_model_name: str = LLM_CONFIG["base_model"],
    train_texts: List[str] = None,
    train_labels: List[int] = None,
    val_texts: List[str] = None,
    val_labels: List[int] = None,
    output_dir: str = "checkpoints/llm_lora",
    task: str = "sentence",
    num_epochs: int = LLM_CONFIG["num_epochs"],
    batch_size: int = LLM_CONFIG["batch_size"],
    learning_rate: float = LLM_CONFIG["learning_rate"],
    seed: int = RANDOM_SEED,
):
    """
    Train an LLM with LoRA/QLoRA for CEFR classification via SFT.

    Returns:
        (trainer, model, tokenizer)
    """
    from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

    from src.data_utils import set_seed

    set_seed(seed)

    model, tokenizer = load_llm(base_model_name)
    model = apply_lora(model)

    train_dataset = build_sft_dataset(
        train_texts, train_labels, tokenizer, task=task
    )
    val_dataset = (
        build_sft_dataset(val_texts, val_labels, tokenizer, task=task)
        if val_texts
        else None
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        eval_strategy="epoch" if val_dataset else "no",
        save_strategy="epoch",
        seed=seed,
        logging_steps=50,
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    trainer.train()
    return trainer, model, tokenizer


def predict_llm(
    model,
    tokenizer,
    texts: List[str],
    task: str = "sentence",
    max_new_tokens: int = 10,
) -> np.ndarray:
    """
    Generate CEFR predictions using an LLM.

    Returns:
        Array of predicted label ids (-1 if label not extracted).
    """
    import torch

    model.eval()
    predictions = []
    for text in texts:
        prompt = format_prompt(text, task)
        inputs = tokenizer(prompt, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        label = extract_predicted_label(output_text)
        predictions.append(LABEL2ID.get(label, -1))
    return np.array(predictions)

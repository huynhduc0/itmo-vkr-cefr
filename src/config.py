"""
Configuration for CEFR level classification.
"""

LABEL2ID = {
    "A1": 0,
    "A2": 1,
    "B1": 2,
    "B2": 3,
    "C1": 4,
    "C2": 5,
}

ID2LABEL = {v: k for k, v in LABEL2ID.items()}

CEFR_LEVELS = list(LABEL2ID.keys())

RANDOM_SEED = 42

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Data preparation configuration (DATA_PREP.md)
DATA_PREP_CONFIG = {
    "tokenizer": "roberta-base",
    # Sentence-level track: 5 ≤ n_tokens ≤ 64
    "sentence_min_tokens": 5,
    "sentence_max_tokens": 64,
    # Essay-level track: n_tokens ≥ 128  (65–127 excluded to avoid overlap)
    "essay_min_tokens": 128,
    # Drop CEFR classes with fewer than this many samples (per track)
    "min_class_samples": 100,
}

# TF-IDF baseline hyperparameters
TFIDF_CONFIG = {
    "word_ngram_range": (1, 2),
    "char_ngram_range": (3, 5),
    "max_features": 50000,
    "lr_max_iter": 2000,
    "lr_C": 1.0,
}

# Transformer fine-tuning hyperparameters
TRANSFORMER_CONFIG = {
    "sentence_model": "roberta-base",
    "essay_model": "roberta-base",
    "max_length_sentence": 128,
    "max_length_essay": 512,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "num_epochs": 5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
}

# LLM + LoRA hyperparameters
LLM_CONFIG = {
    "base_model": "meta-llama/Llama-3.2-3B-Instruct",
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "max_length": 512,
    "batch_size": 4,
    "learning_rate": 2e-4,
    "num_epochs": 3,
    "use_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": "float16",
}

# Dataset configuration
DATASET_CONFIG = {
    "dataset_name": "UniversalCEFR/cefr_sp_en",
    "text_column": "text",
    "label_column": "cefr_level",
}

# Prompt templates
SENTENCE_PROMPT = (
    "Classify the CEFR level of the following English sentence.\n\n"
    "Sentence:\n{text}\n\n"
    "Answer with one label from: A1, A2, B1, B2, C1, C2."
)

ESSAY_PROMPT = (
    "Classify the CEFR level of the following English learner text.\n\n"
    "Text:\n{text}\n\n"
    "Return only one label: A1, A2, B1, B2, C1, C2."
)

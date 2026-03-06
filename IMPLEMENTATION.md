# CEFR Level Classification (Sentence-level & Essay-level)

## 1. Task definition

The goal is to build models for automatic classification of learner texts into CEFR levels (A1, A2, B1, B2, C1, C2).

Two tasks are considered:
- Sentence-level CEFR classification
- Essay-level CEFR classification

Both tasks are treated as:
- multi-class classification
- ordinal classification (adjacent levels are closer than distant ones)

## 2. Dataset

Public dataset: `UniversalCEFR` (English subsets only).

Example datasets:
- `UniversalCEFR/cefr_sp_en`
- Other English subcorpora from the UniversalCEFR collection

Fields used:
- `text`
- `cefr_level`

Preprocessing steps:
- remove empty samples
- normalize labels to `{A1,A2,B1,B2,C1,C2}`
- remove samples with unknown labels
- stratified train / validation / test split

## 3. Environment

```bash
python>=3.10
pip install datasets transformers accelerate peft bitsandbytes
pip install scikit-learn evaluate torch
```

## 4. Label encoding

```python
label2id = {
    "A1": 0,
    "A2": 1,
    "B1": 2,
    "B2": 3,
    "C1": 4,
    "C2": 5
}
id2label = {v:k for k,v in label2id.items()}
```

## 5. Baseline: TF-IDF + Logistic Regression

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(max_iter=2000)
```

Use word n-grams (1–2) and character n-grams (3–5).

## 6. Transformer encoder baseline

**Model**:
- `roberta-base` (sentence)
- `roberta-base` or `deberta-v3-base` (essay, with truncation)

**Training**:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=6,
    id2label=id2label,
    label2id=label2id
)
```

## 7. LLM + LoRA / QLoRA

**Model Example**:
- `meta-llama/Llama-3.2-3B-Instruct`

Loading with 4-bit quantization:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16"
)

model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="auto"
)
```

## 8. LoRA configuration

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)
```

## 9. Prompt format for LLM classification

### Sentence-level

```text
Classify the CEFR level of the following English sentence.

Sentence:
{TEXT}

Answer with one label from: A1, A2, B1, B2, C1, C2.
```

### Essay-level

```text
Classify the CEFR level of the following English learner text.

Text:
{TEXT}

Return only one label: A1, A2, B1, B2, C1, C2.
```

The output is post-processed to extract the predicted label.

## 10. Training with SFT (LLM)

Each training example is converted into an instruction–answer pair:
- **input**: classification prompt
- **target**: CEFR label

## 11. Evaluation

Metrics:
- Accuracy
- Macro-F1
- Quadratic Weighted Kappa

```python
from sklearn.metrics import f1_score, cohen_kappa_score
qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
```

## 12. Error analysis

- Confusion matrix
- Adjacent-level confusion analysis (A2↔B1, B1↔B2, B2↔C1)

## 13. Sentence vs Essay comparison

Two independent evaluation tracks:
- Short texts (sentence-level subsets)
- Long texts (essay-level subsets)

Performance and inference latency are compared separately.

## 14. Reproducibility

- Fixed random seeds
- Full list of hyperparameters
- Saved model checkpoints
- Versioned datasets from HuggingFace
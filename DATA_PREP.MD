# 📄 DATA_PREP.md

UniversalCEFR → Sentence-level & Essay-level splits

## 1. Dataset source

Source: [https://huggingface.co/UniversalCEFR](https://huggingface.co/UniversalCEFR)

We use only English subcorpora.

Main working dataset: `UniversalCEFR/cefr_sp_en`

Additional English subcorpora may be used for domain transfer experiments.

## 2. Expected raw fields

After loading:

```python
from datasets import load_dataset

ds = load_dataset("UniversalCEFR/cefr_sp_en")
```

Expected minimal fields:
- `text` (string)
- `cefr_level` (string)

## 3. Label normalization

Accepted labels: `A1, A2, B1, B2, C1, C2`

Mapping rules:
- keep only these labels
- drop samples with missing or non-standard labels

## 4. Text normalization

Apply:
- strip spaces
- replace multiple whitespaces
- keep original casing (do NOT lowercase)

No grammar correction or spell normalization is applied.

## 5. Length-based split (sentence vs essay)

We create two tracks from the same dataset using token length.

Tokenizer: `roberta-base` tokenizer

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("roberta-base")
```

For each text:

```python
n_tokens = len(tok(text)["input_ids"])
```

### Sentence-level subset

Keep samples where: `5 <= n_tokens <= 64`

This range filters out:
- very short fragments
- long learner texts

### Essay-level subset

Keep samples where: `n_tokens >= 128`

Samples with `65 <= n_tokens < 128` are excluded to avoid overlap.

Rationale (for thesis):
- sentence-level texts represent local proficiency evidence
- essay-level texts represent global writing proficiency

## 6. Duplicate removal

Deduplicate by: `(text, cefr_level)`

## 7. Stratified split

Perform split separately for:
- sentence-level subset
- essay-level subset

Split:
- train: 80%
- validation: 10%
- test: 10%

Stratification key: `cefr_level`

## 8. Minimum class size constraint

After filtering:
- drop CEFR classes with fewer than 100 samples (applied independently for sentence and essay tracks)

If a class is dropped in one track, it is kept in the other track.

## 9. Reproducibility

- random seed = 42
- save final splits as JSONL:
  - `train.jsonl`
  - `dev.jsonl`
  - `test.jsonl`

Fields per line:
- `text`
- `label`
- `n_tokens`

## 10. Reference implementation (minimal)

```python
from datasets import load_dataset
from transformers import AutoTokenizer
from collections import defaultdict
import random

random.seed(42)

ds = load_dataset("UniversalCEFR/cefr_sp_en", split="train")

tok = AutoTokenizer.from_pretrained("roberta-base")

samples = []

for ex in ds:
    text = ex["text"].strip()
    label = ex["cefr_level"]

    if label not in {"A1","A2","B1","B2","C1","C2"}:
        continue

    n_tokens = len(tok(text)["input_ids"])

    samples.append({
        "text": text,
        "label": label,
        "n_tokens": n_tokens
    })
```

Filtering and stratified splitting are performed afterwards according to Sections 5–7.

## 11. Important note for the thesis

In the experimental section, explicitly state:

> Sentence-level and essay-level datasets are constructed from the same CEFR-annotated corpus using token-length-based filtering, in order to enable controlled comparison between short-text and long-text CEFR classification.
# EXPERIMENT_PLAN.md

CEFR Level Classification – Sentence-level & Essay-level

## 1. Experimental goals

The goal of the experiments is to compare different modeling paradigms for automatic CEFR level classification:
- sentence-level texts
- essay-level texts

The experiments evaluate:
- classification quality,
- robustness across adjacent CEFR levels,
- training and inference efficiency.

## 2. Datasets

Public datasets from the UniversalCEFR collection (English only).

Two subsets are formed:

### 2.1 Sentence-level subset

Texts with short length (single sentences or short learner utterances).

### 2.2 Essay-level subset

Long learner texts (multi-sentence learner productions).

Preprocessing:
- label normalization to `{A1,A2,B1,B2,C1,C2}`
- stratified splitting
- duplicate removal

## 3. Metrics

Primary metrics:
- Accuracy
- Macro-F1
- Quadratic Weighted Kappa (QWK)

Additional analysis:
- confusion matrix
- adjacent-level confusion rate

## 4. Experimental setup

All experiments are conducted separately for:
- sentence-level track
- essay-level track

The same label space is used in all experiments.

## 5. Experiments

### Exp 0 — Majority baseline
- **Predict**: the most frequent CEFR class.
- **Purpose**: sanity check baseline.

### Exp 1 — TF-IDF + Logistic Regression
- **Features**: word n-grams (1–2), character n-grams (3–5).
- **Purpose**: classical baseline.

### Exp 2 — Transformer encoder baseline
- **Model**: `roberta-base`
- **Training**: full fine-tuning
- **Tracks**: sentence-level, essay-level (with truncation)
- **Purpose**: strong neural baseline.

### Exp 3 — Ordinal classification variant
- **Base model**: `roberta-base`
- **Method**: ordinal regression formulation using cumulative link approach (K−1 binary classifiers or ordinal head).
- **Purpose**: explicit modeling of CEFR ordering.

### Exp 4 — LLM + LoRA (instruction-based classification)
- **Base model**: `LLaMA-3.2-3B-Instruct`
- **Method**: QLoRA fine-tuning using instruction–answer pairs.
- **Tracks**: sentence-level, essay-level
- **Purpose**: evaluate LLMs under parameter-efficient fine-tuning.

### Exp 5 — Hybrid long-text strategy
- **Model**: sentence-level classifier + aggregation for essay-level prediction.
- **Method**:
  - split essay into sentences
  - classify each sentence
  - aggregate predictions (mean / weighted vote)
- **Purpose**: compare direct essay classification vs sentence aggregation.

### Exp 6 — Domain transfer experiment
- **Method**: Train on one UniversalCEFR English subcorpus and evaluate on another one.
- **Purpose**: evaluate cross-corpus robustness.

### Exp 7 — TF-IDF + Linear SVM
- **Features**: word n-grams (1–2), character n-grams (3–5).
- **Classifier**: LinearSVC.
- **Purpose**: margin-based classical baseline for sparse text features.

### Exp 8 — TF-IDF + Complement Naive Bayes
- **Features**: word n-grams (1–2), character n-grams (3–5).
- **Classifier**: ComplementNB.
- **Purpose**: fast probabilistic baseline, often strong on imbalanced text classes.

### Exp 9 — Word-only TF-IDF + Logistic Regression
- **Features**: word n-grams only (1–2), no character features.
- **Classifier**: Logistic Regression.
- **Purpose**: ablation against Exp 1 to measure impact of char n-grams.

### Exp 10 — Ensemble (LR + ComplementNB)
- **Method**: train Exp 1-style LR and Exp 8-style ComplementNB, then average probabilities.
- **Purpose**: lightweight CPU ensemble to improve robustness.

## 6. Hyperparameter search

For Exp 2–Exp 4:
- learning rate search
- batch size search
- LoRA rank and alpha search (for Exp 4)

## 7. Reporting

For each experiment and each track:
- Accuracy
- Macro-F1
- QWK
- inference latency
- confusion matrix

A unified comparison table is produced.

## 8. Expected outcome

The experiments are expected to show:
- performance gap between classical and neural approaches,
- the benefit of ordinal modeling for CEFR levels,
- the effectiveness of QLoRA-based LLM fine-tuning for long-text CEFR classification.

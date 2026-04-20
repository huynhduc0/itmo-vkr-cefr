# itmo-vkr-cefr

Dự án nghiên cứu về phân loại trình độ CEFR đa ngôn ngữ (`en`, `ru`, `it`, `es`, `de`, `fr`).
Bao gồm pipeline hoàn chỉnh: chuẩn bị dữ liệu → huấn luyện → đánh giá → lưu kết quả.

---

## Kiến trúc tổng quan

```
HuggingFace Dataset
        │
        ▼
src/prepare_data.py          ← Tải & chuẩn hoá dữ liệu, tạo splits JSONL
        │
        ▼  data/{sentence,essay}/{train,dev,test}.jsonl
        │
        ▼
src/run_experiments.py       ← Chạy các thí nghiệm (Exp 0–14)
        │
        ▼  results/{run_id}/{task}/{lang}/results.json  &  results.csv
        │
        ▼
results/ + visuals/generated/     ← Kết quả, chart, badge được lưu vĩnh viễn trong git
```

---

## Các thí nghiệm

| ID | Tên | Yêu cầu |
|----|-----|---------|
| 0 | Majority baseline | CPU |
| 1 | TF-IDF + Logistic Regression | CPU |
| 2 | RoBERTa fine-tune | GPU |
| 3 | Ordinal classifier (CORAL) | GPU |
| 4 | LLaMA + LoRA | GPU + `HF_TOKEN` |
| 5 | Hybrid essay classifier | CPU |
| 6 | Domain transfer (cross-corpus) | CPU / GPU |
| 7 | TF-IDF + LinearSVC | CPU |
| 8 | Zero-shot XLM-R | GPU / full ML stack |
| 9 | Word-only TF-IDF + Logistic Regression | CPU |
| 10 | Ensemble (LR + ComplementNB soft voting) | CPU |
| 11 | DeBERTa-v3 fine-tune | GPU |
| 12 | DeBERTa-v3 + Ordinal CORAL | GPU |
| 13 | Transformer + Ordinal late fusion | GPU |
| 14 | LLaMA + LoRA self-consistency (3 seeds) | GPU + `HF_TOKEN` |

---

## Chạy cục bộ

### 1. Cài đặt

```bash
pip install -r requirements-dev.txt          # Để chạy tests
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt              # Để chạy pipeline đầy đủ
```

### 2. Chạy unit tests & smoke tests

```bash
python -m pytest tests/ -v --tb=short
```

### 3. Chuẩn bị dữ liệu

```bash
python -m src.prepare_data --dataset UniversalCEFR/cefr_sp_en --output data/
```

Sinh ra:

```
data/
  sentence/
    train.jsonl  dev.jsonl  test.jsonl
  essay/
    train.jsonl  dev.jsonl  test.jsonl
```

### 4. Chạy thí nghiệm

```bash
# CPU baselines (không cần GPU, không cần HF_TOKEN):
python -m src.run_experiments \
    --task      sentence \
    --exps      0 1 5 7 \
    --data_dir  data/ \
    --save_results results/

# Kết quả được lưu tại:
#   results/results.json   ← JSON với đầy đủ metrics
#   results/results.csv    ← CSV để mở bằng Excel / pandas
```

---

## CI/CD

### CI – Tests (`ci.yml`)

Chạy tự động trên **mọi push và pull request**:

```
Checkout → Python 3.10 → pip install → pytest (114 tests)
```

Tất cả tests không cần GPU hoặc kết nối mạng.

### Full Pipeline (`full_pipeline.yml`) — kích hoạt thủ công

Workflow `workflow_dispatch` để chạy toàn bộ pipeline với dữ liệu thực:

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `task` | `sentence` | Track phân loại (`sentence` hoặc `essay`) |
| `exps` | `0 1 5 7` | IDs thí nghiệm cách nhau bởi dấu cách |
| `dataset` | `UniversalCEFR/cefr_sp_en` | Dataset trên HuggingFace |
| `epochs` | `3` | Số epochs cho Exp 2–4, 11–13 |

#### Các giai đoạn của Full Pipeline

```
Stage 1 – lint-and-test         Unit tests (giống CI)
Stage 2 – prepare-data          Tải dataset từ HuggingFace, sinh JSONL splits
Stage 3 – run-cpu-experiments   Chạy Exp 0, 1, 5, 6, 7, 9, 10 (CPU-only)
Stage 4 – run-transformer-exp   Chạy Exp 2, 3, 8, 11, 12, 13 (full ML stack / GPU)
Stage 5 – run-llm-experiment    Chạy Exp 4, 14 (GPU + HF_TOKEN)
Stage 6 – commit-results        ★ Commit kết quả + visualization vào repo
```

#### Kết quả được đẩy về đâu?

**Stage 6 (`commit-results`)** tự động commit kết quả trở lại repository:

```
results/
  {github.run_id}/
    {task}/
      {lang}/
        results.json    ← Metrics của các experiments cho ngôn ngữ đó
        results.csv     ← Cùng nội dung, định dạng CSV
      logs/
        results_cpu_*.txt
        results_transformer_*.txt
        results_llm_*.txt
      run_info.json     ← Metadata: run ID, actor, sha, timestamp, inputs
```

- Được commit với message: `ci: save experiment results run #N (task=..., exps=...) [skip ci]`
- Push vào **cùng branch** đã trigger workflow (`${{ github.ref }}`)
- Dùng `git pull --rebase` trước khi push để tránh conflict khi chạy song song

> **Lưu ý**: Mỗi lần trigger sẽ tạo một thư mục mới `results/{run_id}/` nên các lần chạy không ghi đè lên nhau.

<!-- AUTO-RESULTS-START -->
## Results Dashboard

Visualization được cập nhật tự động từ multilingual runs mới nhất của pipeline.

### Visuals
- `sentence`:
  - [QWK Heatmap](visuals/generated/sentence/qwk_heatmap.svg)
  - [Best QWK by Language](visuals/generated/sentence/best_qwk_by_language.svg)
- `essay`:
  - [QWK Heatmap](visuals/generated/essay/qwk_heatmap.svg)
  - [Best QWK by Language](visuals/generated/essay/best_qwk_by_language.svg)

### Badges

![Sentence Best QWK](visuals/generated/sentence/badges/best-qwk.svg)
![Sentence Best Macro-F1](visuals/generated/sentence/badges/best-macro-f1.svg)
![Sentence Languages](visuals/generated/sentence/badges/languages.svg)

### Sentence

Run mới nhất: [`results/24663346552/sentence`](results/24663346552/sentence)

| Language | Best experiment by QWK | QWK | Accuracy | Note |
|----------|-------------------------|-----|----------|------|
| `en` | `Exp 7` | `0.6996` | `0.5195` | - |
| `ru` | `Exp 7` | `0.6634` | `0.4331` | - |
| `it` | `N/A` | `N/A` | `N/A` | no committed results for this language |
| `es` | `Exp 7` | `0.8612` | `0.9674` | - |
| `de` | `N/A` | `N/A` | `N/A` | no committed results for this language |
| `fr` | `Exp 10` | `0.7097` | `0.5282` | - |

Inputs: `task=sentence`, `exps=0 1 5 7 9 10`, `language=all`

### Essay

Chưa có multilingual run nào được commit cho task này.
<!-- AUTO-RESULTS-END -->

#### GitHub Actions artifacts (tạm thời)

Ngoài việc commit vào repo, mỗi stage còn upload **artifact** lên GitHub Actions (lưu 30 ngày):

- `prepared-data` — Dữ liệu đã chuẩn bị (Stage 2)
- `results-cpu-{task}` — Kết quả CPU experiments (Stage 3)
- `results-transformer-{task}` — Kết quả transformer experiments (Stage 4)
- `results-llm-{task}` — Kết quả LLM experiments (Stage 5)

---

## Cấu trúc thư mục

```
itmo-vkr-cefr/
├── .github/
│   └── workflows/
│       ├── ci.yml              ← CI: chạy tests trên mọi push/PR
│       └── full_pipeline.yml   ← Full pipeline: manual trigger
├── src/
│   ├── config.py               ← Cấu hình toàn cục (CEFR labels, splits, ...)
│   ├── data_utils.py           ← Tiện ích xử lý dữ liệu
│   ├── prepare_data.py         ← Script chuẩn bị dữ liệu
│   ├── run_experiments.py      ← Unified experiment runner (Exp 0–14)
│   ├── evaluate.py             ← Tính metrics (accuracy, F1, QWK)
│   ├── majority_baseline.py    ← Exp 0: Majority baseline
│   ├── baseline_tfidf.py       ← Exp 1: TF-IDF baseline
│   ├── hybrid_essay.py         ← Exp 5: Hybrid essay classifier
│   ├── transformer_classifier.py ← Exp 2–3: RoBERTa / CORAL
│   ├── ordinal_classifier.py   ← CORAL ordinal decoding
│   ├── llm_lora.py             ← Exp 4: LLaMA + LoRA
│   ├── train_baseline.py       ← Training loop (TF-IDF)
│   ├── train_transformer.py    ← Training loop (transformer)
│   └── train_llm.py            ← Training loop (LLM)
├── tests/                      ← Unit tests + smoke tests (114 tests)
├── results/                    ← ★ Kết quả experiments (commit bởi pipeline)
├── requirements.txt            ← Full dependencies (torch, transformers, ...)
└── requirements-dev.txt        ← Test dependencies (pytest, scikit-learn, numpy)
```

---

## Secrets cần thiết

| Secret | Bắt buộc khi | Mô tả |
|--------|-------------|-------|
| `HF_TOKEN` | Exp 4 (LLaMA) hoặc dataset private | HuggingFace access token |

Thêm tại: **Settings → Secrets and variables → Actions → New repository secret**

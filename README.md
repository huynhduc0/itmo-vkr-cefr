# itmo-vkr-cefr

Dự án nghiên cứu về phân loại trình độ tiếng Anh theo thang CEFR (A1–C2).
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
src/run_experiments.py       ← Chạy các thí nghiệm (Exp 0–10)
        │
        ▼  results/{run_id}/{task}/results.json  &  results.csv
        │
        ▼
results/ (được commit vào repo)   ← Kết quả được lưu vĩnh viễn trong git
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
| 8 | TF-IDF + Complement Naive Bayes | CPU |
| 9 | Word-only TF-IDF + Logistic Regression | CPU |
| 10 | Ensemble (LR + ComplementNB soft voting) | CPU |

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
| `epochs` | `3` | Số epochs cho Exp 2–4 |

#### Các giai đoạn của Full Pipeline

```
Stage 1 – lint-and-test         Unit tests (giống CI)
Stage 2 – prepare-data          Tải dataset từ HuggingFace, sinh JSONL splits
Stage 3 – run-cpu-experiments   Chạy Exp 0, 1, 5, 6, 7, 8, 9, 10 (CPU-only)
Stage 4 – run-transformer-exp   Chạy Exp 2, 3 (yêu cầu GPU)
Stage 5 – run-llm-experiment    Chạy Exp 4 LLaMA+LoRA (GPU + HF_TOKEN)
Stage 6 – commit-results        ★ Commit kết quả vào repo
```

#### Kết quả được đẩy về đâu?

**Stage 6 (`commit-results`)** tự động commit kết quả trở lại repository:

```
results/
  {github.run_id}/
    {task}/
      results.json      ← Metrics của tất cả experiments đã chạy
      results.csv       ← Cùng nội dung, định dạng CSV
      results_cpu.txt   ← Console log của CPU experiments
      run_info.json     ← Metadata: run ID, actor, sha, timestamp, inputs
```

- Được commit với message: `ci: save experiment results run #N (task=..., exps=...) [skip ci]`
- Push vào **cùng branch** đã trigger workflow (`${{ github.ref }}`)
- Dùng `git pull --rebase` trước khi push để tránh conflict khi chạy song song

> **Lưu ý**: Mỗi lần trigger sẽ tạo một thư mục mới `results/{run_id}/` nên các lần chạy không ghi đè lên nhau.

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
│   ├── run_experiments.py      ← Unified experiment runner (Exp 0–10)
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

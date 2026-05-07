# Data directory

Git tracks only placeholders; real CSVs are produced locally or on Colab.

## Binary task (love vs rest, default)

Run from `emotions_rec`:

```bash
python scripts/prepare_emotions_binary.py --label love
```

This downloads **`dair-ai/emotion`** via HuggingFace **`datasets`** and writes under **`data/processed/`**:

| File | Role |
|------|------|
| `emotions_love_train.csv` | Full training split (`label` = multiclass emotion id **0–5**) |
| `emotions_love_validation.csv` | Full validation split |
| `emotions_love_test.csv` | Test split |
| `emotions_love_smoke_train.csv` | Small train subset for quick runs |
| `emotions_love_smoke_validation.csv` | Small val subset (matches `run_configs/binary_love_quick_run.txt`) |

**Active learning** expects **`-filename`** without `.csv` (prefix only), e.g. `data/processed/emotions_love_smoke_train`, and **`-val_path`** as a full CSV path.

## Runtime artifacts (not in Git)

Under `data/processed/` and the experiment root you may see `*_lda.csv`, `*_training_data.csv`, `*_model_results.json`, plus checkpoints under **`models/`**.

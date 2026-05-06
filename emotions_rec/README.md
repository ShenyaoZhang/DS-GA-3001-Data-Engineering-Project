# Binary emotion — **joy vs rest** (`emotion-rec-joy`)

This package implements **one active-learning pipeline**: binary classification for a single positive emotion (**`joy`** by default) vs all other `dair-ai/emotion` labels. Other emotions (love, sadness, …) can be selected later via **`-positive_label`**; extending to **six simultaneous binary detectors** or full **6-way** classification is follow-on work.

## Layout

| Path | Role |
|------|------|
| `scripts/prepare_emotions_binary.py` | Build CSVs from `dair-ai/emotion` (binary labels) |
| `src/main_cluster_emotion_binary.py` | LDA + Thompson/random sampling + Qwen + BERT |
| `src/eval_emotion_binary.py` | Evaluate a saved `models/...` checkpoint |
| `run_configs/binary_joy_*.txt` | Example commands |

## Quick start

```bash
cd emotions_rec
python scripts/prepare_emotions_binary.py --label joy
python -u src/main_cluster_emotion_binary.py \
  -filename "data/processed/emotions_joy_smoke_train" \
  -val_path "data/processed/emotions_joy_smoke_validation.csv" \
  ...  # see run_configs/binary_joy_quick_run.txt
python src/eval_emotion_binary.py \
  -test_path "data/processed/emotions_joy_test.csv" \
  -model_path "models/<checkpoint_dir>" \
  -target_emotion joy
```

## Sanity check (no GPU training)

```bash
python scripts/smoke_check_emotions_rec.py
```

## Branch

Use branch **`emotion-rec-joy`** (rename locally if needed: `git checkout -b emotion-rec-joy`).

The **`LTS/`** Reuters / Newsgroups-style pipeline is kept at the **main**-aligned **vanilla** copy in this repo and is **not** part of the joy binary workflow.

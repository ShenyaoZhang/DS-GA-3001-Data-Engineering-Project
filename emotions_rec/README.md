# Emotions project (`emotions_rec`)

## What actually ships on this branch

Runnable **binary** pipeline (target emotion vs rest, default **love**):

| Piece | Path |
|--------|------|
| Prepare HF CSVs | `scripts/prepare_emotions_binary.py` |
| Active learning + BERT fine-tune | `LTS/main_cluster_emotion_binary.py` |
| Test metrics | `LTS/eval_emotion_binary.py` |
| Colab walkthrough | `notebooks/emotions_rec_sentiment_pipeline.ipynb` |

**Fast verification** (no training, ~seconds):

```bash
cd emotions_rec
python scripts/smoke_check_emotions_rec.py
```

**Note:** `main_cluster_sentiment.py`, `eval_sentiment.py`, and `sentiment_labels.py` are **not** in this repository. The notebook `notebooks/emotions_rec_sentiment_repro.ipynb` describes a 3-class sentiment workflow but those driver scripts are absent here—use the binary notebook above or implement/port sentiment drivers separately.

## Binary quick start (local / Colab shell)

From `emotions_rec`:

```bash
python scripts/prepare_emotions_binary.py --label love
python -u LTS/main_cluster_emotion_binary.py \
  -sample_size 200 \
  -filename "data/processed/emotions_love_smoke_train" \
  -val_path "data/processed/emotions_love_smoke_validation.csv" \
  -balance False \
  -sampling thompson \
  -filter_label True \
  -model_finetune bert-base-uncased \
  -labeling qwen \
  -model text \
  -baseline 0.10 \
  -metric f1_pos \
  -cluster_size 8 \
  -positive_label love \
  -hf_model_id "Qwen/Qwen2.5-3B-Instruct" \
  -max_iterations 3 \
  -num_train_epochs 2 \
  -max_length 128 \
  -batch_size 16 \
  -confidence_threshold 0.40
```

Evaluate (set `model_path` to a folder under `models/` that contains `config.json`):

```bash
python LTS/eval_emotion_binary.py \
  -test_path "data/processed/emotions_love_test.csv" \
  -model_path "models/binary_love_fine_tunned_0_bandit_0" \
  -target_emotion love \
  -base_model "bert-base-uncased" \
  -max_length 128
```

## Colab

Use `notebooks/emotions_rec_sentiment_pipeline.ipynb` and `COLAB.md`.

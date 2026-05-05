# Emotions → binary LTS (dair-ai/emotion)

Mirror the **20 Newsgroups / LTS** workflow: prepare data → run `main_cluster`-style training → evaluate checkpoint.

**Task:** binary triage for one target emotion (default `love`) vs all other emotions (`0` = not target, `1` = target).

## Main files

| Step | File |
|------|------|
| Prepare CSVs | `scripts/prepare_emotions_binary.py` |
| Few-shot from train (optional) | `scripts/build_few_shot_emotions.py` |
| Committed few-shot examples | `prompts/few_shot_examples_emotion_love.json` |
| Active learning loop | `src/main_cluster_emotion_binary.py` |
| Evaluation | `src/eval_emotion_binary.py` |
| Colab notebook | `notebooks/emotions_rec_sentiment_pipeline.ipynb` |

Raw labels in prepared CSVs follow **dair-ai/emotion** ids: `0 sadness, 1 joy, 2 love, 3 anger, 4 fear, 5 surprise`. The training script maps them to binary using `-positive_label`.

## Quick start (local)

Prepare (writes full + smoke splits under `data/processed/`):

```bash
python -u scripts/prepare_emotions_binary.py --label love
```

(Optional) Regenerate few-shot from the full train file:

```bash
python -u scripts/build_few_shot_emotions.py \
  --train_csv "data/processed/emotions_love_train.csv" \
  --out "prompts/few_shot_examples_emotion_love.json" \
  --n_per_class 2
```

Train (see `run_configs/binary_love_quick_run.txt`):

```bash
python -u src/main_cluster_emotion_binary.py \
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
  -positive_label "love" \
  -few_shot_path "prompts/few_shot_examples_emotion_love.json" \
  -hf_model_id "Qwen/Qwen2.5-3B-Instruct" \
  -max_iterations 3 \
  -num_train_epochs 2 \
  -max_length 128 \
  -batch_size 16 \
  -confidence_threshold 0.40
```

Evaluate (use the checkpoint folder with highest validation **`f1_pos`**; optional **`-run_id`** puts checkpoints under `models/<run_id>/`). Add **`-tune_threshold_val_path`** so F1(pos) matches the calibrated operating point:

```bash
python -u src/eval_emotion_binary.py \
  -test_path "data/processed/emotions_love_test.csv" \
  -tune_threshold_val_path "data/processed/emotions_love_validation.csv" \
  -model_path "models/binary_love_fine_tunned_2_bandit_2" \
  -target_emotion "love" \
  -base_model "bert-base-uncased" \
  -max_length 128
```

## Reporting “LLM + bandit/clustering vs imbalance”

Use **`run_configs/paper_thomp_vs_imbalance_love.txt`** (Thompson + LDA clusters) against:

- **`paper_baseline_uniform_love.txt`** — random acquisition from the raw pool (**no** cluster stratification; isolates clustering + exploration).
- **`paper_baseline_lda_strat_random_love.txt`** — random within LDA strata (**no** Thompson); isolates adaptive cluster choice.

Give each command a distinct **`-run_id`** (already baked into those files) so `wins.txt`, `results/`, and `models/` do not collide. Checkpoints appear under **`models/<run_id>/`**.

Evaluate on held-out **`emotions_*_test.csv`** with **`eval_emotion_binary.py`**. Prefer **PR-AUC (pos)** and **ROC-AUC (pos)** on the test fold (shown by the script) together with **F1(pos)**. Use **`-tune_threshold_val_path ..._validation.csv`** so the positivity threshold isn’t pinned to minority-hostile 0.5 under imbalance.

## Colab

Open `notebooks/emotions_rec_sentiment_pipeline.ipynb` and follow `COLAB.md`.

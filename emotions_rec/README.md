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

Evaluate (set `-model_path` to your best checkpoint under `models/`):

```bash
python -u src/eval_emotion_binary.py \
  -test_path "data/processed/emotions_love_test.csv" \
  -model_path "models/binary_love_fine_tunned_0_bandit_0" \
  -target_emotion "love" \
  -base_model "bert-base-uncased" \
  -max_length 128
```

## Colab

Open `notebooks/emotions_rec_sentiment_pipeline.ipynb` and follow `COLAB.md`.

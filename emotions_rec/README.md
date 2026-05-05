# Sentiment Classification Pipeline (dair-ai/emotion)

This branch keeps only the sentiment workflow.

Sentiment labels:
- `0` negative (sadness, anger, fear)
- `1` neutral (surprise)
- `2` positive (joy, love)

## Main files

- `src/main_cluster_sentiment.py` — active-learning training loop
- `src/eval_sentiment.py` — test-set evaluation
- `src/sentiment_labels.py` — label mapping
- `notebooks/emotions_rec_sentiment_repro.ipynb` — end-to-end Colab notebook

## Quick start (local)

Train:

```bash
python -u src/main_cluster_sentiment.py \
  -sampling thompson \
  -sample_size 300 \
  -filter_label True \
  -model_finetune bert-base-uncased \
  -labeling qwen \
  -filename "data/processed/train_inner_emotions_sentiment" \
  -model text \
  -metric f1_macro \
  -val_path "data/processed/val_emotions_sentiment.csv" \
  -cluster_size 10 \
  -few_shot_path "prompts/few_shot_examples_sentiment.json" \
  -hf_model_id "Qwen/Qwen2.5-3B-Instruct" \
  -max_iterations 8 \
  -confidence_threshold 0.35
```
Logs are printed directly to console/notebook output.

Evaluate:

```bash
python src/eval_sentiment.py \
  -test_path "data/processed/test_emotions_sentiment.csv" \
  -model_path "models/sentiment_fine_tunned_0_bandit_0" \
  -base_model "bert-base-uncased"
```

## Colab

Use `notebooks/emotions_rec_sentiment_repro.ipynb` and follow `COLAB.md`.

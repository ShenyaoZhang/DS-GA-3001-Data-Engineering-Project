# Running the sentiment pipeline on Google Colab

This branch is sentiment-only.

Labels:
- `0` negative (sadness, anger, fear)
- `1` neutral (surprise)
- `2` positive (joy, love)

## 1) Open the notebook

Open:

- `emotions_rec/notebooks/emotions_rec_sentiment_repro.ipynb`

It includes Drive setup, branch checkout, dataset build, training, and evaluation cells.

## 2) Runtime and auth

- Set runtime to **GPU** in Colab.
- Provide a Hugging Face token (read access) for Qwen model loading.

## 3) Train

Run:

```bash
# Recommended in Colab: redirect all stdout/stderr to a log file.
log_path="outputs/sentiment_train_$(date +%s).log"

python src/main_cluster_sentiment.py \
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
  -confidence_threshold 0.35 \
  -outputs_dir "outputs" \
  -console_logs False > "$log_path" 2>&1
```

Logs are saved to:

`outputs/sentiment_train_*.log`

## 4) Evaluate

```bash
python src/eval_sentiment.py \
  -test_path "data/processed/test_emotions_sentiment.csv" \
  -model_path "models/sentiment_fine_tunned_0_bandit_0" \
  -base_model "bert-base-uncased"
```

Replace `model_path` with your best checkpoint.

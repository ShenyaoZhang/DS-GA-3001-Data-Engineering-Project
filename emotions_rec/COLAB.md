# Running the Emotions binary LTS pipeline on Google Colab

This matches the **prepare → train → eval** structure used for 20 Newsgroups / LTS.

## 1) Open the notebook

- `emotions_rec/notebooks/emotions_rec_sentiment_pipeline.ipynb`

## 2) Runtime and auth

- GPU runtime.
- Hugging Face token for Qwen (`HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN`).

## 3) Prepare data

The notebook runs:

```bash
python -u scripts/prepare_emotions_binary.py --label love
```

CSV columns: `id`, `title`, `label` where `label` is the **raw emotion id 0–5** (training maps to binary internally).

## 4) Few-shot examples

- Committed default: `prompts/few_shot_examples_emotion_love.json`
- The notebook can refresh them with `scripts/build_few_shot_emotions.py` from `emotions_love_train.csv`.

## 5) Train (`main_cluster`-style)

Same flags as `run_configs/binary_love_quick_run.txt` (Thompson sampling, Qwen labeling, BERT fine-tune).

## 6) Evaluate

```bash
python -u src/eval_emotion_binary.py \
  -test_path "data/processed/emotions_love_test.csv" \
  -model_path "models/<your_best_checkpoint>" \
  -target_emotion "love" \
  -base_model "bert-base-uncased" \
  -max_length 128
```

Use `models/binary_love_fine_tunned_*_bandit_*` from your run.

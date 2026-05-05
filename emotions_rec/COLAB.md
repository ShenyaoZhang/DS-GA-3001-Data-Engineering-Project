# Running the Emotions binary LTS pipeline on Google Colab

This matches the **prepare → train → eval** structure used for 20 Newsgroups / LTS.

## 1) Open the notebook

- `emotions_rec/notebooks/emotions_rec_sentiment_pipeline.ipynb`

## 2) Runtime and auth

- **GPU**, ideally **GPU T4** with **High RAM** (recommended for long sessions).
- Set **HF token** (`HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN`) so Hub downloads hit higher rate limits and finish faster than anonymous pulls.

Keeping the HF cache under the Colab VM (default **`~/.cache/huggingface`**) avoids slow repeated downloads; put only the cloned repo / outputs on **Drive** if you need persistence.

## 3) Prepare data

The notebook runs:

```bash
python -u scripts/prepare_emotions_binary.py --label love
```

CSV columns: `id`, `title`, `label` where `label` is the **raw emotion id 0–5** (training maps to binary internally).

## 4) Few-shot examples

- Committed default: `prompts/few_shot_examples_emotion_love.json`
- The notebook refreshes them with `scripts/build_few_shot_emotions.py` from `emotions_love_train.csv`.

## 5) Train (`main_cluster`-style, ~≤1 hr budget on T4 + High RAM)

The notebook aligns with **`run_configs/binary_love_quick_run.txt`**: **5 active-learning rounds**, **3 BERT epochs/round**, **batch 24**, **`-confidence_threshold 0.38`**, **`sample_size 224`**. Cold Qwen downloads add extra wall time beyond that hour.

Optional ablations (`run_configs/paper_*.txt`) use **`-run_id`** so `models/` and `results/` do not collide.

## 6) Evaluate

Tune the positive-class probability threshold on **`data/processed/emotions_love_validation.csv`** (full val, unlike smoke val used during training):

```bash
python -u src/eval_emotion_binary.py \
  -test_path "data/processed/emotions_love_test.csv" \
  -tune_threshold_val_path "data/processed/emotions_love_validation.csv" \
  -model_path "models/<your_best_checkpoint>" \
  -target_emotion "love" \
  -base_model "bert-base-uncased" \
  -max_length 128
```

Pick **`<your_best_checkpoint>`** from **`models/binary_love_fine_tunned_*`** using the highest **`eval_f1_pos`** printed during training.

For runs with **`-run_id`**, checkpoints live under **`models/<run_id>/`**.

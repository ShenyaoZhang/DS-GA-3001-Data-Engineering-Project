# Emotion pipelines (dair-ai/emotion)

## A) Tiered emotion classification

Two-stage design:

1. **Tier 1** — `src/main_cluster_binary.py`: **negative** vs **positive** (BERT, with optional Qwen pseudo-labels and topic bandits).
2. **Tier 2** — `src/main_cluster_hierarchical.py`:
   - If **negative** → **sadness / anger / fear** (3-class BERT).
   - If **positive** → **joy / love** (2-class BERT).

The **surprise** class in the original dataset is treated as **positive** in tier 1 only; this hierarchy does not emit **surprise** as a leaf label (see `src/tiered_labels.py`).

## B) Sentiment-only classification (new branch flow)

Single-stage 3-class sentiment model:

- `0` = negative (sadness, anger, fear)
- `1` = neutral (surprise)
- `2` = positive (joy, love)

Files:

- `src/main_cluster_sentiment.py` — active-learning loop in the same style as `LTS/main_cluster.py`
- `src/sentiment_labels.py` — source-of-truth mapping from emotion ids to sentiment ids
- `src/eval_sentiment.py` — test-set evaluation for 3-class sentiment checkpoints
- `notebooks/emotions_rec_sentiment_repro.ipynb` — Colab workflow for sentiment pipeline

## Folder layout

```text
emotions_rec/
├── README.md
├── COLAB.md                      ← Google Colab walkthrough
├── notebooks/
│   ├── emotions_rec_repro.ipynb  ← tiered data prep + train/eval commands
│   └── emotions_rec_sentiment_repro.ipynb  ← sentiment data prep + train command
├── src/
│   ├── tiered_labels.py          ← all label id maps
│   ├── sentiment_labels.py       ← emotion→sentiment map (3 classes)
│   ├── main_cluster_binary.py    ← tier 1 training only
│   ├── main_cluster_hierarchical.py  ← train/eval full cascade
│   ├── main_cluster_sentiment.py ← sentiment-only train loop
│   ├── preprocessing.py
│   ├── LDA.py
│   ├── labeling.py
│   ├── fine_tune.py
│   ├── random_sampling.py
│   └── thompson_sampling.py
├── prompts/
│   └── few_shot_examples_emotion.json   ← produced by the notebook (emotion slug)
├── data/
│   └── processed/                ← train_inner_emotions_emotion*.csv (generated)
└── run_configs/
    ├── random_run.txt
    ├── thompson_run.txt
    ├── sentiment_random_run.txt
    ├── sentiment_thompson_run.txt
    └── sentiment_eval_run.txt
```

## Quick start (local)

From the `emotions_rec` directory, with `data/processed/train_inner_emotions_emotion.csv` and validation CSV in place (see notebook or `data/README.md`):

```bash
python src/main_cluster_hierarchical.py train \
  -filename "data/processed/train_inner_emotions_emotion" \
  -val_path "data/processed/val_emotions_emotion.csv" \
  -few_shot_path "prompts/few_shot_examples_emotion.json" \
  -hf_model_id "Qwen/Qwen2.5-3B-Instruct" \
  -max_iterations 8
```

```bash
python src/main_cluster_hierarchical.py eval \
  -val_path "data/processed/test_emotions_emotion.csv" \
  -binary_model "models/binary_fine_tunned_0_bandit_0" \
  -neg_model "models/neg_sub_fine_tunned_0_bandit_0" \
  -pos_model "models/pos_sub_fine_tunned_0_bandit_0"
```

Replace model paths with your saved checkpoints.

## Quick start (sentiment-only)

From the `emotions_rec` directory:

```bash
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
  -console_logs False
```

Training logs are written to `outputs/sentiment_train_*.log` (useful for long Colab runs).

Evaluate:

```bash
python src/eval_sentiment.py \
  -test_path "data/processed/test_emotions_sentiment.csv" \
  -model_path "models/sentiment_fine_tunned_0_bandit_0" \
  -base_model "bert-base-uncased"
```

## Google Colab

See **[COLAB.md](./COLAB.md)** for Drive mounting, Hugging Face login, GPU runtime, and step-by-step cells aligned with `notebooks/emotions_rec_repro.ipynb`.

## Dataset

Loaded from Hugging Face:

```python
from datasets import load_dataset
load_dataset("dair-ai/emotion")
```

No manual download is required.

## Requirements

Install from the repository root `requirements.txt` (`pandas`, `numpy`, `torch`, `transformers`, `datasets`, `scikit-learn`, `nltk`, ...).

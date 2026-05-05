# Tiered emotion classification (dair-ai/emotion)

Two-stage design:

1. **Tier 1** — `src/main_cluster_binary.py`: **negative** vs **positive** (BERT, with optional Qwen pseudo-labels and topic bandits).
2. **Tier 2** — `src/main_cluster_hierarchical.py`:
   - If **negative** → **sadness / anger / fear** (3-class BERT).
   - If **positive** → **joy / love** (2-class BERT).

The **surprise** class in the original dataset is treated as **positive** in tier 1 only; this hierarchy does not emit **surprise** as a leaf label (see `src/tiered_labels.py`).

## Folder layout

```text
emotions_rec/
├── README.md
├── COLAB.md                      ← Google Colab walkthrough
├── notebooks/
│   └── emotions_rec_repro.ipynb  ← data prep + optional train commands
├── src/
│   ├── tiered_labels.py          ← all label id maps
│   ├── main_cluster_binary.py    ← tier 1 training only
│   ├── main_cluster_hierarchical.py  ← train/eval full cascade
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
    └── thompson_run.txt
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

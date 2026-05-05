# Running the tiered emotion pipeline on Google Colab

This guide assumes you use **Google Colab** with **Google Drive** mounted so the repo can live on Drive (persistent checkpoints and data).

## What gets trained

| Stage | Script | Role |
|--------|--------|------|
| Tier 1 | `src/main_cluster_binary.py` | Negative vs positive (BERT + bandit sampling + optional Qwen pseudo-labels) |
| Tier 2a | `main_cluster_hierarchical.py train` (second phase) | Negative → sadness / anger / fear |
| Tier 2b | same (third phase) | Positive → **joy** / **love** (2 classes) |

Label definitions and id maps are in `src/tiered_labels.py`.

## 1. Open the notebook

In Colab, open `emotions_rec/notebooks/emotions_rec_repro.ipynb` (upload it or clone the repo and open the file from Drive). The first cells mount Drive and clone or pull the repository.

## 2. Set paths and install dependencies

- After the clone cell, your working directory should be the **repository root** or `emotions_rec` as in the notebook’s `EXPERIMENT_ROOT`.
- Run the **dependency install** cell (uses the repo `requirements.txt` at the project root). If you use only `emotions_rec`, you can also run:

  `pip install -r requirements.txt`

  from the repo root, or `pip install pandas numpy scipy scikit-learn transformers datasets torch nltk` if you keep a minimal set.

- **GPU**: Runtime → Change runtime type → **GPU** (T4 or better recommended for Qwen + BERT fine-tuning).

## 3. Hugging Face token (for Qwen)

Pseudo-labeling uses a gated or large model (`Qwen/...`). In Colab:

```python
from huggingface_hub import login
login()  # paste token when prompted
```

Create a token at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) if needed.

## 4. Build the 6-class CSVs and few-shot JSON

Run the notebook sections that:

1. Load `dair-ai/emotion` from HuggingFace.
2. Write **`TARGET_SLUG = "emotion"`** so you get:
   - `data/processed/train_inner_emotions_emotion.csv`
   - `data/processed/val_emotions_emotion.csv`
   - `data/processed/test_emotions_emotion.csv`
3. Build `prompts/few_shot_examples_emotion.json` (the notebook cell that picks few-shot examples per class).

**Working directory for all CLI commands below:** the `emotions_rec` folder (so `data/processed/...` paths resolve).

```python
import os
os.chdir("/content/drive/MyDrive/.../DS-GA-3001-Data-Engineering-Project/emotions_rec")
```

Adjust the path to match where you cloned the repo.

## 5. Train the full tiered stack

From `emotions_rec/`, run the hierarchical trainer (trains binary, then neg 3-class, then pos 2-class):

```bash
python src/main_cluster_hierarchical.py train \
  -filename "data/processed/train_inner_emotions_emotion" \
  -val_path "data/processed/val_emotions_emotion.csv" \
  -few_shot_path "prompts/few_shot_examples_emotion.json" \
  -hf_model_id "Qwen/Qwen2.5-3B-Instruct" \
  -max_iterations 8
```

You can add the other flags from `run_configs/thompson_run.txt` (sampling, `sample_size`, `confidence_threshold`, etc.).

**Outputs (typical):**

- `models/binary_fine_tunned_*` — tier 1
- `models/neg_sub_fine_tunned_*` — tier 2 negative
- `models/pos_sub_fine_tunned_*` — tier 2 positive (2 labels)
- LDA cache: `data/processed/train_inner_emotions_emotion_lda.csv`
- State files in the current directory (`selected_ids.txt`, etc.) — remove before a clean re-run if needed

## 6. Evaluate on the test set

Pick the best checkpoint folders from `models/`, then:

```bash
python src/main_cluster_hierarchical.py eval \
  -val_path "data/processed/test_emotions_emotion.csv" \
  -binary_model "models/binary_fine_tunned_7_bandit_5" \
  -neg_model "models/neg_sub_fine_tunned_7_bandit_3" \
  -pos_model "models/pos_sub_fine_tunned_7_bandit_1"
```

Replace the three model paths with your actual run’s directory names. The eval step prints a full 6-class report and, if the test set contains **surprise** labels, a second report that excludes gold **surprise** (this pipeline never predicts surprise at tier 2).

## 7. Memory and runtime tips

- Use a smaller Qwen for labeling if VRAM is tight, e.g. `Qwen/Qwen2.5-3B-Instruct` instead of 7B.
- Reduce `-sample_size` or `-max_iterations` for a quicker smoke test.
- If CUDA runs out of memory after Qwen, restart runtime or call `torch.cuda.empty_cache()` between stages (the training scripts already try to free the labeler).

## 8. Repo layout reference

```
emotions_rec/
  src/
    tiered_labels.py              # label maps (single source of truth)
    main_cluster_binary.py       # tier 1 only
    main_cluster_hierarchical.py # orchestrates tier 1 + tier 2 train/eval
    preprocessing.py, LDA.py, fine_tune.py, labeling.py, ...
  notebooks/emotions_rec_repro.ipynb
  data/processed/                 # generated CSVs (not in git)
  prompts/                        # few_shot_examples_emotion.json
  run_configs/                    # example train commands
```

For questions about the old single-task `joy` vs rest experiment, that entry point was removed in favor of this tiered setup; the same notebook can still export data—use the **emotion** slug and the hierarchical script as above.

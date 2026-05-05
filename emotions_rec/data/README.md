# Data directory

Nothing large is committed here. Use `scripts/prepare_emotions_binary.py` (or the Colab notebook) to download **dair-ai/emotion** via Hugging Face `datasets` and write processed CSVs.

## `processed/` outputs (binary `love` example)

After `python -u scripts/prepare_emotions_binary.py --label love`:

| File | Contents |
|------|----------|
| `emotions_love_train.csv` | Full train pool: `id`, `title`, `label` (`label` = raw emotion 0–5) |
| `emotions_love_validation.csv` | Validation (raw 0–5) |
| `emotions_love_test.csv` | Test (raw 0–5) |
| `emotions_love_smoke_train.csv` | Small train subset for quick runs |
| `emotions_love_smoke_validation.csv` | Small val subset |

Training maps the target emotion (e.g. `love` → id `2`) to **binary** gold labels on the fly; eval maps test the same way.

Runtime artifacts (not usually tracked): `<filename>_lda.csv`, `*_training_data.csv`, `*_model_results.json`, and checkpoints under `models/`.

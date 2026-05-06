# Data (`emotions_rec`)

CSV files are **generated**, not committed. Use:

```bash
python scripts/prepare_emotions_binary.py --label joy
```

Outputs under `data/processed/` (prefix **`emotions_joy`** when `--label joy`):

| File | Description |
|------|-------------|
| `emotions_joy_train.csv` | Full train split, binary labels |
| `emotions_joy_validation.csv` | Validation |
| `emotions_joy_test.csv` | Test |
| `emotions_joy_smoke_train.csv` | Small train subset |
| `emotions_joy_smoke_validation.csv` | Small val subset |

**Active learning:** pass **`-filename`** as the path **without** `.csv` (e.g. `data/processed/emotions_joy_smoke_train`). Pass **`-val_path`** as the full validation CSV path.

Runtime artifacts: `*_lda.csv`, `*_training_data.csv`, `*_model_results.json`, `models/` checkpoints.

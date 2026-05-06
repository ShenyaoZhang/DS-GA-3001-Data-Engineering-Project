# Colab — binary **joy** vs rest

1. Mount Drive, clone/checkout branch **`emotion-rec-joy`** (or your fork with these files).
2. Runtime: **GPU**.
3. `cd emotions_rec` and `pip install -r ../requirements.txt` (plus `datasets` if missing).
4. Set **`HF_TOKEN`** for Qwen downloads.

```bash
python scripts/smoke_check_emotions_rec.py
python scripts/prepare_emotions_binary.py --label joy
```

Training (paste from `run_configs/binary_joy_quick_run.txt` or shorten iterations for a smoke test):

```bash
python -u src/main_cluster_emotion_binary.py \
  -filename "data/processed/emotions_joy_smoke_train" \
  -val_path "data/processed/emotions_joy_smoke_validation.csv" \
  -positive_label joy \
  ...
```

Evaluation: use **`eval_emotion_binary.py`** with **`-model_path`** pointing at a folder under `models/` that contains **`config.json`** (absolute path recommended).

Multiline `!python ... \` cells in notebooks often break `{VAR}` expansion — prefer `subprocess.run([...])` with real path strings.

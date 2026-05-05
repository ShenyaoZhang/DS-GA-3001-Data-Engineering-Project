# Google Colab — `emotions_rec`

## Before a long GPU run

From `emotions_rec` in the Colab terminal (or a small code cell):

```bash
python scripts/smoke_check_emotions_rec.py
```

This checks Python syntax, `import torch` / `transformers`, and `--help` on the binary CLIs. It does **not** run training.

## Binary pipeline (supported)

1. Open **`notebooks/emotions_rec_sentiment_pipeline.ipynb`** (name is historical; it runs the **binary** love-vs-rest flow).
2. GPU runtime + Hugging Face token (read) for Qwen.
3. Run cells in order: clone/pull → install deps → `prepare_emotions_binary` → training → eval.

Training uses `src/main_cluster_emotion_binary.py`; evaluation uses `src/eval_emotion_binary.py`.

## Sentiment 3-class notebook

**`notebooks/emotions_rec_sentiment_repro.ipynb`** documents a 3-class sentiment mapping, but this branch does **not** include `main_cluster_sentiment.py` or `eval_sentiment.py`. Do not run its training cells as-is unless you add those scripts. For a working Colab path, use **`emotions_rec_sentiment_pipeline.ipynb`** above.

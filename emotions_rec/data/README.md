# Data directory

Raw data stays empty in Git; `dair-ai/emotion` is loaded via HuggingFace `datasets` in `notebooks/emotions_rec_sentiment_repro.ipynb`.

## processed/

After running the notebook with **`TARGET_SLUG = "sentiment"`**, expect:

- `train_inner_emotions_sentiment.csv`
- `val_emotions_sentiment.csv`
- `test_emotions_sentiment.csv`

Runtime artifacts (not tracked) include `*_lda.csv`, `*_training_data.csv`, `*_data_labeled.csv`, and `*_model_results.json` under this folder and `models/` in the experiment root.

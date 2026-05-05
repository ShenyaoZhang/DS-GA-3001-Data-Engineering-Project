"""
Evaluate a trained sentiment model (negative / neutral / positive).

Usage:
  python src/eval_sentiment.py \
    -test_path "data/processed/test_emotions_sentiment.csv" \
    -model_path "models/sentiment_fine_tunned_7_bandit_3" \
    -base_model "bert-base-uncased"
"""

import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import BertForSequenceClassification, BertTokenizer

from preprocessing import TextPreprocessor
from sentiment_labels import SENTIMENT_NAMES


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate 3-class sentiment model")
    parser.add_argument("-test_path", type=str, required=True, help="CSV with columns: title, label")
    parser.add_argument("-model_path", type=str, required=True, help="Fine-tuned model directory")
    parser.add_argument("-base_model", type=str, default="bert-base-uncased", help="Tokenizer base model")
    parser.add_argument("-batch_size", type=int, default=64)
    parser.add_argument("-max_length", type=int, default=128)
    return parser.parse_args()


def prepare_test_data(path):
    preprocessor = TextPreprocessor()
    df = pd.read_csv(path)
    df = preprocessor.preprocess_df(df)
    if "label" not in df.columns:
        raise ValueError("test_path must include a 'label' column.")
    df["training_text"] = df["clean_title"] if "clean_title" in df.columns else df["title"]
    return df


def predict_labels(df, model, tokenizer, device, batch_size, max_length):
    texts = df["training_text"].fillna("").astype(str).tolist()
    preds = []
    confs = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=1)
        preds.append(probs.argmax(dim=1).cpu().numpy())
        confs.append(probs.max(dim=1).values.cpu().numpy())
    return np.concatenate(preds), np.concatenate(confs)


def main():
    args = parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    df = prepare_test_data(args.test_path)
    tokenizer = BertTokenizer.from_pretrained(args.base_model)
    model = BertForSequenceClassification.from_pretrained(args.model_path, num_labels=3).to(device).eval()

    y_true = df["label"].astype(int).values
    y_pred, y_conf = predict_labels(df, model, tokenizer, device, args.batch_size, args.max_length)

    labels = sorted(SENTIMENT_NAMES.keys())
    names = [SENTIMENT_NAMES[i] for i in labels]

    print("\n=== Sentiment evaluation ===")
    print(classification_report(y_true, y_pred, labels=labels, target_names=names, zero_division=0))
    print(f"Accuracy:    {accuracy_score(y_true, y_pred):.4f}")
    print(f"F1 Macro:    {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"F1 Weighted: {f1_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"Mean confidence: {float(np.mean(y_conf)):.4f}")


if __name__ == "__main__":
    main()

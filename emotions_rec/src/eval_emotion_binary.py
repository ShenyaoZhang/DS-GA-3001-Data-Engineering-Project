"""
Evaluate a binary emotion model (target emotion vs rest).
"""

import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from transformers import BertForSequenceClassification, BertTokenizer

from labeling import EMOTION_MAP
from preprocessing import TextPreprocessor


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate binary emotion model")
    parser.add_argument("-test_path", type=str, required=True)
    parser.add_argument("-model_path", type=str, required=True)
    parser.add_argument("-target_emotion", type=str, default="love", choices=sorted(EMOTION_MAP.keys()))
    parser.add_argument("-base_model", type=str, default="bert-base-uncased")
    parser.add_argument("-batch_size", type=int, default=64)
    parser.add_argument("-max_length", type=int, default=128)
    parser.add_argument(
        "-tune_threshold_val_path",
        type=str,
        default=None,
        help="Optional validation CSV (same schema as test). Search probability threshold maximizing F1 for the positive class, then evaluate test at that threshold.",
    )
    parser.add_argument(
        "-positive_prob_threshold",
        type=float,
        default=None,
        help="Fixed threshold on P(positive): predict 1 if prob >= this. Ignored when -tune_threshold_val_path is set.",
    )
    return parser.parse_args()


def prepare_test_data(path, target_id):
    preprocessor = TextPreprocessor()
    df = pd.read_csv(path)
    df = preprocessor.preprocess_df(df)
    df["training_text"] = df["clean_title"] if "clean_title" in df.columns else df["title"]
    df["label"] = df["label"].astype(int).apply(lambda x: 1 if x == target_id else 0)
    return df


def predict_probs_batches(df, model, tokenizer, device, batch_size, max_length):
    texts = df["training_text"].fillna("").astype(str).tolist()
    probs_all = []
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
        probs_all.append(probs.cpu().numpy())
    return np.concatenate(probs_all, axis=0)


def eval_at_threshold(y_true, probs, t):
    p_pos = probs[:, 1]
    y_pred = (p_pos >= t).astype(int)
    return y_pred, p_pos


def tune_threshold(y_true, probs):
    p_pos = probs[:, 1]
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 19):
        y_pred = (p_pos >= t).astype(int)
        f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t, best_f1


def main():
    args = parse_args()
    target_id = EMOTION_MAP[args.target_emotion]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    df = prepare_test_data(args.test_path, target_id)
    tokenizer = BertTokenizer.from_pretrained(args.base_model)
    model = BertForSequenceClassification.from_pretrained(args.model_path, num_labels=2).to(device).eval()
    y_true = df["label"].astype(int).values
    probs = predict_probs_batches(df, model, tokenizer, device, args.batch_size, args.max_length)

    threshold = None
    if args.tune_threshold_val_path:
        vdf = prepare_test_data(args.tune_threshold_val_path, target_id)
        v_probs = predict_probs_batches(vdf, model, tokenizer, device, args.batch_size, args.max_length)
        vy = vdf["label"].astype(int).values
        threshold, vtune_f1 = tune_threshold(vy, v_probs)
        print(f"\n=== Threshold tuning (validation) → t={threshold:.4f} (max F1 pos≈{vtune_f1:.4f}) ===")
    elif args.positive_prob_threshold is not None:
        threshold = args.positive_prob_threshold
        print(f"\n=== Using fixed P(positive) threshold={threshold:.4f} ===")

    if threshold is not None:
        y_pred, y_conf_pos = eval_at_threshold(y_true, probs, threshold)
    else:
        y_pred = probs.argmax(axis=1)
        y_conf_pos = probs.max(axis=1)

    print(f"\n=== Binary evaluation ({args.target_emotion} vs rest) ===")
    print(classification_report(y_true, y_pred, labels=[0, 1], target_names=["other", args.target_emotion], zero_division=0))
    print(f"Accuracy:         {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision (pos):  {precision_score(y_true, y_pred, pos_label=1, zero_division=0):.4f}")
    print(f"Recall (pos):     {recall_score(y_true, y_pred, pos_label=1, zero_division=0):.4f}")
    print(f"F1 (pos):         {f1_score(y_true, y_pred, pos_label=1, zero_division=0):.4f}")
    print(f"F1 Macro:         {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    try:
        print(f"ROC-AUC (pos):    {roc_auc_score(y_true, probs[:, 1]):.4f}")
    except ValueError:
        print("ROC-AUC (pos):    n/a (needs both classes on test)")
    print(f"PR-AUC (pos):    {average_precision_score(y_true, probs[:, 1]):.4f}")
    print(f"Mean max-prob:   {float(np.mean(probs.max(axis=1))):.4f}")


if __name__ == "__main__":
    main()

"""
Evaluate a binary emotion model (target emotion vs rest).
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from transformers import BertForSequenceClassification, BertTokenizer

from labeling import EMOTION_MAP
from preprocessing import TextPreprocessor


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate binary emotion model")
    parser.add_argument("-test_path", type=str, required=True)
    parser.add_argument(
        "-model_path",
        type=str,
        required=True,
        help="Local folder with config.json (checkpoint). On Colab prefer an absolute path under .../emotions_rec/models/...",
    )
    parser.add_argument("-target_emotion", type=str, default="joy", choices=sorted(EMOTION_MAP.keys()))
    parser.add_argument("-base_model", type=str, default="bert-base-uncased")
    parser.add_argument("-batch_size", type=int, default=64)
    parser.add_argument("-max_length", type=int, default=128)
    return parser.parse_args()


def prepare_test_data(path, target_id):
    preprocessor = TextPreprocessor()
    df = pd.read_csv(path)
    df = preprocessor.preprocess_df(df)
    df["training_text"] = df["clean_title"] if "clean_title" in df.columns else df["title"]
    df["label"] = df["label"].astype(int).apply(lambda x: 1 if x == target_id else 0)
    return df


def predict_labels(df, model, tokenizer, device, batch_size, max_length):
    texts = df["training_text"].fillna("").astype(str).tolist()
    preds, confs = [], []
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


def _normalize_model_path(raw: str) -> str:
    """Fix common typo 'odels/...' -> 'models/...' so HF Hub is not queried by mistake."""
    p = raw.strip()
    if p.startswith("./odels/"):
        p = "./models/" + p[len("./odels/") :]
    elif p.startswith("odels/"):
        p = "models/" + p[len("odels/") :]
    elif p.startswith("odels"):
        p = "m" + p
    return p


def main():
    args = parse_args()
    target_id = EMOTION_MAP[args.target_emotion]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    df = prepare_test_data(args.test_path, target_id)
    model_path = os.path.abspath(os.path.expanduser(_normalize_model_path(args.model_path)))
    if not os.path.isdir(model_path):
        raise FileNotFoundError(
            f"-model_path is not an existing directory: {model_path!r}\n"
            "Use the folder that contains config.json, e.g.\n"
            "  /content/drive/MyDrive/DS-GA-3001-Data-Engineering-Project/emotions_rec/models/binary_love_fine_tunned_2_bandit_2\n"
            "Common mistake: typing odels/ instead of models/ makes Transformers hit huggingface.co/odels/... (404)."
        )
    if not os.path.isfile(os.path.join(model_path, "config.json")):
        raise FileNotFoundError(f"No config.json under {model_path!r}.")

    tokenizer = BertTokenizer.from_pretrained(args.base_model)
    model = BertForSequenceClassification.from_pretrained(
        model_path, num_labels=2, local_files_only=True
    ).to(device).eval()
    y_true = df["label"].astype(int).values
    y_pred, y_conf = predict_labels(df, model, tokenizer, device, args.batch_size, args.max_length)

    print(f"\n=== Binary evaluation ({args.target_emotion} vs rest) ===")
    print(classification_report(y_true, y_pred, labels=[0, 1], target_names=["other", args.target_emotion], zero_division=0))
    print(f"Accuracy:         {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision (pos):  {precision_score(y_true, y_pred, pos_label=1, zero_division=0):.4f}")
    print(f"Recall (pos):     {recall_score(y_true, y_pred, pos_label=1, zero_division=0):.4f}")
    print(f"F1 (pos):         {f1_score(y_true, y_pred, pos_label=1, zero_division=0):.4f}")
    print(f"F1 Macro:         {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"Mean confidence:  {float(np.mean(y_conf)):.4f}")


if __name__ == "__main__":
    main()

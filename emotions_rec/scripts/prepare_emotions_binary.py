"""
Prepare dair-ai/emotion CSVs for target-emotion vs rest.

Output column `label` stays the **raw dataset emotion id** (0–5). Smoke splits are class-balanced
by raw emotions; `main_cluster_emotion_binary` / `eval_emotion_binary` binarize using `-positive_label`.
"""

import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import pandas as pd
from datasets import load_dataset

from labeling import EMOTION_MAP


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare emotions binary CSVs")
    parser.add_argument("--label", type=str, default="love", choices=sorted(EMOTION_MAP.keys()))
    parser.add_argument("--output_dir", type=str, default="data/processed")
    parser.add_argument("--smoke_train_n", type=int, default=1200)
    parser.add_argument("--smoke_val_n", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def split_to_df(split_data, id_offset=0):
    rows = []
    for i, ex in enumerate(split_data):
        rows.append(
            {
                "id": id_offset + i,
                "title": str(ex["text"]).strip(),
                "label": int(ex["label"]),
            }
        )
    return pd.DataFrame(rows)


def save_csv(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[prepare_emotions_binary] wrote {path} ({len(df)} rows)")


def main():
    args = parse_args()
    target = args.label
    target_id = EMOTION_MAP[target]
    base = f"emotions_{target}"

    ds = load_dataset("dair-ai/emotion")
    train = split_to_df(ds["train"], 0)
    val = split_to_df(ds["validation"], len(train))
    test = split_to_df(ds["test"], len(train) + len(val))

    # CSV `label` must stay as raw dair-ai/emotion ids (0–5): training code maps to binary internally.
    train_gold = train[["id", "title", "label"]].copy()
    val_gold = val[["id", "title", "label"]].copy()
    test_gold = test[["id", "title", "label"]].copy()

    out = args.output_dir
    train_path = os.path.join(out, f"{base}_train.csv")
    val_path = os.path.join(out, f"{base}_validation.csv")
    test_path = os.path.join(out, f"{base}_test.csv")
    smoke_train_path = os.path.join(out, f"{base}_smoke_train.csv")
    smoke_val_path = os.path.join(out, f"{base}_smoke_validation.csv")

    save_csv(train_gold, train_path)
    save_csv(val_gold, val_path)
    save_csv(test_gold, test_path)

    pos_train = train_gold[train_gold["label"] == target_id]
    neg_train = train_gold[train_gold["label"] != target_id]
    n_pos = min(len(pos_train), args.smoke_train_n // 3)
    n_neg = min(len(neg_train), args.smoke_train_n - n_pos)
    smoke_train = pd.concat(
        [pos_train.sample(n_pos, random_state=args.seed), neg_train.sample(n_neg, random_state=args.seed)]
    ).sample(frac=1, random_state=args.seed).reset_index(drop=True)

    pos_val = val_gold[val_gold["label"] == target_id]
    neg_val = val_gold[val_gold["label"] != target_id]
    n_pos_v = min(len(pos_val), args.smoke_val_n // 3)
    n_neg_v = min(len(neg_val), args.smoke_val_n - n_pos_v)
    smoke_val = pd.concat(
        [pos_val.sample(n_pos_v, random_state=args.seed), neg_val.sample(n_neg_v, random_state=args.seed)]
    ).sample(frac=1, random_state=args.seed).reset_index(drop=True)

    save_csv(smoke_train, smoke_train_path)
    save_csv(smoke_val, smoke_val_path)

    print(f"[prepare_emotions_binary] target='{target}' positive_id={target_id}")
    bin_train = train_gold["label"].apply(lambda x: 1 if x == target_id else 0)
    print("[prepare_emotions_binary] train (binary target vs rest):", bin_train.value_counts().to_dict())
    bin_val = val_gold["label"].apply(lambda x: 1 if x == target_id else 0)
    print("[prepare_emotions_binary] val (binary target vs rest):", bin_val.value_counts().to_dict())
    bin_test = test_gold["label"].apply(lambda x: 1 if x == target_id else 0)
    print("[prepare_emotions_binary] test (binary target vs rest):", bin_test.value_counts().to_dict())
    print("[prepare_emotions_binary] use this with main_cluster_emotion_binary.py:")
    print(
        f'  -filename "{os.path.join(out, base + "_smoke_train").replace(".csv", "")}" '
        f'-val_path "{smoke_val_path}" -positive_label "{target}"'
    )


if __name__ == "__main__":
    main()

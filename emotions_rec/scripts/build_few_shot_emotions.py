"""
Build few-shot JSON for Qwen 6-class emotion prompts (0–5).

Reads a prepared CSV with columns: id, title, label (raw emotion id).
Outputs a list of {\"id\", \"text\", \"label\"} for Labeling.load_few_shot_examples.
"""

import argparse
import json
import os

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--n_per_class", type=int, default=2)
    p.add_argument("--max_chars", type=int, default=140)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.train_csv)
    if "label" not in df.columns or "title" not in df.columns:
        raise ValueError("train_csv needs columns: title, label (raw emotion int 0–5)")

    examples = []
    for cls in sorted(df["label"].astype(int).unique()):
        part = df[df["label"].astype(int) == cls]
        take = min(len(part), args.n_per_class)
        if take == 0:
            continue
        sample = part.sample(take, random_state=args.seed + int(cls))
        for _, row in sample.iterrows():
            examples.append(
                {
                    "id": int(row["id"]),
                    "text": str(row["title"])[: args.max_chars],
                    "label": str(int(row["label"])),
                }
            )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)

    print(f"[build_few_shot_emotions] wrote {args.out} ({len(examples)} examples)")


if __name__ == "__main__":
    main()

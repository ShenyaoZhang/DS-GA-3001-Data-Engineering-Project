"""
Build few-shot Qwen prompt examples from a prepared emotions CSV (has raw_label + title).
"""

from __future__ import annotations

import json
import os
import random
from typing import Any

import pandas as pd

from labeling import EMOTION_MAP


def build_few_shot_examples_from_prepared_csv(
    path: str,
    *,
    n_per_class: int = 2,
    max_total: int = 8,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Sample up to *n_per_class* documents per raw emotion class (0–5), then cap at *max_total*.

    Each item matches ``labeling.Labeling`` expectations: ``{"text": str, "label": str}``
    with *label* the digit 0–5.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "raw_label" not in df.columns:
        raise ValueError(f"{path} has no raw_label column; re-run prepare_emotions_binary.py")
    if "title" not in df.columns:
        raise ValueError(f"{path} must contain a title column")

    df = df.copy()
    df["raw_label"] = df["raw_label"].astype(int)

    rng = pd.Series(range(len(df)))
    chunks: list[dict[str, Any]] = []
    classes = sorted(set(EMOTION_MAP.values()))
    for cls in classes:
        sub = df[df["raw_label"] == cls]
        if sub.empty:
            continue
        k = min(n_per_class, len(sub))
        part = sub.sample(n=k, random_state=seed + cls, replace=False)
        for _, row in part.iterrows():
            chunks.append({"text": str(row["title"]), "label": str(int(cls))})

    rnd = random.Random(seed)
    rnd.shuffle(chunks)
    return chunks[:max_total]


def write_few_shot_json(examples: list[dict[str, Any]], out_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)


def main_cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Build few-shot JSON from prepared emotions CSV")
    p.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to prepared CSV (e.g. data/processed/emotions_<label>_train.csv)",
    )
    p.add_argument(
        "--label",
        type=str,
        default=None,
        choices=sorted(EMOTION_MAP.keys()),
        help="Use data/processed/emotions_<label>_train.csv when --csv is omitted",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output JSON path (default: data/processed/few_shot_<label>.json)",
    )
    p.add_argument("--n_per_class", type=int, default=2)
    p.add_argument("--max_total", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.csv:
        csv_path = args.csv
        tag = args.label or "custom"
    else:
        if not args.label:
            p.error("provide --csv or --label")
        tag = args.label
        csv_path = os.path.join(root, "data", "processed", f"emotions_{tag}_train.csv")

    out_path = args.out or os.path.join(root, "data", "processed", f"few_shot_{tag}.json")

    ex = build_few_shot_examples_from_prepared_csv(
        csv_path,
        n_per_class=args.n_per_class,
        max_total=args.max_total,
        seed=args.seed,
    )
    write_few_shot_json(ex, out_path)
    print(f"Wrote {len(ex)} examples to {out_path}")


if __name__ == "__main__":
    main_cli()

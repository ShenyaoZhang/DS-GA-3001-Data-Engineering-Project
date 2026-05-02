"""Convert Reuters archive CSVs into LTS-ready pool + validation files.

Input files live under data/archive/ (ModApte_train.csv, ModApte_test.csv, ...).
Output is written to data_use_cases/ with the columns expected by main_cluster.py:

  - Pool CSV (used as -filename, code appends ".csv"):
        id, title, description, label_earn
    The 'label_earn' column is the gold binary label (1 if 'earn' in topics
    else 0) -- LTS does not need it for -labeling qwen, but it is useful for
    diagnostics or for the -labeling file oracle baseline.

  - Validation CSV (used as -val_path, full path):
        id, title, description, label
    'label' is the same earn/not-earn binary; this is what fine_tune.py reads.

A small "_smoke" subset is also written so a fast end-to-end run is possible.
"""

from __future__ import annotations

import argparse
import ast
import html
import re
from pathlib import Path

import pandas as pd


def _clean_id(raw: object) -> str:
    """Reuters new_id values come wrapped in literal double quotes (e.g. '\"1\"')."""
    return str(raw).strip().strip('"').strip("'")


def _decode_text(raw: object, max_chars: int) -> str:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return ""
    text = html.unescape(str(raw))
    text = re.sub(r"\s+", " ", text).strip()
    if max_chars and len(text) > max_chars:
        text = text[:max_chars]
    return text

REPO_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_DIR = REPO_ROOT / "data" / "archive"
OUT_DIR = REPO_ROOT / "data_use_cases"


def parse_topics(raw: object) -> list[str]:
    """Return a list of topic strings from the messy archive 'topics' column."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    text = str(raw).strip()
    if not text or text == "[]":
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            return [str(t) for t in parsed]
    except Exception:
        pass
    inner = re.sub(r"^\[|\]$", "", text)
    parts = re.findall(r"'([^']*)'", inner)
    if parts:
        return parts
    return [tok for tok in inner.split() if tok]


def has_label(topics: list[str], target: str) -> int:
    return int(target in topics)


def build_frame(src_csv: Path, target_label: str, max_desc_chars: int) -> pd.DataFrame:
    raw = pd.read_csv(src_csv)
    raw = raw.dropna(subset=["title"]).copy()
    raw["title"] = raw["title"].apply(lambda t: _decode_text(t, 0))
    raw = raw[raw["title"].str.len() > 0]

    raw["topics_list"] = raw["topics"].apply(parse_topics)
    raw["label_earn"] = raw["topics_list"].apply(lambda ts: has_label(ts, target_label))

    raw["id"] = raw["new_id"].apply(_clean_id)

    body_src = raw["text"] if "text" in raw.columns else pd.Series([""] * len(raw))
    raw["description"] = body_src.apply(lambda b: _decode_text(b, max_desc_chars))

    out = raw[["id", "title", "description", "label_earn"]].drop_duplicates(subset=["id"])
    return out.reset_index(drop=True)


def write_pool_and_validation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    out_dir: Path,
    stem: str,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pool_path = out_dir / f"{stem}_train.csv"
    val_df = test_df.rename(columns={"label_earn": "label"})[
        ["id", "title", "description", "label"]
    ]
    val_path = out_dir / f"{stem}_validation.csv"
    train_df.to_csv(pool_path, index=False)
    val_df.to_csv(val_path, index=False)
    return {"pool": pool_path, "validation": val_path}


def make_smoke_subset(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_pool: int,
    n_val: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = pd.Series(range(len(train_df))).sample(
        n=min(n_pool, len(train_df)), random_state=seed
    ).tolist()
    pool_smoke = train_df.iloc[rng].reset_index(drop=True)

    pos = test_df[test_df["label_earn"] == 1]
    neg = test_df[test_df["label_earn"] == 0]
    take_pos = min(len(pos), max(1, n_val // 2))
    take_neg = min(len(neg), n_val - take_pos)
    val_smoke = pd.concat(
        [pos.sample(take_pos, random_state=seed), neg.sample(take_neg, random_state=seed)]
    ).sample(frac=1, random_state=seed).reset_index(drop=True)

    return pool_smoke, val_smoke


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", default="ModApte", choices=["ModApte", "ModHayes", "ModLewis"])
    p.add_argument("--label", default="earn", help="Reuters topic to use as positive class")
    p.add_argument("--max-desc-chars", type=int, default=1500,
                   help="Truncate article body for the description column (0 = keep full)")
    p.add_argument("--smoke-pool", type=int, default=500)
    p.add_argument("--smoke-val", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-stem", default=None,
                   help="Override output filename stem (default: reuters_{split_lower}_{label})")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train_csv = ARCHIVE_DIR / f"{args.split}_train.csv"
    test_csv = ARCHIVE_DIR / f"{args.split}_test.csv"
    if not train_csv.exists() or not test_csv.exists():
        raise SystemExit(f"Missing input under {ARCHIVE_DIR}: {train_csv.name} / {test_csv.name}")

    print(f"[prepare_reuters] reading {train_csv.name} and {test_csv.name}")
    train_df = build_frame(train_csv, args.label, args.max_desc_chars)
    test_df = build_frame(test_csv, args.label, args.max_desc_chars)

    stem = args.out_stem or f"reuters_{args.split.lower()}_{args.label}"
    paths = write_pool_and_validation(train_df, test_df, OUT_DIR, stem)
    print(f"[prepare_reuters] pool       -> {paths['pool']}  ({len(train_df)} rows, "
          f"pos={int(train_df['label_earn'].sum())})")
    print(f"[prepare_reuters] validation -> {paths['validation']}  ({len(test_df)} rows, "
          f"pos={int(test_df['label_earn'].sum())})")

    pool_smoke, val_smoke = make_smoke_subset(
        train_df, test_df, args.smoke_pool, args.smoke_val, args.seed
    )
    smoke_paths = write_pool_and_validation(pool_smoke, val_smoke, OUT_DIR, f"{stem}_smoke")
    print(f"[prepare_reuters] smoke pool -> {smoke_paths['pool']}  ({len(pool_smoke)} rows, "
          f"pos={int(pool_smoke['label_earn'].sum())})")
    print(f"[prepare_reuters] smoke val  -> {smoke_paths['validation']}  ({len(val_smoke)} rows, "
          f"pos={int(val_smoke['label_earn'].sum())})")


if __name__ == "__main__":
    main()

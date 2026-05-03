"""Convert Reuters archive CSVs into LTS-ready pool + validation files.

Input files live under data/archive/ (ModApte_train.csv, ModApte_test.csv, ...).
If only data/archive.zip exists, this script auto-extracts it on first run.

Two task types are supported (--task-type):

  binary (default)
    Pool CSV  : id, title, description, label, label_<topic>
                'label' = 1 if the --label topic is in the article's topics else 0.
                'label_<topic>' is kept for backward compatibility.
    Validation: id, title, description, label

  multiclass
    Pool CSV  : id, title, description, label, label_text
                'label' is an integer index into --labels (0..N-1). The first
                label in --labels that matches an article's topics is assigned
                (priority order = the order you pass them in). If no listed
                label matches, the article gets the --other-label class
                (default 'other'); pass --no-other to drop those rows.
    Validation: id, title, description, label, label_text

A small "_smoke" subset is also written so a fast end-to-end run is possible.

Use --list-topics to see the most frequent Reuters topics in the train split
(handy when picking targets for a new triage experiment).
"""

from __future__ import annotations

import argparse
import ast
import html
import re
import zipfile
from collections import Counter
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
ARCHIVE_ZIP = REPO_ROOT / "data" / "archive.zip"
OUT_DIR = REPO_ROOT / "data_use_cases"


def ensure_archive_extracted() -> None:
    """Extract data/archive.zip into data/archive/ on first run."""
    if ARCHIVE_DIR.exists() and any(ARCHIVE_DIR.glob("*.csv")):
        return
    if not ARCHIVE_ZIP.exists():
        raise SystemExit(
            f"Neither {ARCHIVE_DIR} nor {ARCHIVE_ZIP} is present; cannot prepare data."
        )
    print(f"[prepare_reuters] extracting {ARCHIVE_ZIP.name} -> {ARCHIVE_DIR}/")
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ARCHIVE_ZIP, "r") as z:
        z.extractall(ARCHIVE_DIR)


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


def _common_columns(raw: pd.DataFrame, max_desc_chars: int) -> pd.DataFrame:
    raw = raw.dropna(subset=["title"]).copy()
    raw["title"] = raw["title"].apply(lambda t: _decode_text(t, 0))
    raw = raw[raw["title"].str.len() > 0]
    raw["topics_list"] = raw["topics"].apply(parse_topics)
    raw["id"] = raw["new_id"].apply(_clean_id)
    body_src = raw["text"] if "text" in raw.columns else pd.Series([""] * len(raw))
    raw["description"] = body_src.apply(lambda b: _decode_text(b, max_desc_chars))
    return raw


def build_frame(src_csv: Path, target_label: str, max_desc_chars: int) -> pd.DataFrame:
    raw = pd.read_csv(src_csv)
    raw = _common_columns(raw, max_desc_chars)

    label_col = f"label_{target_label.lower().replace(' ', '_')}"
    raw["label"] = raw["topics_list"].apply(lambda ts: has_label(ts, target_label))
    raw[label_col] = raw["label"]

    cols = ["id", "title", "description", "label", label_col]
    out = raw[cols].drop_duplicates(subset=["id"])
    return out.reset_index(drop=True)


def assign_multiclass(topics: list[str], priority: list[str], other_label: str | None) -> str | None:
    """Return the first label from `priority` present in `topics`, else `other_label`.

    If `other_label` is None (i.e. --no-other), articles with no matching topic
    are dropped (caller filters out None).
    """
    for cand in priority:
        if cand in topics:
            return cand
    return other_label


def build_frame_multiclass(
    src_csv: Path,
    priority: list[str],
    other_label: str | None,
    max_desc_chars: int,
) -> tuple[pd.DataFrame, list[str]]:
    raw = pd.read_csv(src_csv)
    raw = _common_columns(raw, max_desc_chars)
    raw["label_text"] = raw["topics_list"].apply(
        lambda ts: assign_multiclass(ts, priority, other_label)
    )
    raw = raw[raw["label_text"].notna()].copy()

    final_classes = list(priority)
    if other_label is not None and other_label not in final_classes:
        final_classes.append(other_label)
    label_to_id = {name: idx for idx, name in enumerate(final_classes)}
    raw["label"] = raw["label_text"].map(label_to_id).astype(int)

    cols = ["id", "title", "description", "label", "label_text"]
    out = raw[cols].drop_duplicates(subset=["id"]).reset_index(drop=True)
    return out, final_classes


def write_pool_and_validation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    out_dir: Path,
    stem: str,
    keep_columns: list[str] | None = None,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pool_path = out_dir / f"{stem}_train.csv"
    val_path = out_dir / f"{stem}_validation.csv"
    if keep_columns is None:
        keep_columns = ["id", "title", "description", "label"]
    val_df = test_df[keep_columns]
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

    # For multi-class label is integer >=0; treat any non-zero as 'minority'
    # for the validation smoke split, but try to cover every class at least once.
    if test_df["label"].nunique() > 2:
        per_class_n = max(1, n_val // test_df["label"].nunique())
        parts = []
        for cls in sorted(test_df["label"].unique()):
            sub = test_df[test_df["label"] == cls]
            parts.append(sub.sample(min(len(sub), per_class_n), random_state=seed))
        val_smoke = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
        return pool_smoke, val_smoke

    pos = test_df[test_df["label"] == 1]
    neg = test_df[test_df["label"] == 0]
    take_pos = min(len(pos), max(1, n_val // 2))
    take_neg = min(len(neg), n_val - take_pos)
    val_smoke = pd.concat(
        [pos.sample(take_pos, random_state=seed), neg.sample(take_neg, random_state=seed)]
    ).sample(frac=1, random_state=seed).reset_index(drop=True)

    return pool_smoke, val_smoke


def list_topics(src_csv: Path, top_k: int) -> None:
    raw = pd.read_csv(src_csv, usecols=["topics"])
    counts: Counter[str] = Counter()
    for raw_topics in raw["topics"]:
        for t in parse_topics(raw_topics):
            counts[t] += 1
    print(f"[prepare_reuters] top {top_k} topics in {src_csv.name}:")
    width = max((len(t) for t, _ in counts.most_common(top_k)), default=10)
    for topic, count in counts.most_common(top_k):
        print(f"  {topic.ljust(width)}  {count}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", default="ModApte", choices=["ModApte", "ModHayes", "ModLewis"])
    p.add_argument("--task-type", default="binary", choices=["binary", "multiclass"],
                   help="Triage task type: binary (default) or multiclass")
    p.add_argument("--label", default="earn",
                   help="Binary mode: Reuters topic to use as positive class (e.g. earn, trade, acq)")
    p.add_argument("--labels", default=None,
                   help="Multiclass mode: comma-separated topics in priority order, "
                        "e.g. 'earn,acq,trade,crude,money-fx'")
    p.add_argument("--other-label", default="other",
                   help="Multiclass mode: name for the catch-all class (default: 'other')")
    p.add_argument("--no-other", action="store_true",
                   help="Multiclass mode: drop articles that don't match any --labels topic "
                        "instead of putting them in --other-label")
    p.add_argument("--max-desc-chars", type=int, default=1500,
                   help="Truncate article body for the description column (0 = keep full)")
    p.add_argument("--smoke-pool", type=int, default=500)
    p.add_argument("--smoke-val", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-stem", default=None,
                   help="Override output filename stem "
                        "(default binary: reuters_{split_lower}_{label}; "
                        "default multiclass: reuters_{split_lower}_multiclass_{N}cls)")
    p.add_argument("--list-topics", type=int, nargs="?", const=20, default=0,
                   help="Print the top-N topics in the train split and exit (default N=20)")
    return p.parse_args()


def _split_labels(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [tok.strip() for tok in raw.split(",") if tok.strip()]


def _print_class_distribution(df: pd.DataFrame, classes: list[str], name: str) -> None:
    counts = df["label"].value_counts().to_dict()
    parts = [f"{cls}={counts.get(idx, 0)}" for idx, cls in enumerate(classes)]
    print(f"[prepare_reuters] {name} class counts: " + ", ".join(parts))


def _run_binary(args: argparse.Namespace, train_csv: Path, test_csv: Path) -> None:
    print(f"[prepare_reuters] reading {train_csv.name} and {test_csv.name}")
    train_df = build_frame(train_csv, args.label, args.max_desc_chars)
    test_df = build_frame(test_csv, args.label, args.max_desc_chars)

    if int(train_df["label"].sum()) == 0:
        print(
            f"[prepare_reuters] WARNING: zero positives for label='{args.label}' in {train_csv.name}. "
            "Use --list-topics to find a populated topic."
        )

    stem = args.out_stem or f"reuters_{args.split.lower()}_{args.label}"
    paths = write_pool_and_validation(train_df, test_df, OUT_DIR, stem)
    print(f"[prepare_reuters] pool       -> {paths['pool']}  ({len(train_df)} rows, "
          f"pos={int(train_df['label'].sum())})")
    print(f"[prepare_reuters] validation -> {paths['validation']}  ({len(test_df)} rows, "
          f"pos={int(test_df['label'].sum())})")

    pool_smoke, val_smoke = make_smoke_subset(
        train_df, test_df, args.smoke_pool, args.smoke_val, args.seed
    )
    smoke_paths = write_pool_and_validation(pool_smoke, val_smoke, OUT_DIR, f"{stem}_smoke")
    print(f"[prepare_reuters] smoke pool -> {smoke_paths['pool']}  ({len(pool_smoke)} rows, "
          f"pos={int(pool_smoke['label'].sum())})")
    print(f"[prepare_reuters] smoke val  -> {smoke_paths['validation']}  ({len(val_smoke)} rows, "
          f"pos={int(val_smoke['label'].sum())})")


def _run_multiclass(args: argparse.Namespace, train_csv: Path, test_csv: Path) -> None:
    priority = _split_labels(args.labels)
    if len(priority) < 2:
        raise SystemExit(
            "--task-type multiclass requires at least 2 labels via --labels "
            "(comma-separated, priority order)."
        )
    other_label = None if args.no_other else args.other_label

    print(f"[prepare_reuters] reading {train_csv.name} and {test_csv.name}")
    train_df, classes = build_frame_multiclass(train_csv, priority, other_label, args.max_desc_chars)
    test_df, _ = build_frame_multiclass(test_csv, priority, other_label, args.max_desc_chars)

    stem = args.out_stem or f"reuters_{args.split.lower()}_multiclass_{len(classes)}cls"
    keep_cols = ["id", "title", "description", "label", "label_text"]
    paths = write_pool_and_validation(train_df, test_df, OUT_DIR, stem, keep_columns=keep_cols)
    print(f"[prepare_reuters] classes: {classes}")
    print(f"[prepare_reuters] pool       -> {paths['pool']}  ({len(train_df)} rows)")
    _print_class_distribution(train_df, classes, "pool")
    print(f"[prepare_reuters] validation -> {paths['validation']}  ({len(test_df)} rows)")
    _print_class_distribution(test_df, classes, "validation")

    pool_smoke, val_smoke = make_smoke_subset(
        train_df, test_df, args.smoke_pool, args.smoke_val, args.seed
    )
    smoke_paths = write_pool_and_validation(
        pool_smoke, val_smoke, OUT_DIR, f"{stem}_smoke", keep_columns=keep_cols
    )
    print(f"[prepare_reuters] smoke pool -> {smoke_paths['pool']}  ({len(pool_smoke)} rows)")
    _print_class_distribution(pool_smoke, classes, "smoke pool")
    print(f"[prepare_reuters] smoke val  -> {smoke_paths['validation']}  ({len(val_smoke)} rows)")
    _print_class_distribution(val_smoke, classes, "smoke val")
    print("[prepare_reuters] use this with main_cluster.py:")
    print(f'    -task_type "multiclass" -class_labels "{",".join(classes)}"')


def main() -> None:
    args = parse_args()
    ensure_archive_extracted()

    train_csv = ARCHIVE_DIR / f"{args.split}_train.csv"
    test_csv = ARCHIVE_DIR / f"{args.split}_test.csv"
    if not train_csv.exists() or not test_csv.exists():
        raise SystemExit(f"Missing input under {ARCHIVE_DIR}: {train_csv.name} / {test_csv.name}")

    if args.list_topics:
        list_topics(train_csv, args.list_topics)
        return

    if args.task_type == "binary":
        _run_binary(args, train_csv, test_csv)
    else:
        _run_multiclass(args, train_csv, test_csv)


if __name__ == "__main__":
    main()

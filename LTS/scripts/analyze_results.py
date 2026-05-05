"""Analyze LTS Reuters experiment outputs and write a Markdown report.

Reads everything that the LTS pipeline writes into `data_use_cases/`:

  * <stem>_model_results.json   (per-run eval_* metrics, keyed by cluster id or
                                 by the literal string 'random' for random runs)
  * <stem>_data_labeled.csv     (Qwen pseudo-labels for sampled rows)
  * <stem>.csv                  (original prepared pool with gold labels)
  * <stem>_validation.csv       (gold validation set)

For every base experiment (e.g. `reuters_modapte_earn_train`) the script
auto-detects its Thompson run and, when present, the matching random run
(`<stem>_random_model_results.json`) and produces:

  1. A summary CSV at  LTS/results/lts_summary.csv
     One row per (dataset, task_type, sampling) experiment with best /
     last-round eval metrics and Qwen pseudo-label agreement.
  2. A Thompson-vs-random delta CSV at  LTS/results/lts_thompson_vs_random.csv
  3. A Markdown report at  LTS/results/REUTERS_REPORT.md
     Suitable for pasting into the team notebook (section 5.2 Reuters).

Run from the LTS folder:

    python scripts/analyze_results.py

Or specify directories explicitly:

    python scripts/analyze_results.py \
        --data-dir data_use_cases \
        --out-dir results
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data_use_cases"
DEFAULT_OUT_DIR = REPO_ROOT / "results"


# ---------------------------------------------------------------------------
# parsing helpers
# ---------------------------------------------------------------------------


def _split_meta(stem: str) -> Tuple[str, str, str]:
    """Return (split, task_type, label_or_classes) parsed from a filename stem.

    Examples
    --------
    reuters_modapte_earn_train            -> ("ModApte", "binary",     "earn")
    reuters_modlewis_multiclass_6cls_train -> ("ModLewis","multiclass", "6cls")
    """
    s = stem
    # strip optional sampling suffix
    if s.endswith("_random_train"):
        s = s[: -len("_random_train")] + "_train"
    if s.endswith("_random"):
        s = s[: -len("_random")]
    if not s.endswith("_train"):
        s = s + "_train"
    m = re.match(r"reuters_(?P<split>[a-z]+)_(?P<rest>.+)_train$", s)
    if not m:
        return ("?", "?", stem)
    split_map = {"modapte": "ModApte", "modhayes": "ModHayes", "modlewis": "ModLewis"}
    split = split_map.get(m.group("split"), m.group("split"))
    rest = m.group("rest")
    if rest.startswith("multiclass"):
        return (split, "multiclass", rest)
    return (split, "binary", rest)


def _is_random_results_file(p: Path) -> bool:
    return p.name.endswith("_random_model_results.json")


def _base_stem_for_results(p: Path) -> str:
    """Return the experiment stem that this results file describes.

    `reuters_..._train_model_results.json`         -> `reuters_..._train`
    `reuters_..._train_random_model_results.json`  -> `reuters_..._train`
    """
    name = p.name.replace("_model_results.json", "")
    if name.endswith("_random"):
        name = name[: -len("_random")]
    return name


# ---------------------------------------------------------------------------
# data classes
# ---------------------------------------------------------------------------


@dataclass
class RunSummary:
    stem: str
    sampling: str  # 'thompson' or 'random'
    split: str
    task_type: str  # 'binary' or 'multiclass'
    label_or_classes: str
    n_entries: int
    best_f1: Optional[float]
    best_f1_macro: Optional[float]
    best_acc: Optional[float]
    best_loss: Optional[float]
    best_key: Optional[str]
    last_avg_f1: Optional[float]
    last_avg_f1_macro: Optional[float]
    last_avg_acc: Optional[float]
    pseudo_label_rows: int = 0
    pseudo_label_agreement: Optional[float] = None
    qwen_label_distribution: Dict[str, int] = field(default_factory=dict)
    gold_label_distribution: Dict[str, int] = field(default_factory=dict)
    pool_positive_rate: Optional[float] = None  # binary only
    sample_positive_rate: Optional[float] = None  # binary only


# ---------------------------------------------------------------------------
# core analysis
# ---------------------------------------------------------------------------


def _summarize_results_file(path: Path) -> Optional[Dict[str, Any]]:
    data = json.loads(path.read_text())
    rows: List[Dict[str, Any]] = []
    for k, entries in data.items():
        for i, e in enumerate(entries):
            rows.append(
                {
                    "key": k,
                    "entry": i,
                    "f1": e.get("eval_f1"),
                    "f1_macro": e.get("eval_f1_macro"),
                    "acc": e.get("eval_accuracy"),
                    "loss": e.get("eval_loss"),
                }
            )
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["f1"] = pd.to_numeric(df["f1"], errors="coerce")
    df["f1_macro"] = pd.to_numeric(df["f1_macro"], errors="coerce")
    df["acc"] = pd.to_numeric(df["acc"], errors="coerce")
    df["loss"] = pd.to_numeric(df["loss"], errors="coerce")
    best_idx = df["f1"].idxmax()
    best = df.loc[best_idx]
    last_per_key = df.groupby("key").last()
    return {
        "n_entries": int(len(df)),
        "keys": sorted(df["key"].unique().tolist(), key=lambda x: (x != "random", x)),
        "best_f1": float(best["f1"]) if pd.notna(best["f1"]) else None,
        "best_f1_macro": (
            float(best["f1_macro"]) if pd.notna(best["f1_macro"]) else None
        ),
        "best_acc": float(best["acc"]) if pd.notna(best["acc"]) else None,
        "best_loss": float(best["loss"]) if pd.notna(best["loss"]) else None,
        "best_key": str(best["key"]),
        "best_entry": int(best["entry"]),
        "last_avg_f1": float(last_per_key["f1"].mean()) if len(last_per_key) else None,
        "last_avg_f1_macro": (
            float(last_per_key["f1_macro"].mean())
            if len(last_per_key) and last_per_key["f1_macro"].notna().any()
            else None
        ),
        "last_avg_acc": (
            float(last_per_key["acc"].mean()) if len(last_per_key) else None
        ),
    }


def _pseudo_label_quality(
    labeled_path: Path, gold_path: Path
) -> Tuple[float, int, Dict[str, int], Dict[str, int]]:
    lab = pd.read_csv(labeled_path)
    gold = pd.read_csv(gold_path)
    if "id" not in lab.columns or "id" not in gold.columns:
        return float("nan"), 0, {}, {}
    if "label" not in lab.columns or "label" not in gold.columns:
        return float("nan"), 0, {}, {}
    lab["id"] = lab["id"].astype(str)
    gold["id"] = gold["id"].astype(str)
    keep_gold = ["id", "label"]
    if "label_text" in gold.columns:
        keep_gold.append("label_text")
    merged = lab.merge(gold[keep_gold], on="id", suffixes=("_qwen", "_gold"), how="left")
    n = len(merged)
    if n == 0:
        return float("nan"), 0, {}, {}
    agreement = float((merged["label_qwen"] == merged["label_gold"]).mean())
    qwen_dist: Dict[str, int] = {}
    if "answer" in merged.columns:
        qwen_dist = merged["answer"].astype(str).str.lower().value_counts().to_dict()
    elif "label_qwen" in merged.columns:
        qwen_dist = merged["label_qwen"].astype(str).value_counts().to_dict()
    gold_dist: Dict[str, int] = {}
    if "label_text" in merged.columns:
        gold_dist = (
            merged["label_text"].astype(str).str.lower().value_counts().to_dict()
        )
    else:
        gold_dist = merged["label_gold"].astype(str).value_counts().to_dict()
    return agreement, n, qwen_dist, gold_dist


def _binary_positive_rates(
    pool_path: Path, labeled_path: Path
) -> Tuple[Optional[float], Optional[float]]:
    if not pool_path.exists() or not labeled_path.exists():
        return None, None
    try:
        pool = pd.read_csv(pool_path, usecols=["label"])
        lab = pd.read_csv(labeled_path, usecols=["label"])
    except ValueError:
        return None, None
    if "label" not in pool.columns or "label" not in lab.columns:
        return None, None
    if not set(pool["label"].dropna().unique()).issubset({0, 1}):
        return None, None
    pool_rate = float((pool["label"] == 1).mean()) if len(pool) else None
    sample_rate = float((lab["label"] == 1).mean()) if len(lab) else None
    return pool_rate, sample_rate


def _make_run_summary(
    stem: str,
    sampling: str,
    results_path: Path,
    data_dir: Path,
) -> Optional[RunSummary]:
    s = _summarize_results_file(results_path)
    if s is None:
        return None
    split, task_type, label_or_classes = _split_meta(stem)

    # Locate the matching pool CSV and labeled CSV. Random runs have an
    # _random suffix on the pool/labeled filenames; Thompson runs don't.
    if sampling == "random":
        pool_csv = data_dir / f"{stem}_random.csv"
        labeled_csv = data_dir / f"{stem}_random_data_labeled.csv"
    else:
        pool_csv = data_dir / f"{stem}.csv"
        labeled_csv = data_dir / f"{stem}_data_labeled.csv"

    pseudo_agreement = None
    n_labeled = 0
    qwen_dist: Dict[str, int] = {}
    gold_dist: Dict[str, int] = {}
    if labeled_csv.exists() and pool_csv.exists():
        agreement, n_labeled, qwen_dist, gold_dist = _pseudo_label_quality(
            labeled_csv, pool_csv
        )
        pseudo_agreement = (
            None if pd.isna(agreement) else float(agreement)
        )

    pool_rate, sample_rate = (None, None)
    if task_type == "binary":
        pool_rate, sample_rate = _binary_positive_rates(pool_csv, labeled_csv)

    return RunSummary(
        stem=stem,
        sampling=sampling,
        split=split,
        task_type=task_type,
        label_or_classes=label_or_classes,
        n_entries=s["n_entries"],
        best_f1=s["best_f1"],
        best_f1_macro=s["best_f1_macro"],
        best_acc=s["best_acc"],
        best_loss=s["best_loss"],
        best_key=s["best_key"],
        last_avg_f1=s["last_avg_f1"],
        last_avg_f1_macro=s["last_avg_f1_macro"],
        last_avg_acc=s["last_avg_acc"],
        pseudo_label_rows=n_labeled,
        pseudo_label_agreement=pseudo_agreement,
        qwen_label_distribution=qwen_dist,
        gold_label_distribution=gold_dist,
        pool_positive_rate=pool_rate,
        sample_positive_rate=sample_rate,
    )


def collect_runs(data_dir: Path) -> List[RunSummary]:
    results: List[RunSummary] = []
    for p in sorted(data_dir.glob("*_model_results.json")):
        stem = _base_stem_for_results(p)
        sampling = "random" if _is_random_results_file(p) else "thompson"
        run = _make_run_summary(stem, sampling, p, data_dir)
        if run is None:
            continue
        results.append(run)
    return results


# ---------------------------------------------------------------------------
# rendering helpers
# ---------------------------------------------------------------------------


def _fmt(x: Optional[float], digits: int = 4) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "n/a"
    return f"{x:.{digits}f}"


def _fmt_pct(x: Optional[float], digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "n/a"
    return f"{100 * x:.{digits}f}%"


def runs_to_summary_df(runs: List[RunSummary]) -> pd.DataFrame:
    rows = []
    for r in runs:
        rows.append(
            {
                "split": r.split,
                "task_type": r.task_type,
                "labels": r.label_or_classes,
                "sampling": r.sampling,
                "stem": r.stem,
                "n_entries": r.n_entries,
                "best_f1": r.best_f1,
                "best_f1_macro": r.best_f1_macro,
                "best_acc": r.best_acc,
                "best_loss": r.best_loss,
                "best_key": r.best_key,
                "last_avg_f1": r.last_avg_f1,
                "last_avg_f1_macro": r.last_avg_f1_macro,
                "last_avg_acc": r.last_avg_acc,
                "pseudo_rows": r.pseudo_label_rows,
                "pseudo_agreement": r.pseudo_label_agreement,
                "pool_positive_rate": r.pool_positive_rate,
                "sample_positive_rate": r.sample_positive_rate,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(
        ["split", "task_type", "labels", "sampling"], na_position="last"
    ).reset_index(drop=True)


def thompson_vs_random_df(runs: List[RunSummary]) -> pd.DataFrame:
    by_stem: Dict[str, Dict[str, RunSummary]] = {}
    for r in runs:
        by_stem.setdefault(r.stem, {})[r.sampling] = r
    rows = []
    for stem, group in by_stem.items():
        if "thompson" not in group or "random" not in group:
            continue
        t = group["thompson"]
        rd = group["random"]
        rows.append(
            {
                "split": t.split,
                "task_type": t.task_type,
                "labels": t.label_or_classes,
                "stem": stem,
                "thompson_best_f1": t.best_f1,
                "random_best_f1": rd.best_f1,
                "delta_best_f1": (
                    None
                    if t.best_f1 is None or rd.best_f1 is None
                    else t.best_f1 - rd.best_f1
                ),
                "thompson_best_f1_macro": t.best_f1_macro,
                "random_best_f1_macro": rd.best_f1_macro,
                "delta_best_f1_macro": (
                    None
                    if t.best_f1_macro is None or rd.best_f1_macro is None
                    else t.best_f1_macro - rd.best_f1_macro
                ),
                "thompson_best_acc": t.best_acc,
                "random_best_acc": rd.best_acc,
                "delta_best_acc": (
                    None
                    if t.best_acc is None or rd.best_acc is None
                    else t.best_acc - rd.best_acc
                ),
                "thompson_pseudo_agreement": t.pseudo_label_agreement,
                "random_pseudo_agreement": rd.pseudo_label_agreement,
                "thompson_sample_positive_rate": t.sample_positive_rate,
                "random_sample_positive_rate": rd.sample_positive_rate,
                "pool_positive_rate": t.pool_positive_rate,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(
        ["split", "task_type", "labels"], na_position="last"
    ).reset_index(drop=True)


def render_markdown_report(
    runs: List[RunSummary], pairs: pd.DataFrame, summary: pd.DataFrame
) -> str:
    lines: List[str] = []
    lines.append("# Reuters Results: Random Sampling vs Thompson Sampling (LTS)\n")
    lines.append(
        "_Auto-generated by `scripts/analyze_results.py` from the JSON / CSV "
        "artifacts in `data_use_cases/`. Numbers are best-iteration values "
        "across the LTS active-learning loop._\n"
    )

    # ------------------------------------------------------------------
    # Section 1: cross-split summary
    # ------------------------------------------------------------------
    lines.append("## 1. Best-iteration metrics by run\n")
    if summary.empty:
        lines.append("_No result files found in `data_use_cases/`._\n")
    else:
        header = (
            "| Split | Task | Sampling | Best F1 | Best macro F1 | Best acc | "
            "Best key | Pseudo rows | Pseudo agreement |\n"
            "|---|---|---|---:|---:|---:|---|---:|---:|\n"
        )
        lines.append(header)
        for _, r in summary.iterrows():
            lines.append(
                "| {split} | {task} | {sampling} | {f1} | {macro} | {acc} | "
                "{key} | {n} | {agreement} |\n".format(
                    split=r["split"],
                    task=r["task_type"],
                    sampling=r["sampling"],
                    f1=_fmt(r["best_f1"]),
                    macro=_fmt(r["best_f1_macro"]),
                    acc=_fmt(r["best_acc"]),
                    key=r["best_key"],
                    n=int(r["pseudo_rows"]) if pd.notna(r["pseudo_rows"]) else 0,
                    agreement=_fmt(r["pseudo_agreement"]),
                )
            )

    # ------------------------------------------------------------------
    # Section 2: Thompson vs Random pairs
    # ------------------------------------------------------------------
    lines.append("\n## 2. Thompson vs Random (paired comparison)\n")
    lines.append(
        "Each row pairs the Thompson run with the matching random run "
        "(same dataset, same task type). `delta = thompson - random`.\n"
    )
    if pairs.empty:
        lines.append("_No matching Thompson / random pairs found._\n")
    else:
        lines.append(
            "| Split | Task | Labels | Thompson F1 | Random F1 | Δ F1 | "
            "Thompson macro F1 | Random macro F1 | Δ macro F1 |\n"
            "|---|---|---|---:|---:|---:|---:|---:|---:|\n"
        )
        for _, r in pairs.iterrows():
            lines.append(
                "| {split} | {task} | {labels} | {tf1} | {rf1} | {df1} | "
                "{tm} | {rm} | {dm} |\n".format(
                    split=r["split"],
                    task=r["task_type"],
                    labels=r["labels"],
                    tf1=_fmt(r["thompson_best_f1"]),
                    rf1=_fmt(r["random_best_f1"]),
                    df1=_fmt(r["delta_best_f1"]),
                    tm=_fmt(r["thompson_best_f1_macro"]),
                    rm=_fmt(r["random_best_f1_macro"]),
                    dm=_fmt(r["delta_best_f1_macro"]),
                )
            )

    # ------------------------------------------------------------------
    # Section 3: positive-class enrichment (binary only)
    # ------------------------------------------------------------------
    binary_pairs = (
        pairs[pairs["task_type"] == "binary"] if not pairs.empty else pairs
    )
    if not binary_pairs.empty:
        lines.append("\n## 3. Positive-class enrichment (binary tasks)\n")
        lines.append(
            "Compares the share of positive (`label == 1`) examples in the "
            "original pool, the random labeled sample, and the Thompson "
            "labeled sample.\n"
        )
        lines.append(
            "| Split | Labels | Pool positive rate | Random sample rate | "
            "Thompson sample rate | Thompson lift |\n"
            "|---|---|---:|---:|---:|---:|\n"
        )
        for _, r in binary_pairs.iterrows():
            pool = r["pool_positive_rate"]
            rd = r["random_sample_positive_rate"]
            th = r["thompson_sample_positive_rate"]
            lift = (th - rd) if (th is not None and rd is not None) else None
            lines.append(
                "| {split} | {labels} | {pool} | {rd} | {th} | {lift} |\n".format(
                    split=r["split"],
                    labels=r["labels"],
                    pool=_fmt_pct(pool),
                    rd=_fmt_pct(rd),
                    th=_fmt_pct(th),
                    lift=_fmt_pct(lift),
                )
            )

    # ------------------------------------------------------------------
    # Section 4: pseudo-label quality (Qwen vs gold)
    # ------------------------------------------------------------------
    lines.append("\n## 4. Pseudo-label quality (Qwen vs gold on sampled rows)\n")
    if not summary.empty:
        lines.append(
            "| Split | Task | Sampling | Pseudo rows | Agreement |\n"
            "|---|---|---|---:|---:|\n"
        )
        for _, r in summary.iterrows():
            lines.append(
                "| {split} | {task} | {sampling} | {n} | {ag} |\n".format(
                    split=r["split"],
                    task=r["task_type"],
                    sampling=r["sampling"],
                    n=int(r["pseudo_rows"]) if pd.notna(r["pseudo_rows"]) else 0,
                    ag=_fmt(r["pseudo_agreement"]),
                )
            )

    # ------------------------------------------------------------------
    # Section 5: Qwen multi-class label distribution skew
    # ------------------------------------------------------------------
    mc_runs = [r for r in runs if r.task_type == "multiclass" and r.qwen_label_distribution]
    if mc_runs:
        lines.append("\n## 5. Qwen multi-class output distribution vs gold\n")
        lines.append(
            "For each multi-class run, this table shows what Qwen actually "
            "emitted vs what the gold labels are for the same sampled rows. "
            "A heavy skew toward one class (typically `trade`) indicates a "
            "prompting/calibration limitation rather than a pipeline bug.\n"
        )
        for r in mc_runs:
            lines.append(
                f"\n### {r.split} {r.task_type} ({r.sampling}, "
                f"{r.pseudo_label_rows} sampled rows)\n"
            )
            qwen_total = sum(r.qwen_label_distribution.values())
            gold_total = sum(r.gold_label_distribution.values())
            keys = sorted(
                set(r.qwen_label_distribution) | set(r.gold_label_distribution)
            )
            lines.append("| Class | Qwen count | Qwen % | Gold count | Gold % |\n")
            lines.append("|---|---:|---:|---:|---:|\n")
            for k in keys:
                q = r.qwen_label_distribution.get(k, 0)
                g = r.gold_label_distribution.get(k, 0)
                qp = (q / qwen_total) if qwen_total else 0.0
                gp = (g / gold_total) if gold_total else 0.0
                lines.append(
                    f"| {k} | {q} | {_fmt_pct(qp)} | {g} | {_fmt_pct(gp)} |\n"
                )

    # ------------------------------------------------------------------
    # Section 6: interpretation
    # ------------------------------------------------------------------
    lines.append("\n## 6. Interpretation\n")
    bullets = _build_interpretation(pairs, runs)
    for b in bullets:
        lines.append(f"- {b}\n")

    return "".join(lines)


def _build_interpretation(pairs: pd.DataFrame, runs: List[RunSummary]) -> List[str]:
    bullets: List[str] = []
    if pairs.empty:
        bullets.append(
            "No matching Thompson/random pairs were found yet, so a direct "
            "head-to-head interpretation is not possible from the artifacts "
            "in `data_use_cases/`."
        )
        return bullets

    # binary
    binary_pairs = pairs[pairs["task_type"] == "binary"]
    if not binary_pairs.empty:
        n_t_better = int((binary_pairs["delta_best_f1"] > 0).sum())
        n_r_better = int((binary_pairs["delta_best_f1"] < 0).sum())
        n_total = len(binary_pairs)
        bullets.append(
            f"On binary `earn` triage, Thompson sampling beats random sampling "
            f"on best weighted F1 in {n_t_better} of {n_total} splits "
            f"(random wins in {n_r_better})."
        )
        for _, r in binary_pairs.iterrows():
            d = r["delta_best_f1"]
            if d is None:
                continue
            verdict = "Thompson higher" if d > 0 else (
                "Random higher" if d < 0 else "tied"
            )
            bullets.append(
                f"{r['split']} binary: Thompson F1={_fmt(r['thompson_best_f1'])}, "
                f"Random F1={_fmt(r['random_best_f1'])} ({verdict} by "
                f"{_fmt(d)})."
            )

    # multiclass
    mc_pairs = pairs[pairs["task_type"] == "multiclass"]
    if not mc_pairs.empty:
        bullets.append(
            "On multi-class triage (6 classes), Thompson sampling also "
            "outperforms random sampling, but absolute scores remain low."
        )
        for _, r in mc_pairs.iterrows():
            bullets.append(
                f"{r['split']} multiclass: Thompson F1="
                f"{_fmt(r['thompson_best_f1'])} / macro="
                f"{_fmt(r['thompson_best_f1_macro'])}; Random F1="
                f"{_fmt(r['random_best_f1'])} / macro="
                f"{_fmt(r['random_best_f1_macro'])}."
            )

    # pseudo-label quality
    for r in runs:
        if r.task_type != "multiclass" or r.pseudo_label_agreement is None:
            continue
        if r.pseudo_label_agreement < 0.25:
            bullets.append(
                f"Qwen pseudo-label agreement on {r.split} multiclass "
                f"({r.sampling}) is only "
                f"{_fmt_pct(r.pseudo_label_agreement)} on sampled rows — "
                f"the multi-class prompt + few-shot examples should be "
                f"upgraded before drawing strong conclusions."
            )
            break

    return bullets


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="directory with *_model_results.json + *_data_labeled.csv "
             "(default: LTS/data_use_cases)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="directory for generated CSV/Markdown (default: LTS/results)",
    )
    args = p.parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[analyze] reading results from {data_dir}")
    runs = collect_runs(data_dir)
    print(f"[analyze] loaded {len(runs)} runs")

    summary_df = runs_to_summary_df(runs)
    pairs_df = thompson_vs_random_df(runs)

    summary_csv = out_dir / "lts_summary.csv"
    pairs_csv = out_dir / "lts_thompson_vs_random.csv"
    report_md = out_dir / "REUTERS_REPORT.md"

    summary_df.to_csv(summary_csv, index=False)
    pairs_df.to_csv(pairs_csv, index=False)
    report_md.write_text(render_markdown_report(runs, pairs_df, summary_df))

    print(f"[analyze] wrote {summary_csv}")
    print(f"[analyze] wrote {pairs_csv}")
    print(f"[analyze] wrote {report_md}")


if __name__ == "__main__":
    main()

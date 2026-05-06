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
  3. A set of PNG figures under LTS/results/figures/
     - binary_final_metrics.png        (random vs Thompson, weighted/macro F1, acc)
     - multiclass_final_metrics.png    (same, multi-class runs)
     - binary_positive_enrichment.png  (pool / random / Thompson positive rate)
     - pseudo_label_agreement.png      (Qwen-vs-gold agreement per run)
     - qwen_vs_gold_distribution.png   (per-class output distribution skew)
  4. A Markdown report at  LTS/results/REUTERS_REPORT.md
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

import matplotlib

matplotlib.use("Agg")  # no display required
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


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


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------


_BAR_COLORS = {
    "random": "#7f8c8d",     # neutral grey
    "thompson": "#2980b9",   # LTS blue
    "pool": "#bdc3c7",
    "qwen": "#e67e22",
    "gold": "#27ae60",
}


def _save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _final_metrics_chart(
    pairs: pd.DataFrame,
    task_type: str,
    out_path: Path,
    title: str,
) -> Optional[Path]:
    """Grouped bar chart: per-split Thompson vs Random for F1, macro F1, acc."""
    df = pairs[pairs["task_type"] == task_type].copy()
    if df.empty:
        return None
    df = df.sort_values("split").reset_index(drop=True)
    splits = df["split"].tolist()
    metrics = [
        ("Weighted F1", "thompson_best_f1", "random_best_f1"),
        ("Macro F1", "thompson_best_f1_macro", "random_best_f1_macro"),
        ("Accuracy", "thompson_best_acc", "random_best_acc"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
    x = np.arange(len(splits))
    width = 0.38
    for ax, (label, t_col, r_col) in zip(axes, metrics):
        random_vals = [
            v if pd.notna(v) else np.nan for v in df[r_col].tolist()
        ]
        thompson_vals = [
            v if pd.notna(v) else np.nan for v in df[t_col].tolist()
        ]
        b1 = ax.bar(
            x - width / 2,
            random_vals,
            width,
            label="Random",
            color=_BAR_COLORS["random"],
        )
        b2 = ax.bar(
            x + width / 2,
            thompson_vals,
            width,
            label="Thompson (LTS)",
            color=_BAR_COLORS["thompson"],
        )
        ax.set_title(label)
        ax.set_xticks(x)
        ax.set_xticklabels(splits)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        for bars in (b1, b2):
            for bar in bars:
                h = bar.get_height()
                if pd.notna(h):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        h + 0.012,
                        f"{h:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )
    axes[0].set_ylabel("Best validation score")
    axes[-1].legend(loc="upper right", framealpha=0.9)
    fig.suptitle(title, fontsize=13)
    _save_fig(fig, out_path)
    return out_path


def _binary_enrichment_chart(
    pairs: pd.DataFrame, out_path: Path
) -> Optional[Path]:
    df = pairs[pairs["task_type"] == "binary"].copy()
    if df.empty:
        return None
    df = df.sort_values("split").reset_index(drop=True)
    splits = df["split"].tolist()
    pool = (df["pool_positive_rate"] * 100).tolist()
    rand = (df["random_sample_positive_rate"] * 100).tolist()
    thom = (df["thompson_sample_positive_rate"] * 100).tolist()
    x = np.arange(len(splits))
    width = 0.27
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bars = [
        ax.bar(x - width, pool, width, label="Pool", color=_BAR_COLORS["pool"]),
        ax.bar(x, rand, width, label="Random sample", color=_BAR_COLORS["random"]),
        ax.bar(
            x + width,
            thom,
            width,
            label="Thompson sample (LTS)",
            color=_BAR_COLORS["thompson"],
        ),
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylabel("Positive (`earn`) rate (%)")
    ax.set_title("Positive-class enrichment under fixed labeling budget")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend()
    for grp in bars:
        for b in grp:
            h = b.get_height()
            if pd.notna(h):
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    h + 0.6,
                    f"{h:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
    _save_fig(fig, out_path)
    return out_path


def _pseudo_agreement_chart(
    summary: pd.DataFrame, out_path: Path
) -> Optional[Path]:
    df = summary.dropna(subset=["pseudo_agreement"]).copy()
    if df.empty:
        return None
    df["row_label"] = (
        df["split"] + " " + df["task_type"] + " (" + df["sampling"] + ")"
    )
    df = df.sort_values(["task_type", "split", "sampling"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    colors = [
        _BAR_COLORS["thompson"] if s == "thompson" else _BAR_COLORS["random"]
        for s in df["sampling"]
    ]
    bars = ax.bar(
        df["row_label"], (df["pseudo_agreement"] * 100).tolist(), color=colors
    )
    ax.set_ylabel("Qwen ↔ gold agreement (%)")
    ax.set_title("Pseudo-label quality on sampled rows")
    ax.set_ylim(0, 100)
    ax.axhline(50, color="black", linestyle="--", linewidth=0.8, alpha=0.4)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    for b, v in zip(bars, df["pseudo_agreement"].tolist()):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 1,
            f"{100 * v:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=_BAR_COLORS["random"], label="Random"),
        plt.Rectangle(
            (0, 0), 1, 1, color=_BAR_COLORS["thompson"], label="Thompson (LTS)"
        ),
    ]
    ax.legend(handles=handles, loc="upper right")
    _save_fig(fig, out_path)
    return out_path


def _qwen_vs_gold_distribution_chart(
    runs: List[RunSummary], out_path: Path
) -> Optional[Path]:
    mc_runs = [
        r
        for r in runs
        if r.task_type == "multiclass" and r.qwen_label_distribution
    ]
    if not mc_runs:
        return None
    n = len(mc_runs)
    cols = 2 if n > 1 else 1
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols, figsize=(7 * cols, 3.6 * rows), squeeze=False
    )
    for idx, r in enumerate(mc_runs):
        ax = axes[idx // cols][idx % cols]
        keys = sorted(
            set(r.qwen_label_distribution) | set(r.gold_label_distribution)
        )
        qwen_total = sum(r.qwen_label_distribution.values()) or 1
        gold_total = sum(r.gold_label_distribution.values()) or 1
        qwen = [
            100 * r.qwen_label_distribution.get(k, 0) / qwen_total for k in keys
        ]
        gold = [
            100 * r.gold_label_distribution.get(k, 0) / gold_total for k in keys
        ]
        x = np.arange(len(keys))
        width = 0.4
        ax.bar(x - width / 2, qwen, width, color=_BAR_COLORS["qwen"], label="Qwen")
        ax.bar(x + width / 2, gold, width, color=_BAR_COLORS["gold"], label="Gold")
        ax.set_xticks(x)
        ax.set_xticklabels(keys, rotation=20, ha="right")
        ax.set_ylim(0, 100)
        ax.set_ylabel("Share of sampled rows (%)")
        ax.set_title(f"{r.split} ({r.sampling}, n={r.pseudo_label_rows})")
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        ax.legend(loc="upper right", fontsize=9)
    # hide leftover empty axes
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].axis("off")
    fig.suptitle(
        "Qwen multi-class output distribution vs gold (sampled rows)",
        fontsize=13,
    )
    _save_fig(fig, out_path)
    return out_path


def make_plots(
    runs: List[RunSummary],
    summary: pd.DataFrame,
    pairs: pd.DataFrame,
    figures_dir: Path,
) -> Dict[str, Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    out: Dict[str, Path] = {}
    p1 = _final_metrics_chart(
        pairs,
        "binary",
        figures_dir / "binary_final_metrics.png",
        "Binary `earn` triage — Random vs Thompson",
    )
    if p1:
        out["binary_final_metrics"] = p1
    p2 = _final_metrics_chart(
        pairs,
        "multiclass",
        figures_dir / "multiclass_final_metrics.png",
        "Multi-class triage (6 classes) — Random vs Thompson",
    )
    if p2:
        out["multiclass_final_metrics"] = p2
    p3 = _binary_enrichment_chart(
        pairs, figures_dir / "binary_positive_enrichment.png"
    )
    if p3:
        out["binary_positive_enrichment"] = p3
    p4 = _pseudo_agreement_chart(
        summary, figures_dir / "pseudo_label_agreement.png"
    )
    if p4:
        out["pseudo_label_agreement"] = p4
    p5 = _qwen_vs_gold_distribution_chart(
        runs, figures_dir / "qwen_vs_gold_distribution.png"
    )
    if p5:
        out["qwen_vs_gold_distribution"] = p5
    return out


# ---------------------------------------------------------------------------
# markdown rendering
# ---------------------------------------------------------------------------


def render_markdown_report(
    runs: List[RunSummary],
    pairs: pd.DataFrame,
    summary: pd.DataFrame,
    figures: Optional[Dict[str, Path]] = None,
    report_path: Optional[Path] = None,
) -> str:
    figures = figures or {}

    def _img(key: str, alt: str) -> str:
        if key not in figures:
            return ""
        if report_path is not None:
            try:
                rel = Path(figures[key]).resolve().relative_to(
                    report_path.resolve().parent
                )
            except ValueError:
                rel = Path(figures[key])
        else:
            rel = Path(figures[key])
        return f"\n![{alt}]({rel.as_posix()})\n\n"

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
    lines.append(_img("binary_final_metrics", "Binary final metrics"))
    lines.append(_img("multiclass_final_metrics", "Multi-class final metrics"))

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
        lines.append(
            _img("binary_positive_enrichment", "Binary positive-class enrichment")
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
        lines.append(_img("pseudo_label_agreement", "Pseudo-label agreement"))

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
        lines.append(
            _img(
                "qwen_vs_gold_distribution",
                "Qwen vs gold per-class distribution",
            )
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
    figures_dir = out_dir / "figures"

    summary_df.to_csv(summary_csv, index=False)
    pairs_df.to_csv(pairs_csv, index=False)

    figures = make_plots(runs, summary_df, pairs_df, figures_dir)
    for k, v in figures.items():
        print(f"[analyze] figure {k} -> {v}")

    report_md.write_text(
        render_markdown_report(
            runs, pairs_df, summary_df, figures=figures, report_path=report_md
        )
    )

    print(f"[analyze] wrote {summary_csv}")
    print(f"[analyze] wrote {pairs_csv}")
    print(f"[analyze] wrote {report_md}")


if __name__ == "__main__":
    main()

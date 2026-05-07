"""Validation metrics stratified by LDA cluster (label_cluster column)."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def _float(x):
    return float(x) if x is not None and not (isinstance(x, float) and np.isnan(x)) else None


def _metrics_for_slice(y_true: np.ndarray, y_pred: np.ndarray, num_labels: int) -> Dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    n = len(y_true)
    row: Dict[str, Any] = {"n": int(n)}
    if n == 0:
        row["skipped_reason"] = "empty"
        return row

    uniq_true = np.unique(y_true)
    if len(uniq_true) < 2 and num_labels == 2:
        row["skipped_reason"] = "single_class_in_slice"
        row["label_distribution"] = {str(int(k)): int(np.sum(y_true == k)) for k in uniq_true}
        return row

    row["accuracy"] = _float(accuracy_score(y_true, y_pred))
    row["precision_weighted"] = _float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    row["recall_weighted"] = _float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    row["f1_weighted"] = _float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    row["precision_macro"] = _float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    row["recall_macro"] = _float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    row["f1_macro"] = _float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    if num_labels <= 2 and len(uniq_true) >= 2:
        row["precision_pos"] = _float(precision_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0))
        row["recall_pos"] = _float(recall_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0))
        row["f1_pos"] = _float(f1_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0))

    return row


def compute_validation_per_cluster(
    trainer: Any,
    df: pd.DataFrame,
    num_labels: int,
    cluster_col: str = "label_cluster",
    label_col: str = "label",
    min_support: int = 5,
) -> Dict[str, Any]:
    if getattr(trainer, "trainer", None) is None:
        return {"error": "trainer_not_ready"}

    if cluster_col not in df.columns or label_col not in df.columns:
        return {"error": "missing_columns", "cluster_col": cluster_col, "label_col": label_col}

    work = df.reset_index(drop=True)
    preds = trainer.get_inference(work).cpu().numpy().astype(int)
    y_all = work[label_col].to_numpy().astype(int)

    out: Dict[str, Any] = {}
    cluster_vals = pd.unique(work[cluster_col].dropna())
    try:
        cluster_vals = sorted(cluster_vals, key=lambda x: int(x))
    except (TypeError, ValueError):
        cluster_vals = sorted(cluster_vals, key=lambda x: str(x))

    for cid in cluster_vals:
        key = str(int(cid)) if isinstance(cid, (np.integer, int)) or (isinstance(cid, float) and cid == int(cid)) else str(cid)
        mask = work[cluster_col].to_numpy() == cid
        sub_n = int(mask.sum())
        if sub_n < min_support:
            out[key] = {"n": sub_n, "skipped_reason": "min_support", "min_support": min_support}
            continue
        out[key] = _metrics_for_slice(y_all[mask], preds[mask], num_labels)

    return out
#!/usr/bin/env python3
"""
Run the binary one-vs-rest classifier pipeline once per target emotion.

Usage:
  cd emotions_rec
  python scripts/run_binary_per_target.py --targets sadness,fear
  python scripts/run_binary_per_target.py --all
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
_EXP = os.path.dirname(_ROOT)
_SRC = os.path.join(_EXP, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from labeling import EMOTION_MAP  # noqa: E402


def run_binary_classifiers_for_targets(
    targets: list[str],
    *,
    smoke: bool = True,
    prepare_data: bool = True,
    dry_run: bool = False,
) -> None:
    """Run data prep (optional) then ``main_cluster_emotion_binary`` for each emotion in *targets*."""
    os.chdir(_EXP)
    proc_env = {**os.environ, "PYTHONPATH": _SRC}

    for name in targets:
        if name not in EMOTION_MAP:
            raise ValueError(f"unknown target {name!r}; choose from {sorted(EMOTION_MAP)}")

        base = f"emotions_{name}"
        suffix = f"{base}_smoke_train" if smoke else f"{base}_train"
        filename = os.path.join("data", "processed", suffix)
        val_name = f"{base}_smoke_validation.csv" if smoke else f"{base}_validation.csv"
        val_path = os.path.join("data", "processed", val_name)

        if prepare_data:
            cmd_p = [sys.executable, "scripts/prepare_emotions_binary.py", "--label", name]
            print("\n=== prepare:", " ".join(cmd_p), "===\n")
            if not dry_run:
                subprocess.run(cmd_p, cwd=_EXP, check=True)

        cmd = [
            sys.executable,
            "-u",
            "src/main_cluster_emotion_binary.py",
            "-sample_size",
            "200",
            "-filename",
            filename,
            "-val_path",
            val_path,
            "-balance",
            "False",
            "-sampling",
            "thompson",
            "-filter_label",
            "True",
            "-model_finetune",
            "bert-base-uncased",
            "-labeling",
            "qwen",
            "-model",
            "text",
            "-baseline",
            "0.10",
            "-metric",
            "f1_pos",
            "-cluster_size",
            "8",
            "-positive_label",
            name,
            "-hf_model_id",
            "Qwen/Qwen2.5-3B-Instruct",
            "-max_iterations",
            "3",
            "-num_train_epochs",
            "2",
            "-max_length",
            "128",
            "-batch_size",
            "16",
            "-confidence_threshold",
            "0.40",
            "-run_id",
            name,
            "-few_shot_from_csv",
            filename + ".csv",
        ]
        print("\n=== train:", name, "===\n", " ".join(cmd), "\n")
        if not dry_run:
            subprocess.run(cmd, cwd=_EXP, check=True, env=proc_env)


def _parse_targets(raw: str) -> list[str]:
    parts = [p.strip() for p in raw.replace(",", " ").split() if p.strip()]
    return parts


def main() -> None:
    p = argparse.ArgumentParser(description="Binary classifier for each target emotion")
    p.add_argument(
        "--targets",
        default="",
        help="Comma/space-separated emotion names (required unless --all).",
    )
    p.add_argument("--all", action="store_true", help="Run all six EMOTION_MAP keys.")
    p.add_argument("--full", action="store_true", help="Use full train/val CSVs instead of smoke splits.")
    p.add_argument("--no-prepare", action="store_true", help="Skip prepare_emotions_binary per target.")
    p.add_argument("--dry-run", action="store_true", help="Print commands only.")
    args = p.parse_args()

    targets = sorted(EMOTION_MAP.keys()) if args.all else _parse_targets(args.targets)
    if not targets:
        p.error("provide --targets or use --all")
    run_binary_classifiers_for_targets(
        targets,
        smoke=not args.full,
        prepare_data=not args.no_prepare,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

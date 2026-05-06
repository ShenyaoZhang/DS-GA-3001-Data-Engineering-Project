#!/usr/bin/env python3
"""
Fast sanity checks for emotions_rec (seconds, no training, no GPU).

Run from anywhere:
  python scripts/smoke_check_emotions_rec.py
or:
  cd emotions_rec && python scripts/smoke_check_emotions_rec.py
"""

from __future__ import annotations

import os
import py_compile
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
SCRIPTS = os.path.join(ROOT, "scripts")


def fail(msg: str) -> None:
    print("FAIL:", msg, file=sys.stderr)
    sys.exit(1)


def ok(msg: str) -> None:
    print("OK ", msg)


def main() -> None:
    os.chdir(ROOT)

    for folder, label in [(SRC, "src"), (SCRIPTS, "scripts")]:
        if not os.path.isdir(folder):
            fail(f"missing {label}/ ({folder})")
        for name in sorted(os.listdir(folder)):
            if not name.endswith(".py"):
                continue
            path = os.path.join(folder, name)
            try:
                py_compile.compile(path, doraise=True)
            except py_compile.PyCompileError as e:
                fail(f"syntax {path}: {e}")
    ok("py_compile all .py under src/ and scripts/")

    # prepare script: must resolve labeling without PYTHONPATH (we fixed sys.path in-file)
    r = subprocess.run(
        [sys.executable, os.path.join(SCRIPTS, "prepare_emotions_binary.py"), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        fail(f"prepare_emotions_binary.py --help: {r.stderr or r.stdout}")
    ok("prepare_emotions_binary.py --help")

    for script in ("eval_emotion_binary.py", "main_cluster_emotion_binary.py"):
        r = subprocess.run(
            [sys.executable, os.path.join(SRC, script), "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            fail(f"{script} --help: {r.stderr or r.stdout}")
        ok(f"{script} --help")

    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as e:
        print("WARN: torch/transformers not importable (OK for syntax-only; need for training):", e)
    else:
        ok("import torch, transformers")

    print(
        "NOTE: This package supports binary one-vs-rest for any emotion label. "
        "Colab: notebooks/emotion_rec_joy_repro.ipynb"
    )

    print("\nAll quick checks passed. Training still needs GPU + time; this did not run the model.")


if __name__ == "__main__":
    main()

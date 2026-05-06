#!/usr/bin/env python3
"""Write few-shot JSON from a prepared emotions CSV (see src/few_shot_from_dataset.py)."""

from __future__ import annotations

import os
import sys

_EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_EXP, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from few_shot_from_dataset import main_cli  # noqa: E402

if __name__ == "__main__":
    main_cli()

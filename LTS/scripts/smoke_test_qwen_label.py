"""One-shot smoke test: load Qwen and run get_qwen_label on a synthetic row."""
import os
import sys

# Smaller checkpoint for faster download/CPU inference during CI/local smoke tests.
os.environ.setdefault("QWEN_MODEL_DIR", "Qwen/Qwen2.5-0.5B-Instruct")

# Project root = parent of scripts/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from labeling import Labeling  # noqa: E402


def main() -> None:
    lab = Labeling(label_model="qwen")
    lab.set_model()
    model_id = os.environ.get("QWEN_MODEL_DIR")
    print("model:", model_id)

    for title in (
        "ALLIED-LYONS YEAR PRETAX PROFIT RISES",
        "NATIONAL AVERAGE PRICES FOR FARMER-OWNED RESERVE",
    ):
        row = {"text": lab.generate_prompt(title)}
        out = lab.get_qwen_label(row)
        print("title:", title[:60] + ("…" if len(title) > 60 else ""))
        print("  label:", repr(out))
        if out not in ("earn", "not earn"):
            raise SystemExit(f"unexpected label (must be earn|not earn): {out!r}")

    print("smoke_test_ok: load + generate + parse succeeded")
    print(
        "note: tiny models (e.g. 0.5B) may misclassify; use Qwen2.5-1.5B-Instruct or larger for accuracy."
    )


if __name__ == "__main__":
    main()

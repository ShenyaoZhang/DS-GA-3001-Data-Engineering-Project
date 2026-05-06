"""
Binary active learning for a selected emotion (one-vs-rest).
"""

import argparse
import gc
import json
import os
import random

import numpy as np
import pandas as pd
import torch

from LDA import LDATopicModel
from fine_tune import BertFineTuner
from labeling import EMOTION_MAP, Labeling
from preprocessing import TextPreprocessor
from random_sampling import RandomSampler
from thompson_sampling import ThompsonSampler


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def parse_args():
    parser = argparse.ArgumentParser(description="Binary minority emotion active learning")
    parser.add_argument("-sample_size", type=int, default=200)
    parser.add_argument("-filename", type=str, required=True)
    parser.add_argument("-val_path", type=str, required=True)
    parser.add_argument("-balance", type=str2bool, default=False)
    parser.add_argument("-sampling", type=str, default="thompson", choices=["thompson", "random"])
    parser.add_argument("-filter_label", type=str2bool, default=True)
    parser.add_argument("-model_finetune", type=str, default="bert-base-uncased")
    parser.add_argument("-labeling", type=str, default="qwen")
    parser.add_argument("-model", type=str, default="text")
    parser.add_argument("-metric", type=str, default="f1_pos")
    parser.add_argument(
        "-baseline",
        type=float,
        default=0.10,
        help="First-iteration hurdle for reward=score−baseline only (use a low value, e.g. 0.05–0.10 for f1_pos). "
        "Do NOT use 0.5—validation f1_pos is rarely that high.",
    )
    parser.add_argument("-cluster_size", type=int, default=8)
    parser.add_argument("-positive_label", type=str, required=True, choices=sorted(EMOTION_MAP.keys()))
    parser.add_argument("-hf_model_id", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("-few_shot_path", type=str, default=None)
    parser.add_argument(
        "-few_shot_from_csv",
        type=str,
        default=None,
        help="Prepared train CSV with title + raw_label; build stratified few-shot Qwen examples (overrides -few_shot_path when set)",
    )
    parser.add_argument("-few_shot_n_per_class", type=int, default=2)
    parser.add_argument("-few_shot_max_examples", type=int, default=8)
    parser.add_argument("-few_shot_seed", type=int, default=42)
    parser.add_argument("-seed", type=int, default=42, help="RNG seed (numpy/torch/HF Trainer; few-shot uses -few_shot_seed)")
    parser.add_argument("-max_iterations", type=int, default=3)
    parser.add_argument("-confidence_threshold", type=float, default=0.40)
    parser.add_argument("-num_train_epochs", type=int, default=2)
    parser.add_argument("-max_length", type=int, default=128)
    parser.add_argument("-batch_size", type=int, default=16)
    parser.add_argument(
        "-run_id",
        type=str,
        default="",
        help="Separate wins/selected_ids/training caches and model subdirectory (use 'thomp' vs 'rand' between baseline runs).",
    )
    parser.add_argument(
        "-random_uniform_pool",
        type=str2bool,
        default=False,
        help="When sampling=random: draw i.i.d. from the unlabeled pool (ignores LDA clusters). Fair cheap baseline vs Thompson+LDA.",
    )
    return parser.parse_args()


def ensure_dirs():
    for path in ["models", "data", "log", "results"]:
        os.makedirs(path, exist_ok=True)


def to_binary(label_series, target_id):
    return label_series.astype(int).apply(lambda x: 1 if x == target_id else 0)


def merge_raw_label_from_base(data: pd.DataFrame, base_csv: str) -> pd.DataFrame:
    """Old *_lda.csv caches may lack raw_label; join from the prepared train CSV when present."""
    if "raw_label" in data.columns or not os.path.isfile(base_csv):
        return data
    base = pd.read_csv(base_csv)
    if "raw_label" not in base.columns:
        return data
    out = data.drop(columns=["raw_label"], errors="ignore")
    return out.merge(base[["id", "raw_label"]], on="id", how="left")


def ovr_binary_labels(df: pd.DataFrame, target_id: int) -> pd.Series:
    """Use original 6-class ids when available; else legacy column `label` (multiclass or binary)."""
    src = df["raw_label"] if "raw_label" in df.columns else df["label"]
    return to_binary(src, target_id)


def load_or_build_lda(filename, cluster_size, preprocessor, target_id):
    lda_path = filename + "_lda.csv"
    base_csv = filename + ".csv"
    if os.path.exists(lda_path):
        data = pd.read_csv(lda_path)
        print("using data saved on disk")
        data = merge_raw_label_from_base(data, base_csv)
        data["label"] = ovr_binary_labels(data, target_id)
    else:
        print("Creating LDA")
        data = pd.read_csv(base_csv)
        data = preprocessor.preprocess_df(data)
        data["label"] = ovr_binary_labels(data, target_id)
        lda = LDATopicModel(num_topics=cluster_size)
        data["label_cluster"] = lda.fit_transform(data["clean_title"].to_list())
        data.to_csv(lda_path, index=False)
    return data, int(data["label_cluster"].nunique())


def prepare_validation(path, preprocessor, target_id):
    validation = pd.read_csv(path)
    validation = preprocessor.preprocess_df(validation)
    validation["training_text"] = validation["clean_title"] if "clean_title" in validation.columns else validation["title"]
    validation["label"] = ovr_binary_labels(validation, target_id)
    print(f"Validation binary labels: {validation['label'].value_counts().to_dict()}")
    return validation


def create_sampler(name, n_cluster, run_id="", random_uniform_pool=False):
    rid = (run_id or "").strip()
    if name == "thompson":
        return ThompsonSampler(n_cluster, run_id=rid)
    return RandomSampler(n_cluster, run_id=rid, uniform_pool=random_uniform_pool)


def label_batch(sample_data, args, target_id):
    labeler = Labeling(
        label_model=args.labeling,
        target_class=args.positive_label,
        model_id=args.hf_model_id,
        few_shot_path=args.few_shot_path,
        few_shot_from_csv=args.few_shot_from_csv,
        few_shot_n_per_class=args.few_shot_n_per_class,
        few_shot_max_examples=args.few_shot_max_examples,
        few_shot_seed=args.few_shot_seed,
    )
    labeler.set_model()
    df = labeler.generate_inference_data(sample_data, "clean_title")
    print("df for inference created")
    df["answer"] = df.apply(lambda row: labeler.predict_animal_product(row), axis=1)
    df = df[df["answer"].notna()].copy()
    df["answer"] = df["answer"].astype(str).str.strip()
    df = df[df["answer"].str.match(r"^[0-5]$")].copy()
    df["label"] = df["answer"].astype(int).apply(lambda x: 1 if x == target_id else 0)
    df["pseudo_label"] = df["label"]
    df["training_text"] = df["title"]

    try:
        del labeler.model
        del labeler.tokenizer
    except Exception:
        pass
    del labeler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return df


def maybe_filter_confidence(df, trainer, threshold):
    if trainer.trainer is None or len(df) == 0:
        return df
    if "label" in df.columns and (df["label"] == 1).sum() < 12 and len(df) > 24:
        print("Confidence filter skipped (keep rare positive pseudo-labels).")
        return df
    _, confs = trainer.get_inference_with_probs(df)
    keep = (confs >= threshold).numpy()
    if keep.sum() < max(int(0.3 * len(df)), 10):
        print("Confidence filter skipped (too few rows would remain).")
        return df
    filtered = df[keep].reset_index(drop=True)
    print(f"Confidence filter: kept {len(filtered)}/{len(df)}")
    return filtered


def run(args):
    set_seed(args.seed)
    ensure_dirs()
    target_id = EMOTION_MAP[args.positive_label]
    print(f"Target emotion: {args.positive_label} ({target_id})")

    rid = args.run_id.strip()
    sep = f"_{rid}" if rid else ""
    output_prefix = f"{args.filename}_binary_{args.positive_label}"
    training_path = f"{output_prefix}{sep}_training_data.csv"
    model_results_path = f"{output_prefix}{sep}_model_results.json"
    model_root = os.path.join("models", rid) if rid else "models"
    results_dir = os.path.join("results", rid) if rid else "results"
    log_dir = os.path.join("log", rid) if rid else "log"
    os.makedirs(model_root, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    if args.baseline > 0.25 and args.metric.endswith("pos"):
        print(
            f"[warn] -baseline={args.baseline:g} is very high for `{args.metric}`; "
            f"runs often report eval_f1_pos < 0.4. Prefer -baseline 0.05–0.10 so iteration 1 can save."
        )

    preprocessor = TextPreprocessor()
    validation = prepare_validation(args.val_path, preprocessor, target_id)
    data, n_cluster = load_or_build_lda(args.filename, args.cluster_size, preprocessor, target_id)

    trainer = BertFineTuner(
        args.model_finetune,
        None,
        validation,
        confidence_threshold=args.confidence_threshold,
        num_labels=2,
        max_length=args.max_length,
        num_train_epochs=args.num_train_epochs,
        monitor_metric=f"eval_{args.metric}",
        batch_size=args.batch_size,
        results_dir=results_dir,
        log_dir=log_dir,
        seed=args.seed,
    )
    sampler = create_sampler(args.sampling, n_cluster, args.run_id, args.random_uniform_pool)
    baseline = args.baseline

    for i in range(args.max_iterations):
        print(f"\n{'='*70}")
        print(f"Binary iteration {i + 1} / {args.max_iterations}")
        print(f"{'='*70}")
        sample_data, chosen_bandit = sampler.get_sample_data(data, args.sample_size, args.filter_label, trainer)
        print(f"Bandit: {chosen_bandit} | sampled rows: {len(sample_data)}")

        df = sample_data.copy() if args.labeling == "file" else label_batch(sample_data, args, target_id)
        df = maybe_filter_confidence(df, trainer, args.confidence_threshold)
        print(f"Pseudo-label distribution: {df['label'].value_counts().to_dict()}")

        if os.path.exists(training_path):
            prev = pd.read_csv(training_path)
            prev["training_text"] = prev.get("training_text", prev.get("clean_title", prev.get("title", "")))
            df = pd.concat([df, prev], ignore_index=True).drop_duplicates(subset=["id"])
            print(f"Accumulated training data: {len(df)}")

        last = trainer.get_last_model_acc()
        if last:
            baseline = list(last.values())[0]
            print(f"Previous {args.metric} baseline: {baseline:.6f}")
        else:
            print(f"Starting baseline ({args.metric}): {baseline:.6f}")

        results, _ = trainer.train_data(df, still_unbalenced=True)
        score = results[f"eval_{args.metric}"]
        reward = score - baseline
        print(f"Iteration summary | eval_{args.metric}: {score:.6f} | baseline: {baseline:.6f} | reward: {reward:.6f}")

        if reward > 0:
            model_name = os.path.join(
                model_root, f"binary_{args.positive_label}_fine_tunned_{i}_bandit_{chosen_bandit}"
            )
            trainer.update_model(model_name, score, save_model=True)
            df.to_csv(training_path, index=False)
            if args.filter_label:
                trainer.set_clf(True)
            print(f"Model improved and saved: {model_name}")
        else:
            trainer.reset_to_initial_pretrained()
            print("No improvement this round (BERT reset to pretrained; last_model_acc unchanged).")

        history = (
            json.load(open(model_results_path, encoding="utf-8"))
            if os.path.exists(model_results_path)
            else {}
        )
        history.setdefault(str(chosen_bandit), []).append(results)
        with open(model_results_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        if args.sampling == "thompson":
            sampler.update(chosen_bandit if chosen_bandit != "fallback_random" else 0, reward)

    if hasattr(sampler, "wins"):
        ratio = sampler.wins / np.maximum(sampler.wins + sampler.losses, 1e-8)
        arm = int(np.argmax(ratio))
        print(f"Thompson arm highest win-rate (not argmax checkpoint): arm={arm}")


if __name__ == "__main__":
    run(parse_args())

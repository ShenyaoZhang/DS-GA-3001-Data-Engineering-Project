"""
Sentiment-only active learning pipeline (negative / neutral / positive).

Design mirrors the short-loop style used by LTS/main_cluster.py:
  cluster -> sample -> pseudo-label -> fine-tune -> update sampler
"""

import argparse
import contextlib
import gc
import json
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from LDA import LDATopicModel
from fine_tune import BertFineTuner
from labeling import Labeling
from preprocessing import TextPreprocessor
from random_sampling import RandomSampler
from thompson_sampling import ThompsonSampler
from sentiment_labels import EMOTION_TO_SENTIMENT


class SentimentLabeling(Labeling):
    """Qwen prompt/output parsing for 3-class sentiment labels."""

    def _base_prompt(self, title: str) -> str:
        short = self._clean_for_prompt(title, max_chars=220)
        return (
            "Classify the sentiment of the text.\n"
            "0 = negative (sadness, anger, fear, frustration, anxiety)\n"
            "1 = neutral (mixed/unclear emotion, surprise without clear valence)\n"
            "2 = positive (joy, love, gratitude, excitement)\n"
            "Output only one number: 0, 1, or 2.\n\n"
            f"Document: {short}\nLabel:"
        )

    def _extract_label(self, response_text: str) -> str:
        txt = str(response_text).strip().lower()
        match = re.search(r"\b([012])\b", txt)
        if match:
            return match.group(1)
        if any(w in txt for w in ["negative", "sad", "anger", "fear", "anxiety"]):
            return "0"
        if any(w in txt for w in ["neutral", "mixed", "unclear", "surprise"]):
            return "1"
        if any(w in txt for w in ["positive", "joy", "love", "happy", "gratitude"]):
            return "2"
        return None

    def get_qwen_label(self, row):
        user_prompt = row["text"]
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a sentiment classifier. "
                    "Classify as 0=negative, 1=neutral, 2=positive. "
                    "Output exactly one number."
                ),
            },
            {"role": "user", "content": user_prompt},
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer(
            [text], return_tensors="pt", truncation=True, max_length=768
        ).to(self.model.device)
        input_len = model_inputs.input_ids.shape[1]
        with torch.inference_mode():
            outputs = self.model.generate(
                **model_inputs, max_new_tokens=3, do_sample=False, pad_token_id=self.tokenizer.eos_token_id
            )
        response = self.tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)[0]
        return self._extract_label(response)


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {value}")


def parse_args():
    parser = argparse.ArgumentParser(description="Sentiment active learning (3-class)")
    parser.add_argument("-sampling", type=str, default="thompson")
    parser.add_argument("-sample_size", type=int, default=300)
    parser.add_argument("-filter_label", type=str2bool, default=True)
    parser.add_argument("-model_finetune", type=str, default="bert-base-uncased")
    parser.add_argument("-labeling", type=str, default="qwen")
    parser.add_argument("-filename", type=str, required=True)
    parser.add_argument("-model", type=str, default="text")
    parser.add_argument("-metric", type=str, default="f1_macro")
    parser.add_argument("-val_path", type=str, required=True)
    parser.add_argument("-cluster_size", type=int, default=10)
    parser.add_argument("-target_class", type=str, default="joy")
    parser.add_argument("-few_shot_path", type=str, default=None)
    parser.add_argument("-hf_model_id", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("-max_iterations", type=int, default=8)
    parser.add_argument("-confidence_threshold", type=float, default=0.35)
    parser.add_argument("-baseline", type=float, default=(1.0 / 3.0))
    parser.add_argument("-outputs_dir", type=str, default="outputs")
    parser.add_argument("-console_logs", type=str2bool, default=False)
    return parser.parse_args()


def ensure_dirs(outputs_dir):
    for path in ["models", "data", "log", "results", outputs_dir]:
        os.makedirs(path, exist_ok=True)


def build_log_path(outputs_dir):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(outputs_dir, f"sentiment_train_{stamp}.log")


def load_or_build_lda(filename, cluster_size, preprocessor):
    lda_path = filename + "_lda.csv"
    try:
        data = pd.read_csv(lda_path)
        n_cluster = data["label_cluster"].value_counts().count()
        print("using data saved on disk")
        return data, n_cluster
    except Exception:
        print("Creating LDA")
        data = pd.read_csv(filename + ".csv")
        data = preprocessor.preprocess_df(data)
        lda_model = LDATopicModel(num_topics=cluster_size)
        data["label_cluster"] = lda_model.fit_transform(data["clean_title"].to_list())
        n_cluster = data["label_cluster"].value_counts().count()
        data.to_csv(lda_path, index=False)
        print(f"LDA created with {n_cluster} clusters")
        return data, n_cluster


def prepare_validation(path, preprocessor):
    validation = pd.read_csv(path)
    validation = preprocessor.preprocess_df(validation)
    validation["training_text"] = validation["clean_title"] if "clean_title" in validation.columns else validation["title"]
    if "label" in validation.columns:
        validation["label"] = validation["label"].map(EMOTION_TO_SENTIMENT)
        print(f"Validation sentiment labels: {validation['label'].value_counts().to_dict()}")
    return validation


def create_sampler(name, n_cluster):
    if name == "thompson":
        return ThompsonSampler(n_cluster)
    if name == "random":
        return RandomSampler(n_cluster)
    raise ValueError("Choose thompson or random")


def label_batch(sample_data, args):
    labeler = SentimentLabeling(
        label_model=args.labeling,
        target_class=args.target_class,
        model_id=args.hf_model_id,
        few_shot_path=args.few_shot_path,
    )
    labeler.set_model()
    df = labeler.generate_inference_data(sample_data, "clean_title")
    print("df for inference created")
    df["answer"] = df.apply(lambda row: labeler.predict_animal_product(row), axis=1)
    df = df[df["answer"].notna()].copy()
    df["answer"] = df["answer"].astype(str).str.strip()
    df = df[df["answer"].str.match(r"^[012]$")].copy()
    df["label"] = df["answer"].astype(int)
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
    _, confs = trainer.get_inference_with_probs(df)
    keep = (confs >= threshold).numpy()
    if keep.sum() < max(int(0.3 * len(df)), 10):
        print("Confidence filter skipped (too few rows would remain).")
        return df
    filtered = df[keep].reset_index(drop=True)
    print(f"Confidence filter: kept {len(filtered)}/{len(df)}")
    return filtered


def merge_with_history(df, training_path):
    if not os.path.exists(training_path):
        return df
    prev = pd.read_csv(training_path)
    if "training_text" not in prev.columns:
        prev["training_text"] = prev.get("clean_title", prev.get("title", ""))
    merged = pd.concat([df, prev], ignore_index=True)
    merged = merged.drop_duplicates(subset=["id"])
    print(f"Accumulated training data: {len(merged)}")
    return merged


def is_unbalanced(df):
    counts = df["label"].value_counts()
    ratio = counts.max() / max(counts.min(), 1)
    flag = ratio >= 2.0
    print(f"Unbalanced? {flag} | Distribution: {counts.to_dict()}")
    return flag


def save_round_results(path, chosen_bandit, results):
    existing = json.load(open(path)) if os.path.exists(path) else {}
    existing.setdefault(str(chosen_bandit), []).append(results)
    with open(path, "w") as file:
        json.dump(existing, file, indent=2)


def run_pipeline(args):
    ensure_dirs(args.outputs_dir)

    preprocessor = TextPreprocessor()
    validation = prepare_validation(args.val_path, preprocessor)
    data, n_cluster = load_or_build_lda(args.filename, args.cluster_size, preprocessor)
    if "label" in data.columns:
        data["label"] = data["label"].map(EMOTION_TO_SENTIMENT)

    trainer = BertFineTuner(
        args.model_finetune,
        None,
        validation,
        confidence_threshold=args.confidence_threshold,
        num_labels=3,
    )
    sampler = create_sampler(args.sampling, n_cluster)

    output_prefix = args.filename + "_sentiment"
    training_path = f"{output_prefix}_training_data.csv"
    model_results_path = f"{output_prefix}_model_results.json"
    baseline = args.baseline

    for i in range(args.max_iterations):
        print(f"\n{'='*70}")
        print(f"Sentiment iteration {i + 1} / {args.max_iterations}")
        print(f"{'='*70}")
        sample_data, chosen_bandit = sampler.get_sample_data(data, args.sample_size, args.filter_label, trainer)
        print(f"Bandit: {chosen_bandit} | sampled rows: {len(sample_data)}")

        if args.labeling == "file":
            df = sample_data.copy()
            if "training_text" not in df.columns:
                df["training_text"] = df.get("clean_title", df.get("title", ""))
        else:
            df = label_batch(sample_data, args)

        df = maybe_filter_confidence(df, trainer, args.confidence_threshold)
        print(df["label"].value_counts())

        df = merge_with_history(df, training_path)
        model_acc = trainer.get_last_model_acc()
        if model_acc:
            baseline = model_acc[trainer.get_base_model()]
            print(f"previous model {args.metric} baseline: {baseline}")
        else:
            print(f"Starting with {args.metric} baseline {baseline}")

        results, _ = trainer.train_data(df, is_unbalanced(df))
        reward = results[f"eval_{args.metric}"] - baseline
        print(
            f"Iteration summary | eval_{args.metric}: {results[f'eval_{args.metric}']:.6f} "
            f"| baseline: {baseline:.6f} | reward: {reward:.6f}"
        )

        if reward > 0:
            print(f"Model improved by {reward}")
            model_name = f"models/sentiment_fine_tunned_{i}_bandit_{chosen_bandit}"
            trainer.update_model(model_name, results[f"eval_{args.metric}"], save_model=True)
            df.to_csv(training_path, index=False)
            if args.filter_label:
                trainer.set_clf(True)
        else:
            trainer.update_model(trainer.get_base_model(), baseline, save_model=False)

        save_round_results(model_results_path, chosen_bandit, results)
        if args.sampling == "thompson":
            sampler.update(chosen_bandit if chosen_bandit != "fallback_random" else 0, reward)

    if hasattr(sampler, "wins"):
        ratio = sampler.wins / np.maximum(sampler.wins + sampler.losses, 1e-8)
        print(f"Best bandit: {int(np.argmax(ratio))}")


def main():
    args = parse_args()
    log_path = build_log_path(args.outputs_dir)
    ensure_dirs(args.outputs_dir)

    if args.console_logs:
        print(f"Logging to: {log_path}")
        run_pipeline(args)
        return

    with open(log_path, "w", encoding="utf-8") as log_file:
        with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
            print(f"Run started at: {datetime.now().isoformat()}")
            print(f"Logging to: {log_path}")
            run_pipeline(args)
            print(f"Run ended at: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()

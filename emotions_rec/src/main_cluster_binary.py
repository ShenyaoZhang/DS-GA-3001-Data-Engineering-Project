"""
Binary Emotion Classification Pipeline (Tier 1: positive vs negative)

Maps dair-ai/emotion labels to binary (see tiered_labels.EMOTION_TO_BINARY):
  negative (0): sadness, anger, fear
  positive (1): joy, love, surprise (surprise is positive here only; tier 2 then maps to joy/love)

Includes improvements over the base pipeline:
  - Binary Qwen prompts for better pseudo-label accuracy
  - Label smoothing (0.1)
  - Data accumulation across iterations
  - Reduced max_length (128) for faster training
  - Lower confidence threshold (0.35)
  - Epsilon-greedy Thompson Sampling exploration
  - Drops unparseable labels instead of defaulting to 0

Usage:
  python main_cluster_binary.py \
    -sample_size 300 \
    -filename "data/processed/train_inner_emotions_emotion" \
    -val_path "data/processed/val_emotions_emotion.csv" \
    -sampling "thompson" -filter_label True \
    -model_finetune "bert-base-uncased" -labeling "qwen" -model "text" \
    -baseline 0.5 -metric "f1_macro" -cluster_size 10 \
    -hf_model_id "Qwen/Qwen2.5-7B-Instruct" \
    -max_iterations 8 -confidence_threshold 0.35
"""

import argparse
import pandas as pd
import numpy as np
import re
import os
import json
import gc

import nltk
nltk.download("punkt", quiet=True)

from preprocessing import TextPreprocessor
from fine_tune import BertFineTuner
from thompson_sampling import ThompsonSampler
from random_sampling import RandomSampler
from LDA import LDATopicModel
from labeling import Labeling
from tiered_labels import EMOTION_TO_BINARY


# ---------- Binary-aware Labeling subclass ----------
class BinaryLabeling(Labeling):
    """Overrides prompts and label extraction for binary pos/neg classification."""

    def _base_prompt(self, title: str) -> str:
        examples = self._build_examples_text()
        short_title = self._clean_for_prompt(title, max_chars=200)

        prompt = (
            "Classify the overall sentiment of this text as positive or negative.\n\n"
            "0 = negative (sadness, grief, anger, frustration, fear, anxiety, worry, loneliness, despair, irritation)\n"
            "1 = positive (joy, happiness, love, affection, excitement, surprise, gratitude, delight, contentment)\n\n"
            "Output only the number (0 or 1).\n"
        )
        if examples:
            prompt += "Examples:\n" + examples + "\n\n"
        prompt += f"Document: {short_title}\nLabel:"
        return prompt

    def _build_examples_text(self):
        """Remap 6-class few-shot labels to binary."""
        if not self.few_shot_examples:
            return ""
        lines = []
        for ex in self.few_shot_examples[:8]:
            orig = int(ex["label"])
            binary = EMOTION_TO_BINARY.get(orig, orig)
            short_text = self._clean_for_prompt(ex["text"], max_chars=140)
            lines.append(f"Document: {short_text}")
            lines.append(f"Label: {binary}")
            lines.append("")
        return "\n".join(lines).strip()

    def _extract_label(self, response_text: str) -> str:
        response_text = str(response_text).strip().lower()
        match = re.search(r'\b([01])\b', response_text)
        if match:
            return match.group(1)
        for w in ["positive", "joy", "happy", "love", "surprise", "delight"]:
            if w in response_text:
                return "1"
        for w in ["negative", "sadness", "anger", "fear", "anxiety", "grief"]:
            if w in response_text:
                return "0"
        return None  # drop unparseable

    def get_qwen_label(self, row):
        user_prompt = row["text"]
        messages = [
            {"role": "system",
             "content": "You are a sentiment classifier. Classify the text as: 0=negative (sadness, anger, fear) or 1=positive (joy, love, surprise). Tier 2 will split positive into joy vs love. Output exactly one number."},
            {"role": "user", "content": user_prompt}
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt", truncation=True, max_length=768).to(self.model.device)
        input_len = model_inputs.input_ids.shape[1]
        import torch
        with torch.inference_mode():
            outputs = self.model.generate(**model_inputs, max_new_tokens=3, do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
        generated_ids = outputs[:, input_len:]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return self._extract_label(response)


# ---------- helpers ----------
def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).strip().lower()
    if v in {"true", "1", "yes", "y"}:
        return True
    if v in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {v}")


def main():
    parser = argparse.ArgumentParser(description="Binary (pos/neg) emotion classification pipeline")
    parser.add_argument("-sampling", type=str, required=False)
    parser.add_argument("-sample_size", type=int, required=False)
    parser.add_argument("-filter_label", type=str2bool, required=False)
    parser.add_argument("-balance", type=str2bool, required=False)
    parser.add_argument("-model_finetune", type=str, required=False)
    parser.add_argument("-labeling", type=str, required=False)
    parser.add_argument("-baseline", type=float, required=False, default=0.5)
    parser.add_argument("-filename", type=str, required=False)
    parser.add_argument("-model", type=str, required=False)
    parser.add_argument("-metric", type=str, required=False, default="f1_macro")
    parser.add_argument("-val_path", type=str, required=False)
    parser.add_argument("-cluster_size", type=int, required=False, default=10)
    parser.add_argument("-target_class", type=str, required=False, default="joy")
    parser.add_argument("-few_shot_path", type=str, required=False, default=None)
    parser.add_argument("-hf_model_id", type=str, required=False, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("-max_iterations", type=int, required=False, default=8)
    parser.add_argument("-confidence_threshold", type=float, required=False, default=0.35)
    args = parser.parse_args()

    NUM_LABELS = 2
    filename = args.filename
    metric = args.metric
    baseline = args.baseline

    preprocessor = TextPreprocessor()

    # --- validation: map to binary ---
    validation = pd.read_csv(args.val_path)
    validation = preprocessor.preprocess_df(validation)
    validation["training_text"] = validation["clean_title"] if "clean_title" in validation.columns else validation["title"]
    if "label" in validation.columns:
        validation["label"] = validation["label"].map(EMOTION_TO_BINARY)
        print(f"Validation mapped to binary: {validation['label'].value_counts().to_dict()}")

    for d in ["models", "data", "log", "results"]:
        os.makedirs(d, exist_ok=True)

    # --- LDA (reuse original data) ---
    try:
        data = pd.read_csv(filename + "_lda.csv")
        n_cluster = data["label_cluster"].value_counts().count()
        print("using data saved on disk")
    except Exception:
        print("Creating LDA")
        data = pd.read_csv(filename + ".csv")
        data = preprocessor.preprocess_df(data)
        lda = LDATopicModel(num_topics=args.cluster_size)
        data["label_cluster"] = lda.fit_transform(data["clean_title"].to_list())
        n_cluster = data["label_cluster"].value_counts().count()
        print(n_cluster)
        data.to_csv(filename + "_lda.csv", index=False)
        print("LDA created")

    trainer = BertFineTuner(args.model_finetune, None, validation, confidence_threshold=args.confidence_threshold, num_labels=NUM_LABELS)

    if args.sampling == "thompson":
        sampler = ThompsonSampler(n_cluster)
    elif args.sampling == "random":
        sampler = RandomSampler(n_cluster)
    else:
        raise ValueError("Choose thompson or random")

    output_prefix = filename + "_binary"

    for i in range(args.max_iterations):
        sample_data, chosen_bandit = sampler.get_sample_data(data, args.sample_size, args.filter_label, trainer)

        if args.labeling != "file":
            labeler = BinaryLabeling(
                label_model=args.labeling, target_class=args.target_class,
                model_id=args.hf_model_id, few_shot_path=args.few_shot_path,
            )
            labeler.set_model()
            df = labeler.generate_inference_data(sample_data, "clean_title")
            print("df for inference created")
            df["answer"] = df.apply(lambda x: labeler.predict_animal_product(x), axis=1)

            # Drop unparseable labels
            df = df[df["answer"].notna()].copy()
            df["answer"] = df["answer"].astype(str).str.strip()
            df = df[df["answer"].str.match(r'^[01]$')].copy()
            df["pseudo_label"] = df["answer"].astype(int)
            df["label"] = df["pseudo_label"]
            df["training_text"] = df["title"]

            if os.path.exists(f"{output_prefix}_data_labeled.csv"):
                prev = pd.read_csv(f"{output_prefix}_data_labeled.csv")
                pd.concat([prev, df], ignore_index=True).to_csv(f"{output_prefix}_data_labeled.csv", index=False)
            else:
                df.to_csv(f"{output_prefix}_data_labeled.csv", index=False)

            try:
                del labeler.model, labeler.tokenizer
            except Exception:
                pass
            del labeler
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        else:
            df = sample_data
            if "training_text" not in df.columns:
                df["training_text"] = df["clean_title"] if "clean_title" in df.columns else df["title"]

        # Confidence filter
        if trainer.trainer is not None and len(df) > 0:
            print("Filtering pseudo-labels by model confidence...")
            _, model_confs = trainer.get_inference_with_probs(df)
            mask = (model_confs >= args.confidence_threshold).numpy()
            if mask.sum() >= max(len(df) * 0.3, 10):
                df = df[mask].reset_index(drop=True)
                print(f"Confidence filter: kept {len(df)}/{mask.shape[0]}")
            else:
                print(f"Confidence filter skipped: only {mask.sum()}/{len(df)} passed")

        if "true_label" in df.columns and "pseudo_label" in df.columns:
            true_binary = df["true_label"].astype(float).fillna(-1).astype(int).map(EMOTION_TO_BINARY)
            agreement = (true_binary == df["pseudo_label"].astype(int)).mean()
            print(f"Pseudo-label agreement (binary): {agreement:.3f}")

        print(df["label"].value_counts())

        # Minority carry-over
        if os.path.exists("minority_data_binary.csv"):
            minority = pd.read_csv("minority_data_binary.csv")
            df = pd.concat([df, minority], ignore_index=True).sample(frac=1, random_state=42)

        # Accumulate training data
        if os.path.exists(f"{output_prefix}_training_data.csv"):
            prev_data = pd.read_csv(f"{output_prefix}_training_data.csv")
            if "training_text" not in prev_data.columns:
                prev_data["training_text"] = prev_data.get("clean_title", prev_data.get("title", ""))
            df = pd.concat([df, prev_data], ignore_index=True).drop_duplicates(subset=["id"])
            print(f"Accumulated training data: {len(df)} samples")

        model_name = trainer.get_base_model()
        model_results = trainer.get_last_model_acc()
        if model_results:
            baseline = model_results[model_name]
            print(f"previous model {metric} baseline: {baseline}")
        else:
            print(f"Starting with {metric} baseline {baseline}")

        label_counts = df["label"].value_counts()
        unbalanced = label_counts.max() / max(label_counts.min(), 1) >= 2
        print(f"Unbalanced? {unbalanced} | Distribution: {label_counts.to_dict()}")

        results, _ = trainer.train_data(df, unbalanced)
        reward = results[f"eval_{metric}"] - baseline

        if reward > 0:
            print(f"Model improved by {reward}")
            model_name = f"models/binary_fine_tunned_{i}_bandit_{chosen_bandit}"
            trainer.update_model(model_name, results[f"eval_{metric}"], save_model=True)
            if os.path.exists(f"{output_prefix}_training_data.csv"):
                prev = pd.read_csv(f"{output_prefix}_training_data.csv")
                df = pd.concat([prev, df], ignore_index=True)
            df.to_csv(f"{output_prefix}_training_data.csv", index=False)
            if os.path.exists("minority_data_binary.csv"):
                os.remove("minority_data_binary.csv")
            if args.filter_label:
                trainer.set_clf(True)
        else:
            trainer.update_model(model_name, baseline, save_model=False)
            lc = df["label"].value_counts()
            minority_cls = lc[lc < lc.median()].index.tolist()
            mdf = df[df["label"].isin(minority_cls)]
            if os.path.exists("minority_data_binary.csv"):
                mdf = pd.concat([mdf, pd.read_csv("minority_data_binary.csv")], ignore_index=True).drop_duplicates()
            mdf.to_csv("minority_data_binary.csv", index=False)

        # Save results
        results_path = f"{output_prefix}_model_results.json"
        existing = json.load(open(results_path)) if os.path.exists(results_path) else {}
        existing.setdefault(str(chosen_bandit), []).append(results)
        json.dump(existing, open(results_path, "w"), indent=4)

        if args.sampling == "thompson":
            sampler.update(chosen_bandit if chosen_bandit != "fallback_random" else 0, reward)

    if hasattr(sampler, "wins"):
        print("Best bandit:", np.argmax(sampler.wins / np.maximum(sampler.wins + sampler.losses, 1e-8)))


if __name__ == "__main__":
    main()

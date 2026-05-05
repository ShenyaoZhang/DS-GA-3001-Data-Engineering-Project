"""
Hierarchical Emotion Classification Pipeline

Two-tier cascade (see tiered_labels.py for id maps):
  Tier 1: Binary (negative=0 / positive=1) — main_cluster_binary.py
  Tier 2:
    If negative → 3-class: sadness / anger / fear
    If positive → 2-class: joy / love (surprise is not a leaf; it is still
      "positive" in tier 1, then forced to joy or love in tier 2)

Three BERT heads are trained; at inference their outputs are composed into
dataset label ids 0–4 (and never 5 at tier 2).

Usage:
  python main_cluster_hierarchical.py train \
    -filename "data/processed/train_inner_emotions_emotion" \
    -val_path "data/processed/val_emotions_emotion.csv" \
    -hf_model_id "Qwen/Qwen2.5-7B-Instruct" \
    -few_shot_path "prompts/few_shot_examples_emotion.json" \
    -max_iterations 8

  python main_cluster_hierarchical.py eval \
    -val_path "data/processed/test_emotions_emotion.csv" \
    -binary_model "models/binary_fine_tunned_7_bandit_5" \
    -neg_model "models/neg_sub_fine_tunned_7_bandit_3" \
    -pos_model "models/pos_sub_fine_tunned_7_bandit_1"
"""

import argparse
import subprocess
import sys
import os
import re
import json
import gc

import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, classification_report

from preprocessing import TextPreprocessor
from fine_tune import BertFineTuner
from thompson_sampling import ThompsonSampler
from random_sampling import RandomSampler
from LDA import LDATopicModel
from labeling import Labeling
from tiered_labels import (
    EMOTION_TO_BINARY,
    EMOTION_NAMES,
    LEAF_DATASET_LABELS,
    NEG_SUB_TO_ORIG,
    ORIG_TO_NEG_SUB,
    ORIG_TO_POS_SUB,
    POS_SUB_TO_ORIG,
)

import nltk
nltk.download("punkt", quiet=True)


# ==================== SUB-CLASSIFIER LABELING ====================
class SubLabeling(Labeling):
    """Labeling subclass for 3-class sub-classifiers (neg_sub or pos_sub)."""

    def __init__(self, sub_type="neg_sub", **kwargs):
        self.sub_type = sub_type
        super().__init__(**kwargs)

    def _base_prompt(self, title: str) -> str:
        examples = self._build_examples_text()
        short_title = self._clean_for_prompt(title, max_chars=200)

        if self.sub_type == "neg_sub":
            prompt = (
                "Classify the negative emotion in this text.\n\n"
                "0 = sadness (grief, loneliness, heartbreak, feeling down, despair)\n"
                "1 = anger (frustration, fury, irritation, resentment, outrage)\n"
                "2 = fear (anxiety, terror, nervousness, dread, worry)\n\n"
                "Output only the number (0, 1, or 2).\n"
            )
        else:
            prompt = (
                "The text has positive sentiment. Classify as joy or love.\n\n"
                "0 = joy (happiness, delight, excitement, gratitude, contentment, cheerfulness)\n"
                "1 = love (affection, caring, tenderness, romance, feeling loved)\n\n"
                "Output only 0 or 1.\n"
            )
        if examples:
            prompt += "Examples:\n" + examples + "\n\n"
        prompt += f"Document: {short_title}\nLabel:"
        return prompt

    def _build_examples_text(self):
        if not self.few_shot_examples:
            return ""
        remap = ORIG_TO_NEG_SUB if self.sub_type == "neg_sub" else ORIG_TO_POS_SUB
        lines = []
        for ex in self.few_shot_examples:
            orig = int(ex["label"])
            if orig in remap:
                short_text = self._clean_for_prompt(ex["text"], max_chars=140)
                lines.append(f"Document: {short_text}")
                lines.append(f"Label: {remap[orig]}")
                lines.append("")
        return "\n".join(lines[:24]).strip()  # up to 8 examples (3 lines each)

    def _extract_label(self, response_text: str) -> str:
        response_text = str(response_text).strip().lower()
        if self.sub_type == "neg_sub":
            match = re.search(r"\b([012])\b", response_text)
        else:
            match = re.search(r"\b([01])\b", response_text)
        if match:
            return match.group(1)
        if self.sub_type == "neg_sub":
            kw = {"sadness": "0", "grief": "0", "anger": "1", "frustration": "1", "fear": "2", "anxiety": "2"}
        else:
            kw = {"joy": "0", "happy": "0", "delight": "0", "love": "1", "affection": "1", "romance": "1"}
        for k, v in kw.items():
            if k in response_text:
                return v
        return None

    def get_qwen_label(self, row):
        user_prompt = row["text"]
        if self.sub_type == "neg_sub":
            sys_msg = "You are an emotion classifier. The text expresses a negative emotion. Classify it as: 0=sadness, 1=anger, 2=fear. Output exactly one number."
        else:
            sys_msg = (
                "You are an emotion classifier. The text has positive sentiment. "
                "Classify it as: 0=joy, 1=love. Output exactly one number."
            )

        messages = [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt", truncation=True, max_length=768).to(self.model.device)
        input_len = model_inputs.input_ids.shape[1]
        with torch.inference_mode():
            outputs = self.model.generate(**model_inputs, max_new_tokens=3, do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
        response = self.tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)[0]
        return self._extract_label(response)


# ==================== TRAINING HELPERS ====================
def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).strip().lower()
    if v in {"true", "1", "yes", "y"}:
        return True
    if v in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {v}")


def reset_sampler_state():
    for f in ["selected_ids.txt", "wins.txt", "losses.txt", "minority_data_sub.csv"]:
        if os.path.exists(f):
            os.remove(f)


def train_sub_classifier(sub_type, args, filename, preprocessor):
    """Train tier-2 sub-classifier: neg_sub (3 classes) or pos_sub (2 classes: joy vs love)."""
    if sub_type == "neg_sub":
        num_labels = 3
        baseline = 1.0 / 3.0
        answer_pat = r"^[012]$"
    else:
        num_labels = 2
        baseline = 0.5
        answer_pat = r"^[01]$"

    remap = ORIG_TO_NEG_SUB if sub_type == "neg_sub" else ORIG_TO_POS_SUB
    valid_orig_labels = set(remap.keys())
    output_prefix = filename + f"_{sub_type}"
    metric = args.metric

    # Validation: filter + remap
    validation = pd.read_csv(args.val_path)
    validation = preprocessor.preprocess_df(validation)
    validation["training_text"] = validation["clean_title"] if "clean_title" in validation.columns else validation["title"]
    validation = validation[validation["label"].isin(valid_orig_labels)].copy()
    validation["label"] = validation["label"].map(remap)
    print(f"\n{'='*60}")
    print(f"Training {sub_type} sub-classifier")
    print(f"Validation: {validation['label'].value_counts().to_dict()}")

    # Load LDA data, filter + remap
    data = pd.read_csv(filename + "_lda.csv")
    data = data[data["label"].isin(valid_orig_labels)].copy()
    data["label"] = data["label"].map(remap)
    n_cluster = data["label_cluster"].value_counts().count()
    print(f"Training data: {len(data)} samples, {n_cluster} clusters")

    trainer = BertFineTuner(
        args.model_finetune, None, validation,
        confidence_threshold=args.confidence_threshold, num_labels=num_labels,
    )

    if args.sampling == "thompson":
        sampler = ThompsonSampler(n_cluster)
    else:
        sampler = RandomSampler(n_cluster)

    for i in range(args.max_iterations):
        sample_data, chosen_bandit = sampler.get_sample_data(data, args.sample_size, args.filter_label, trainer)

        if args.labeling != "file":
            labeler = SubLabeling(
                sub_type=sub_type, label_model=args.labeling, target_class=args.target_class,
                model_id=args.hf_model_id, few_shot_path=args.few_shot_path,
            )
            labeler.set_model()
            df = labeler.generate_inference_data(sample_data, "clean_title")
            print("df for inference created")
            df["answer"] = df.apply(lambda x: labeler.predict_animal_product(x), axis=1)
            df = df[df["answer"].notna()].copy()
            df["answer"] = df["answer"].astype(str).str.strip()
            df = df[df["answer"].str.match(answer_pat)].copy()
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
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            df = sample_data
            if "training_text" not in df.columns:
                df["training_text"] = df.get("clean_title", df.get("title", ""))

        # Confidence filter
        if trainer.trainer is not None and len(df) > 0:
            _, confs = trainer.get_inference_with_probs(df)
            mask = (confs >= args.confidence_threshold).numpy()
            if mask.sum() >= max(len(df) * 0.3, 10):
                df = df[mask].reset_index(drop=True)
                print(f"Confidence filter: kept {len(df)}/{mask.shape[0]}")

        print(df["label"].value_counts())

        # Accumulate
        if os.path.exists(f"{output_prefix}_training_data.csv"):
            prev = pd.read_csv(f"{output_prefix}_training_data.csv")
            if "training_text" not in prev.columns:
                prev["training_text"] = prev.get("clean_title", prev.get("title", ""))
            df = pd.concat([df, prev], ignore_index=True).drop_duplicates(subset=["id"])
            print(f"Accumulated: {len(df)} samples")

        model_name = trainer.get_base_model()
        model_results = trainer.get_last_model_acc()
        if model_results:
            baseline = model_results[model_name]

        lc = df["label"].value_counts()
        unbalanced = lc.max() / max(lc.min(), 1) >= 2

        results, _ = trainer.train_data(df, unbalanced)
        reward = results[f"eval_{metric}"] - baseline

        if reward > 0:
            print(f"Model improved by {reward}")
            model_name = f"models/{sub_type}_fine_tunned_{i}_bandit_{chosen_bandit}"
            trainer.update_model(model_name, results[f"eval_{metric}"], save_model=True)
            if os.path.exists(f"{output_prefix}_training_data.csv"):
                prev = pd.read_csv(f"{output_prefix}_training_data.csv")
                df = pd.concat([prev, df], ignore_index=True)
            df.to_csv(f"{output_prefix}_training_data.csv", index=False)
            if args.filter_label:
                trainer.set_clf(True)
        else:
            trainer.update_model(model_name, baseline, save_model=False)

        if args.sampling == "thompson":
            sampler.update(chosen_bandit if chosen_bandit != "fallback_random" else 0, reward)

    print(f"{sub_type} training complete.")


# ==================== HIERARCHICAL PREDICTOR ====================
class HierarchicalPredictor:
    """Chains binary + neg (3-way) + pos (2-way joy/love) BERT models."""

    def __init__(self, binary_model_path, neg_sub_model_path, pos_sub_model_path,
                 tokenizer_name="bert-base-uncased", max_length=128):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

        print("Loading binary model...")
        self.binary_model = BertForSequenceClassification.from_pretrained(binary_model_path, num_labels=2)
        self.binary_model.to(self.device).eval()

        print("Loading negative sub-classifier...")
        self.neg_model = BertForSequenceClassification.from_pretrained(neg_sub_model_path, num_labels=3)
        self.neg_model.to(self.device).eval()

        print("Loading positive sub-classifier (joy vs love)...")
        self.pos_model = BertForSequenceClassification.from_pretrained(pos_sub_model_path, num_labels=2)
        self.pos_model.to(self.device).eval()
        print("All models loaded.")

    def _batch_predict(self, model, texts, batch_size=64):
        all_preds, all_confs = [], []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                probs = torch.softmax(model(**inputs).logits, dim=1)
            all_preds.append(probs.argmax(dim=1).cpu())
            all_confs.append(probs.max(dim=1).values.cpu())
        return torch.cat(all_preds).numpy(), torch.cat(all_confs).numpy()

    def predict(self, df, batch_size=64):
        text_col = "training_text" if "training_text" in df.columns else ("clean_title" if "clean_title" in df.columns else "title")
        texts = df[text_col].fillna("").astype(str).tolist()

        binary_preds, binary_confs = self._batch_predict(self.binary_model, texts, batch_size)
        neg_idx = np.where(binary_preds == 0)[0]
        pos_idx = np.where(binary_preds == 1)[0]

        final_labels = np.zeros(len(texts), dtype=int)

        if len(neg_idx) > 0:
            neg_preds, _ = self._batch_predict(self.neg_model, [texts[i] for i in neg_idx], batch_size)
            for j, idx in enumerate(neg_idx):
                final_labels[idx] = NEG_SUB_TO_ORIG[neg_preds[j]]

        if len(pos_idx) > 0:
            pos_preds, _ = self._batch_predict(self.pos_model, [texts[i] for i in pos_idx], batch_size)
            for j, idx in enumerate(pos_idx):
                final_labels[idx] = POS_SUB_TO_ORIG[pos_preds[j]]

        result = df.copy()
        result["binary_pred"] = binary_preds
        result["binary_conf"] = binary_confs
        result["final_label"] = final_labels
        result["predicted_emotion"] = [EMOTION_NAMES[l] for l in final_labels]
        return result

    def evaluate(self, df, label_col="label"):
        result = self.predict(df)
        true = df[label_col].values
        pred = result["final_label"].values
        all_names = [EMOTION_NAMES[i] for i in sorted(EMOTION_NAMES.keys())]

        print("\n=== Hierarchical classification (full test set: 6 dataset labels) ===")
        print(classification_report(true, pred, labels=sorted(EMOTION_NAMES.keys()), target_names=all_names, zero_division=0))
        print(f"Accuracy:    {accuracy_score(true, pred):.4f}")
        print(f"F1 Macro:    {f1_score(true, pred, average='macro', zero_division=0):.4f}")
        print(f"F1 Weighted: {f1_score(true, pred, average='weighted', zero_division=0):.4f}")

        leaf_ids = sorted(LEAF_DATASET_LABELS)
        leaf_names = [EMOTION_NAMES[i] for i in leaf_ids]
        non_surprise = true != 5
        if non_surprise.any() and (true == 5).any():
            print("\n=== Same metrics excluding gold-label surprise (tier 2 never predicts surprise) ===")
            t2, p2 = true[non_surprise], pred[non_surprise]
            print(classification_report(t2, p2, labels=leaf_ids, target_names=leaf_names, zero_division=0))
            print(f"Accuracy:    {accuracy_score(t2, p2):.4f}")
            print(f"F1 Macro:    {f1_score(t2, p2, average='macro', zero_division=0):.4f}")

        binary_true = np.array([EMOTION_TO_BINARY[int(l)] for l in true])
        print(f"\nTier-1 binary accuracy: {accuracy_score(binary_true, result['binary_pred'].values):.4f}")
        return result


# ==================== CLI ====================
def main():
    parser = argparse.ArgumentParser(description="Hierarchical emotion classification")
    sub = parser.add_subparsers(dest="command")

    # --- train subcommand ---
    train_p = sub.add_parser("train", help="Train all 3 stages")
    train_p.add_argument("-sampling", type=str, default="thompson")
    train_p.add_argument("-sample_size", type=int, default=300)
    train_p.add_argument("-filter_label", type=str2bool, default=True)
    train_p.add_argument("-model_finetune", type=str, default="bert-base-uncased")
    train_p.add_argument("-labeling", type=str, default="qwen")
    train_p.add_argument("-filename", type=str, required=True)
    train_p.add_argument("-model", type=str, default="text")
    train_p.add_argument("-metric", type=str, default="f1_macro")
    train_p.add_argument("-val_path", type=str, required=True)
    train_p.add_argument("-cluster_size", type=int, default=10)
    train_p.add_argument("-target_class", type=str, default="joy")
    train_p.add_argument("-few_shot_path", type=str, default=None)
    train_p.add_argument("-hf_model_id", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    train_p.add_argument("-max_iterations", type=int, default=8)
    train_p.add_argument("-confidence_threshold", type=float, default=0.35)

    # --- eval subcommand ---
    eval_p = sub.add_parser("eval", help="Evaluate hierarchical model")
    eval_p.add_argument("-val_path", type=str, required=True)
    eval_p.add_argument("-binary_model", type=str, required=True)
    eval_p.add_argument("-neg_model", type=str, required=True)
    eval_p.add_argument("-pos_model", type=str, required=True)

    args = parser.parse_args()

    if args.command == "train":
        preprocessor = TextPreprocessor()

        # Stage 1: Binary
        print("\n" + "=" * 60)
        print("STAGE 1: Binary classification (positive vs negative)")
        print("=" * 60)
        # Run binary pipeline as subprocess
        script_dir = os.path.dirname(os.path.abspath(__file__))
        binary_script = os.path.join(script_dir, "main_cluster_binary.py")
        binary_cmd = [
            sys.executable, binary_script,
            "-sampling", args.sampling,
            "-sample_size", str(args.sample_size),
            "-filter_label", str(args.filter_label),
            "-model_finetune", args.model_finetune,
            "-labeling", args.labeling,
            "-filename", args.filename,
            "-model", args.model,
            "-metric", args.metric,
            "-val_path", args.val_path,
            "-cluster_size", str(args.cluster_size),
            "-target_class", args.target_class,
            "-hf_model_id", args.hf_model_id,
            "-max_iterations", str(args.max_iterations),
            "-confidence_threshold", str(args.confidence_threshold),
        ]
        if args.few_shot_path:
            binary_cmd += ["-few_shot_path", args.few_shot_path]
        subprocess.run(binary_cmd, check=True)

        # Stage 2a: Negative sub-classifier
        reset_sampler_state()
        train_sub_classifier("neg_sub", args, args.filename, preprocessor)

        # Stage 2b: Positive sub-classifier
        reset_sampler_state()
        train_sub_classifier("pos_sub", args, args.filename, preprocessor)

        print("\n" + "=" * 60)
        print("All 3 stages complete!")
        print("Use 'eval' command with the best model paths to evaluate.")
        print("=" * 60)

    elif args.command == "eval":
        preprocessor = TextPreprocessor()
        df = pd.read_csv(args.val_path)
        df = preprocessor.preprocess_df(df)
        df["training_text"] = df["clean_title"] if "clean_title" in df.columns else df["title"]

        predictor = HierarchicalPredictor(args.binary_model, args.neg_model, args.pos_model)
        predictor.evaluate(df)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()

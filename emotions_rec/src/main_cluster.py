import argparse
import pandas as pd
import numpy as np
from labeling import Labeling
from random_sampling import RandomSampler
from preprocessing import TextPreprocessor
from fine_tune import BertFineTuner
from thompson_sampling import ThompsonSampler
import nltk
import json
nltk.download("punkt")

import os
from LDA import LDATopicModel

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    v = str(v).strip().lower()
    if v in {"true", "1", "yes", "y"}:
        return True
    if v in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {v}")

def main():
    parser = argparse.ArgumentParser(prog="Sampling fine-tuning", description="Perform Sampling and fine tune")
    parser.add_argument("-sampling", type=str, required=False, help="Name of sampling method")
    parser.add_argument("-sample_size", type=int, required=False, help="sample size")
    parser.add_argument("-filter_label", type=str2bool, required=False, help="use model clf results to filter data")
    parser.add_argument("-balance", type=str2bool, required=False, help="balance positive and neg sample")
    parser.add_argument("-model_finetune", type=str, required=False, help="model base for fine tune")
    parser.add_argument("-labeling", type=str, required=False, help="Model to be used for labeling or file if label already on file")
    parser.add_argument("-baseline", type=float, required=False, help="The initial baseline metric")
    parser.add_argument("-filename", type=str, required=False, help="The initial file to be used")
    parser.add_argument("-model", type=str, required=False, help="The type of model to be finetune")
    parser.add_argument("-metric", type=str, required=False, help="The type of metric to be used for baseline")
    parser.add_argument("-val_path", type=str, required=False, help="path to validation")
    parser.add_argument("-cluster_size", type=int, required=False, help="number of clusters")
    parser.add_argument("-target_class", type=str, required=False, default="joy")
    parser.add_argument("-few_shot_path", type=str, required=False, default=None)
    parser.add_argument("-hf_model_id", type=str, required=False, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("-max_iterations", type=int, required=False, default=2)
    parser.add_argument("-confidence_threshold", type=float, required=False, default=0.85, help="Min confidence to keep pseudo-labels")
    parser.add_argument("-decision_threshold", type=float, required=False, default=0.45, help="Decision threshold for positive class")

    args = parser.parse_args()

    sampling = args.sampling
    sample_size = args.sample_size
    filter_label = args.filter_label
    balance = args.balance
    model_finetune = args.model_finetune
    labeling = args.labeling
    baseline = args.baseline
    filename = args.filename
    model = args.model
    metric = args.metric
    validation_path = args.val_path
    cluster_size = args.cluster_size
    target_class = args.target_class
    few_shot_path = args.few_shot_path
    hf_model_id = args.hf_model_id
    max_iterations = args.max_iterations
    confidence_threshold = args.confidence_threshold
    decision_threshold = args.decision_threshold

    preprocessor = TextPreprocessor()

    validation = pd.read_csv(validation_path)
    validation = preprocessor.preprocess_df(validation)
    validation["training_text"] = validation["clean_title"] if "clean_title" in validation.columns else validation["title"]

    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("log", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    try:
        data = pd.read_csv(filename + "_lda.csv")
        n_cluster = data["label_cluster"].value_counts().count()
        print("using data saved on disk")
    except Exception:
        print("Creating LDA")
        data = pd.read_csv(filename + ".csv")
        data = preprocessor.preprocess_df(data)
        lda_topic_model = LDATopicModel(num_topics=cluster_size)
        topics = lda_topic_model.fit_transform(data["clean_title"].to_list())
        data["label_cluster"] = topics
        n_cluster = data["label_cluster"].value_counts().count()
        print(n_cluster)
        data.to_csv(filename + "_lda.csv", index=False)
        print("LDA created")

    if model == "text":
        trainer = BertFineTuner(model_finetune, None, validation, confidence_threshold=confidence_threshold, decision_threshold=decision_threshold)
    else:
        raise ValueError("Currently only text model is supported")

    if sampling == "thompson":
        sampler = ThompsonSampler(n_cluster)
    elif sampling == "random":
        sampler = RandomSampler(n_cluster)
    else:
        raise ValueError("Choose one of thompson or random")

    for i in range(max_iterations):
        sample_data, chosen_bandit = sampler.get_sample_data(data, sample_size, filter_label, trainer)

        if labeling != "file":
            labeler = Labeling(
                label_model=labeling,
                target_class=target_class,
                model_id=hf_model_id,
                few_shot_path=few_shot_path
            )
            labeler.set_model()

            df = labeler.generate_inference_data(sample_data, "clean_title")
            print("df for inference created")
            df["answer"] = df.apply(lambda x: labeler.predict_animal_product(x), axis=1)
            df["answer"] = df["answer"].astype(str).str.strip()
            df["pseudo_label"] = np.where(df["answer"] == "1", 1, 0)
            df["label"] = df["pseudo_label"]
            df["training_text"] = df["title"]

            if os.path.exists(f"{filename}_data_labeled.csv"):
                train_data = pd.read_csv(f"{filename}_data_labeled.csv")
                train_data = pd.concat([train_data, df], ignore_index=True)
                train_data.to_csv(f"{filename}_data_labeled.csv", index=False)
            else:
                df.to_csv(f"{filename}_data_labeled.csv", index=False)

            try:
                del labeler.model
                del labeler.tokenizer
            except Exception:
                pass
            del labeler

            try:
                import gc
                import torch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        else:
            df = sample_data
            if "training_text" not in df.columns:
                df["training_text"] = df["clean_title"] if "clean_title" in df.columns else df["title"]

        # Filter pseudo-labels by model confidence (kicks in from iteration 2+)
        if trainer.trainer is not None and len(df) > 0:
            print("Filtering pseudo-labels by model confidence...")
            _, model_confs = trainer.get_inference_with_probs(df)
            confident_mask = (model_confs >= confidence_threshold).numpy()
            n_confident = confident_mask.sum()
            n_before = len(df)
            if n_confident >= max(n_before * 0.3, 10):
                df = df[confident_mask].reset_index(drop=True)
                print(f"Confidence filter: kept {len(df)}/{n_before} (threshold={confidence_threshold})")
            else:
                print(f"Confidence filter skipped: only {n_confident}/{n_before} samples passed (threshold={confidence_threshold}). Keeping all.")

        if "true_label" in df.columns and "pseudo_label" in df.columns:
            agreement = (df["true_label"].astype(float).fillna(-1).astype(int) == df["pseudo_label"].astype(int)).mean()
            print(f"Pseudo-label agreement with true_label on sampled batch: {agreement:.3f}")

        print(df["label"].value_counts())

        if os.path.exists("positive_data.csv"):
            pos = pd.read_csv("positive_data.csv")
            df = pd.concat([df, pos], ignore_index=True).sample(frac=1, random_state=42)
            print(f"adding positive data: {df['label'].value_counts()}")

        if balance:
            if len(df[df["label"] == 1]) > 0:
                unbalanced = len(df[df["label"] == 0]) / len(df[df["label"] == 1]) > 2
                if unbalanced:
                    label_counts = df["label"].value_counts()
                    min_count = min(label_counts)
                    balanced_df = pd.concat([
                        df[df["label"] == 0].sample(min_count * 2, random_state=42),
                        df[df["label"] == 1].sample(min_count, random_state=42)
                    ])
                    df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
                    print(f"Balanced data: {df.label.value_counts()}")

        model_name = trainer.get_base_model()
        model_results = trainer.get_last_model_acc()
        if model_results:
            baseline = model_results[model_name]
            print(f"previous model {metric} metric baseline of: {baseline}")
        else:
            print(f"Starting with metric {metric} baseline {baseline}")
        print("Starting training")

        try:
            still_unbalenced = len(df[df["label"] == 0]) / len(df[df["label"] == 1]) >= 2
        except Exception:
            still_unbalenced = True
        print(f"Unbalanced? {still_unbalenced}")

        results, huggingface_trainer = trainer.train_data(df, still_unbalenced)
        reward_difference = results[f"eval_{metric}"] - baseline

        if reward_difference > 0:
            print(f"Model improved with {reward_difference}")
            model_name = f"models/fine_tunned_{i}_bandit_{chosen_bandit}"
            trainer.update_model(model_name, results[f"eval_{metric}"], save_model=True)
            if os.path.exists(f"{filename}_training_data.csv"):
                train_data = pd.read_csv(f"{filename}_training_data.csv")
                df = pd.concat([train_data, df], ignore_index=True)
            df.to_csv(f"{filename}_training_data.csv", index=False)
            if os.path.exists("positive_data.csv"):
                os.remove("positive_data.csv")
            if filter_label:
                trainer.set_clf(True)
        else:
            trainer.update_model(model_name, baseline, save_model=False)
            if os.path.exists("positive_data.csv"):
                positive = pd.read_csv("positive_data.csv")
                df = df[df["label"] == 1]
                df = pd.concat([df, positive], ignore_index=True).drop_duplicates()
            df[df["label"] == 1].to_csv("positive_data.csv", index=False)

        if os.path.exists(f"{filename}_model_results.json"):
            with open(f"{filename}_model_results.json", "r") as file:
                existing_results = json.load(file)
        else:
            existing_results = {}

        if existing_results.get(str(chosen_bandit)):
            existing_results[str(chosen_bandit)].append(results)
        else:
            existing_results[str(chosen_bandit)] = [results]

        with open(f"{filename}_model_results.json", "w") as file:
            json.dump(existing_results, file, indent=4)

        if sampling == "thompson":
            sampler.update(chosen_bandit if chosen_bandit != "fallback_random" else 0, reward_difference)

    if hasattr(sampler, "wins") and hasattr(sampler, "losses"):
        print("Bandit with highest expected improvement:", np.argmax(sampler.wins / np.maximum(sampler.wins + sampler.losses, 1e-8)))
        print(sampler.wins)
        print(sampler.losses)

if __name__ == "__main__":
    main()

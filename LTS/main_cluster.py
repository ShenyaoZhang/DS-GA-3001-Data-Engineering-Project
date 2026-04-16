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
import gc
import torch
nltk.download('punkt')

import os
from LDA import LDATopicModel

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    if str(v).lower() in ("yes", "true", "t", "1", "y"):
        return True
    if str(v).lower() in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

def main():
    # CLI entrypoint for the full LTS loop:
    # cluster -> sample -> pseudo-label -> fine-tune -> reward sampler.
    parser = argparse.ArgumentParser(prog="Sampling fine-tuning", description='Perform Sampling and fine tune')
    # parser.add_argument('-cluster', type=str, required=False,
    #                     help="Name of cluster type")
    
    # -sampling: whether to use thompson sampling or random sampling
    parser.add_argument('-sampling', type=str, required=False,
                        help="Name of sampling method")

    # sample size: how many examples to sample in each round
    parser.add_argument('-sample_size', type=int, required=False,
                        help="sample size")

    # filter_label: whether to use the current classifier's predictions 
    # to filter candidate data before sampling               
    # parser.add_argument('-filter_label', type=bool, required=False,
    #                     help="use model clf results to filter data")
    parser.add_argument('-filter_label', type=str2bool, nargs='?', const=True, default=False, required=False,
                        help="use model clf results to filter data")

    # balance: whether to rebalance the sampled labe
    # parser.add_argument('-balance', type=bool, required=False,
    #                     help="balance positive and neg sample")
    parser.add_argument('-balance', type=str2bool, nargs='?', const=True, default=False, required=False,
                        help="balance positive and neg sample")
                        
    # model_finetune: the base model to finetune
    parser.add_argument('-model_finetune', type=str, required=False,
                        help="model base for fine tune")
    
    # labeling: the model to be used for labeling
    parser.add_argument('-labeling', type=str, required=False,
                        help="Model to be used for labeling or file if label already on file")
    
    # baseline: the initial baseline metric
    parser.add_argument('-baseline', type=float, required=False,
                        help="The initial baseline metric")
    
    # filename: the name of the dataset to be used
    parser.add_argument('-filename', type=str, required=False,
                        help="The initial file to be used")
    
    
    parser.add_argument('-model', type=str, required=False,
                        help="The type of model to be finetune")
    
    # metric: the metric to be used for baseline
    parser.add_argument('-metric', type=str, required=False,
                        help="The type of metric to be used for baseline")
    
    # validation_path: the path to the validation set
    parser.add_argument('-val_path', type=str, required=False,
                        help="path to validation")
    
    # cluster_size: the number of clusters to create
    # parser.add_argument('-cluster_size', type=str, required=False,
    #                     help="path to validation")
    parser.add_argument('-cluster_size', type=int, required=False,
                        help="number of LDA topics")


    # Parse runtime configuration.
    args = parser.parse_args()

    # cluster = args.cluster
    sampling = args.sampling
    sample_size = args.sample_size
    # filter_label = args.filter_label  # Original
    # balance = args.balance  # Original
    filter_label = bool(args.filter_label)
    balance = bool(args.balance)
    model_finetune = args.model_finetune
    labeling = args.labeling
    baseline = args.baseline
    filename = args.filename
    model = args.model
    metric = args.metric
    validation_path = args.val_path
    cluster_size = args.cluster_size


    # Text cleaner used before topic clustering.
    preprocessor = TextPreprocessor()


    # Gold validation set used to score each iteration model.
    validation = pd.read_csv(validation_path)
    validation["training_text"] = validation["title"]

    # Output artifacts written by the training loop.
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("log", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    try:
        # Reuse precomputed clusters when available to avoid reclustering.
        data = pd.read_csv(filename+"_lda.csv")
        n_cluster = data['label_cluster'].value_counts().count()
        print("using data saved on disk")
        # print(sample.head(1))
    except Exception:
        # First run: load raw file, preprocess text, and assign LDA cluster IDs.
        print("Creating LDA")
        data = pd.read_csv(filename+".csv")
        data = preprocessor.preprocess_df(data)
        lda_topic_model = LDATopicModel(num_topics=cluster_size)
        topics = lda_topic_model.fit_transform(data['clean_title'].to_list())
        data["label_cluster"] = topics
        n_cluster = data['label_cluster'].value_counts().count()
        print(n_cluster)
        data.to_csv(filename + "_lda.csv", index=False)
        print("LDA created")


    baseline = baseline

    # Initialize downstream learner (currently text-only in this repo).
    if model == "text":
        trainer = BertFineTuner(model_finetune, None, validation)
    else:
        raise ValueError("Currently only text model is supported")

    # Choose sample selection strategy.
    if sampling == "thompson":
        ## thompson sampler
        sampler = ThompsonSampler(n_cluster)
    elif sampling == "random":
        sampler = RandomSampler(n_cluster)
    else:
        raise ValueError("Choose one of thompson or random")


    # Main active-learning loop: each round adds labels and attempts model improvement.
    for i in range(10):
        # 1) Draw a batch from one/all clusters depending on sampler strategy.
        sample_data, chosen_bandit = sampler.get_sample_data(data, sample_size, filter_label, trainer)
        ## Generate labels
        if labeling != "file":
            # 2) Build prompts and pseudo-label sampled records with the selected label model.
            labeler = Labeling(label_model=labeling)
            labeler.set_model()
            df = labeler.generate_inference_data(sample_data, 'clean_title')
            print("df for inference created")
            df["answer"] = df.apply(lambda x: labeler.predict_animal_product(x), axis=1)
            # df["answer"] = df["answer"].str.strip()  # Original
            # df["label"] = np.where(df["answer"] == 'relevant animal', 1, 0)  # Original: wildlife labels
            df["answer"] = df["answer"].str.strip().str.lower()
            df["label"] = np.where(df["answer"] == 'earn', 1, 0)
            if labeling == "qwen":
                del labeler.model
                del labeler.tokenizer
                del labeler
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            # Persist all pseudo-labeled records across iterations.
            if os.path.exists(f"{filename}_data_labeled.csv"):
                train_data = pd.read_csv(f"{filename}_data_labeled.csv")
                train_data = pd.concat([train_data, df])
                train_data.to_csv(f"{filename}_data_labeled.csv", index=False)
            else:
                df.to_csv(f"{filename}_data_labeled.csv", index=False)
        else:
            # Labels already exist in the sampled file.
            df = sample_data
            # Reuters: map topic-specific label column to generic 'label' if needed.
            if "label" not in df.columns and "label_earn" in df.columns:
                df["label"] = df["label_earn"]
        print(df["label"].value_counts())
        # print(df["answer"].value_counts())

        # ADD POSITIVE DATA IF AVAILABLE

        # Carry forward positives from previous failed rounds to reduce rare-class starvation.
        if os.path.exists('positive_data.csv'):
            pos = pd.read_csv('positive_data.csv')
            df = pd.concat([df, pos]).sample(frac=1)
            print(f"adding positive data: {df['label'].value_counts()}")
        if balance:
            # Optional light undersampling: cap majority class to 2x minority class.
            if len(df[df["label"]==1]) > 0:
                unbalanced = len(df[df["label"]==0]) / len(df[df["label"]==1]) > 2
                if unbalanced:
                    label_counts = df["label"].value_counts()
                    # Determine the number of samples to keep for each label
                    min_count = min(label_counts)
                    balanced_df = pd.concat([
                        df[df["label"] == 0].sample(min_count*2),
                        df[df["label"] == 1].sample(min_count)
                    ])

                    # Shuffle the rows
                    df = balanced_df.sample(frac=1).reset_index(drop=True)
                    print(f"Balanced data: {df.label.value_counts()}")
            # else:
                # if i == 0: # if this is the first model training
                # unbalanced = True
                # print("No positive samples to balance with.")
        ## FINE TUNE MODEL

        # 3) Train candidate model and compare against current baseline metric.
        # previous model
        model_name = trainer.get_base_model()
        print(f"using model {model_name}")
        model_results = trainer.get_last_model_acc()
        if model_results:
            # After first successful iteration, baseline becomes the current best model score.
            baseline = model_results[model_name]
            print(f"previous model {metric} metric baseline of: {baseline}")
        else:
            print(f"Starting with metric {metric} baseline {baseline}")
        print(f"Starting training")

        try:
            still_unbalenced = len(df[df["label"]==0]) / len(df[df["label"]==1])  >= 2
        except Exception:
            still_unbalenced = True
        print(f"Unbalanced? {still_unbalenced}")

        results, huggingface_trainer = trainer.train_data(df, still_unbalenced)
        reward_difference = results[f"eval_{metric}"] - baseline
        if reward_difference > 0:
            # 4a) Improvement: keep model, persist training data, enable model-based filtering.
            print(f"Model improved with {reward_difference}")
            model_name = f"models/fine_tunned_{i}_bandit_{chosen_bandit}"
            trainer.update_model(model_name, results[f"eval_{metric}"], save_model=True)
            # df.to_csv("llama_training_data.csv", index=False)
            if os.path.exists(f'{filename}_training_data.csv'):
                train_data = pd.read_csv(f'{filename}_training_data.csv')
                df = pd.concat([train_data, df])
            df.to_csv(f'{filename}_training_data.csv', index=False)
            if os.path.exists('positive_data.csv'):
                os.remove('positive_data.csv')
            if filter_label:
                trainer.set_clf(True)
                # data["predicted_label"] = trainer.get_inference(data)
                # print(data["predicted_label"].value_counts())
                # if data[data["predicted_label"]==1].empty:
                #     data["predicted_label"] = 1
                # data.to_csv("data_w_predictions.csv", index=False)
            ## save model results
        else:
            # 4b) No improvement: roll back and retain only positives for a future retry.
            # back to initial model
            trainer.update_model(model_name, baseline, save_model=False)
            # save positive sample
            if os.path.exists('positive_data.csv'):
                positive = pd.read_csv("positive_data.csv")
                df = df[df["label"]==1]
                df = pd.concat([df, positive])
                df = df.drop_duplicates()
            df[df["label"]==1].to_csv("positive_data.csv", index=False)


        # Log per-cluster/per-round metrics for analysis and reproducibility.
        if os.path.exists(f'{filename}_model_results.json'):
            with open(f'{filename}_model_results.json', 'r') as file:
                existing_results = json.load(file)
        else:
            existing_results = {}

        if existing_results.get(str(chosen_bandit)):
            existing_results[str(chosen_bandit)].append(results)
        else:
            existing_results[str(chosen_bandit)] = [results]

        # Write the updated list to the file
        with open(f'{filename}_model_results.json', 'w') as file:
            json.dump(existing_results, file, indent=4)
        if sampling == "thompson":
            # Thompson sampler gets binary-like reward signal from metric delta.
            sampler.update(chosen_bandit, reward_difference)


    # Final sampler diagnostics: identify the cluster with best empirical reward.
    # print("Bendt with highest expected improvement:", np.argmax(sampler.wins / (sampler.wins + sampler.losses)))
    # print(sampler.wins)
    # print(sampler.losses)
    if sampling == "thompson":
        print("Bendt with highest expected improvement:", np.argmax(sampler.wins / (sampler.wins + sampler.losses)))
        print(sampler.wins)
        print(sampler.losses)
    else:
        print("Random sampling completed. No bandit win/loss stats available.")
    # Save the DataFrame with cluster labels
    # umap_df.to_csv("./data/gpt_training_with_clusters.csv", index=False)




if __name__ == "__main__":
    main()
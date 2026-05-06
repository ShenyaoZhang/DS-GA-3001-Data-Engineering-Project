from typing import Any

import numpy as np
import pandas as pd


class RandomSampler:
    def __init__(self, n_bandits, run_id="", uniform_pool=False, seed=42):
        self.n_bandits = n_bandits
        self.run_id = (run_id or "").strip()
        self._pfx = f"{self.run_id}_" if self.run_id else ""
        self.uniform_pool = uniform_pool
        self.seed = seed
        try:
            loaded = np.loadtxt(f"{self._pfx}selected_ids.txt", dtype=str)
            self.selected_ids = set(np.atleast_1d(loaded).tolist())
        except Exception:
            self.selected_ids = set()

    def get_sample_data(self, df, sample_size, filter_label: bool, trainer: Any):
        def get_sample(data, size):
            if data.empty:
                return pd.DataFrame()
            return data.sample(min(size, len(data)), random_state=self.seed)

        df = df.copy()
        df["id"] = df["id"].astype(str)
        df = df[~df["id"].isin(self.selected_ids)]

        if df.empty:
            raise ValueError("No unlabeled data left to sample from.")

        if self.uniform_pool:
            sampled = df.sample(min(sample_size, len(df)), random_state=self.seed)
            self.selected_ids.update(sampled["id"].astype(str).tolist())
            with open(f"{self._pfx}selected_ids.txt", "w") as f:
                f.write("\n".join(map(str, sorted(self.selected_ids))))
            return sampled, "uniform_pool"

        unique_clusters = sorted(df["label_cluster"].unique().tolist())
        samples_per_cluster = max(1, int(sample_size / max(1, len(unique_clusters))))
        sampled_data = []

        if filter_label and trainer.get_clf():
            df["predicted_label"] = trainer.get_inference(df)

        for cluster in unique_clusters:
            cluster_data = df[df["label_cluster"] == cluster]
            if cluster_data.empty:
                continue

            if filter_label and "predicted_label" in cluster_data.columns:
                pos = cluster_data[cluster_data["predicted_label"] == 1]
                neg = cluster_data[cluster_data["predicted_label"] == 0]
                n_pos = samples_per_cluster // 2
                pos_sample = get_sample(pos, n_pos)
                neg_sample = get_sample(neg, samples_per_cluster - len(pos_sample))
                chunk = pd.concat([pos_sample, neg_sample], ignore_index=True)
                if chunk.empty:
                    chunk = get_sample(cluster_data, samples_per_cluster)
            else:
                chunk = get_sample(cluster_data, samples_per_cluster)

            if not chunk.empty:
                sampled_data.append(chunk)

        if not sampled_data:
            sampled = get_sample(df, sample_size)
        else:
            sampled = pd.concat(sampled_data, ignore_index=True)
            if len(sampled) > sample_size:
                sampled = sampled.sample(sample_size, random_state=self.seed).reset_index(drop=True)

        self.selected_ids.update(sampled["id"].astype(str).tolist())
        with open(f"{self._pfx}selected_ids.txt", "w") as f:
            f.write("\n".join(map(str, self.selected_ids)))

        return sampled, "random"

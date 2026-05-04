from typing import Any
import numpy as np
from scipy.stats import beta
import pandas as pd


class ThompsonSampler:
    def __init__(self, n_bandits, alpha=0.5, beta=0.5, decay=0.99):
        self.n_bandits = n_bandits
        self.wins = np.zeros(n_bandits)
        self.losses = np.zeros(n_bandits)
        self.alpha = alpha
        self.beta = beta
        self.decay = decay

        try:
            loaded_ids = np.loadtxt("selected_ids.txt", dtype=str)
            if np.isscalar(loaded_ids):
                loaded_ids = np.array([loaded_ids])
            self.selected_ids = set(loaded_ids.tolist())
        except Exception:
            self.selected_ids = set()

        try:
            self.wins = np.loadtxt("wins.txt")
            self.losses = np.loadtxt("losses.txt")
            if np.isscalar(self.wins):
                self.wins = np.array([self.wins])
            if np.isscalar(self.losses):
                self.losses = np.array([self.losses])
        except Exception:
            self.wins = np.zeros(n_bandits)
            self.losses = np.zeros(n_bandits)

    def choose_bandit(self, exclude_bandits=None):
        if exclude_bandits is None:
            exclude_bandits = set()

        betas = beta(self.wins + self.alpha, self.losses + self.beta)
        sampled_rewards = betas.rvs(size=self.n_bandits)

        for b in exclude_bandits:
            if 0 <= b < self.n_bandits:
                sampled_rewards[b] = -1

        return int(np.argmax(sampled_rewards))

    def update(self, chosen_bandit, reward_difference):
        if isinstance(chosen_bandit, str):
            return

        if reward_difference > 0:
            self.wins[chosen_bandit] += 1
        else:
            self.losses[chosen_bandit] += 1

        self.wins *= self.decay
        self.losses *= self.decay

        np.savetxt("wins.txt", self.wins)
        np.savetxt("losses.txt", self.losses)

    def get_sample_data(self, df, sample_size, filter_label: bool, trainer: Any):
        def sample_from_df(df_in, n):
            if df_in.empty:
                return pd.DataFrame()
            return df_in.sample(min(n, len(df_in)), random_state=42)

        df = df[~df["id"].isin(self.selected_ids)].copy()
        if df.empty:
            raise ValueError("No unlabeled data remaining to sample from.")

        data = pd.DataFrame()
        chosen_bandit = None

        # --- Filtered mode: try a few distinct clusters first ---
        if filter_label and trainer.get_clf():
            tried_bandits = set()
            max_filtered_attempts = min(3, self.n_bandits)

            for _ in range(max_filtered_attempts):
                chosen_bandit = self.choose_bandit(exclude_bandits=tried_bandits)
                tried_bandits.add(chosen_bandit)

                print(f"Chosen bandit {chosen_bandit}")
                bandit_df = df[df["label_cluster"] == chosen_bandit].copy()
                print(f"length of bandit {len(bandit_df)}")

                if bandit_df.empty:
                    continue

                preds, confs = trainer.get_inference_with_probs(bandit_df)
                bandit_df["predicted_label"] = preds.numpy()
                bandit_df["confidence"] = confs.numpy()
                print("inference results")
                print(bandit_df["predicted_label"].value_counts())
                print(f"Mean confidence: {bandit_df['confidence'].mean():.3f}")

                # Only use high-confidence predictions for sampling
                confident = bandit_df[bandit_df["confidence"] >= trainer.confidence_threshold]
                pos = confident[confident["predicted_label"] == 1]
                neg = confident[confident["predicted_label"] == 0]

                if not pos.empty:
                    n_pos = sample_size // 2
                    pos_data = sample_from_df(pos, n_pos)
                    neg_data = sample_from_df(neg, sample_size - len(pos_data))
                    data = pd.concat([pos_data, neg_data]).sample(frac=1, random_state=42)
                    break
                else:
                    print("no confident positive predictions in chosen bandit, trying another")

            # Fallback: if filtered mode failed, sample unfiltered from the best remaining bandit
            if data.empty:
                print("No predicted positives found after trying multiple bandits. Falling back to unfiltered Thompson sampling.")
                fallback_bandit = self.choose_bandit()
                chosen_bandit = fallback_bandit
                bandit_df = df[df["label_cluster"] == chosen_bandit].copy()
                print(f"Fallback bandit {chosen_bandit}")
                print(f"length of fallback bandit {len(bandit_df)}")

                if not bandit_df.empty:
                    data = sample_from_df(bandit_df, sample_size)

        # --- Standard unfiltered mode ---
        else:
            chosen_bandit = self.choose_bandit()
            print(f"Chosen bandit {chosen_bandit}")

            bandit_df = df[df["label_cluster"] == chosen_bandit].copy()
            print(f"length of bandit {len(bandit_df)}")

            if not bandit_df.empty:
                data = sample_from_df(bandit_df, sample_size)

        # --- Final fallback if chosen cluster is empty or sampling still failed ---
        if data.empty:
            print("Falling back to random sampling from remaining data.")
            data = df.sample(min(sample_size, len(df)), random_state=42)
            chosen_bandit = "fallback_random"

        self.selected_ids.update(data["id"].astype(str).tolist())
        with open("selected_ids.txt", "w") as f:
            f.write("\n".join(sorted(self.selected_ids)))

        return data, chosen_bandit

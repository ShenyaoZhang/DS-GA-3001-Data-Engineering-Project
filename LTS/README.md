# 🐾 Wildlife Trafficking Detection: Using LLMs to Automate the Creation of Classifiers for Data Triage

This repository contains the code used in the paper:

> **"A Cost-Effective LLM-based Approach to Identify Wildlife Trafficking in Online Marketplaces"**
> Proceedings of the ACM on Management of Data, Volume 3, Issue 3. Article No.: 119, Pages 1 - 23.  
> https://dl.acm.org/doi/10.1145/3725256
> https://arxiv.org/html/2504.21211v1

---

## 📚 Table of Contents

1. [Requirements](#-requirements)
2. [Setup](#-setup)
3. [Use Cases & Reproducibility](#-use-cases--reproduction)

---

## 📦 Requirements

Experiments were conducted using **Python 3.11.2**. All required dependencies are listed in `requirements.txt` and can be installed via pip.

Before running the experiments, you need to set your OpenAI key on labeling.py (line 116)

---

## ⚙️ Setup

### 1. Create a Virtual Environment (Recommended)

Use a virtual environment to avoid dependency conflicts:

```bash
python -m venv venv
source venv/bin/activate  # For Unix/macOS
venv\Scripts\activate     # For Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

🧪 Use Cases & Reproducibility
To reproduce experiments from the paper, run main_cluster.py with the appropriate flags:

The data needed to run all experiments can be found on:
https://drive.google.com/drive/folders/1UO4OYjBmvgKcFz71YeB1kefXqQhMvXGA?usp=sharing

🔧 Required Parameters:

- -sample_size: Number of samples per iteration.

- -filename: Path to the dataset CSV file. Must contain a text column named 'title'.

- -val_path: Path to the validation dataset.

- -balance: Whether to balance the dataset via undersampling (bool).

- -sampling: Sampling strategy (string: "thompson" or "random").

- -filter_label: Whether to filter out negative samples. (bool)

- -model_finetune: Model name for fine-tuning in the first iteration (string: e.g., "bert-base-uncased").

- -labeling: Source of labels (string: gpt, llama, or file).

- -model: Choose model type (string: text, multi-modal).

- -metric: Evaluation metric used to compare models between iterations (string: "f1", "accuracy", "recall", "precision").

- -baseline: Initial baseline metric score for first iteration.

- -cluster_size: Number of clusters to use.



👜 Use Case 1: Leather Products
```bash
python main_cluster.py \
  -sample_size 200 \
  -filename "data_use_cases/data_leather" \
  -val_path "data_use_cases/leather_validation.csv" \
  -balance False \
  -sampling "thompson" \
  -filter_label True \
  -model_finetune "bert-base-uncased" \
  -labeling "gpt" \
  -model "text" \
  -baseline 0.5 \
  -metric "f1" \
  -cluster_size 10
```

🦈 Use Case 2: Shark Products
```bash
python main_cluster.py \
  -sample_size 200 \
  -filename "data_use_cases/shark_trophy" \
  -val_path "data_use_cases/validation_sharks.csv" \
  -balance True \
  -sampling "thompson" \
  -filter_label True \
  -model_finetune "bert-base-uncased" \
  -labeling "gpt" \
  -model "text" \
  -baseline 0.5 \
  -metric "f1" \
  -cluster_size 5
```

Use Case 3: Animal Products
```bash
python main_cluster.py \
  -sample_size 200 \
  -filename "data_use_cases/animals" \
  -val_path "data_use_cases/validation_animals.csv" \
  -balance True \
  -sampling "thompson" \
  -filter_label True \
  -model_finetune "bert-base-uncased" \
  -labeling "gpt" \
  -model "text" \
  -baseline 0.5 \
  -metric "f1" \
  -cluster_size 10
```

---

## 📰 Use Case 4: Reuters-21578 (archive.zip) + Qwen labeler

This use case runs LTS on the Reuters-21578 collection shipped as
`data/archive.zip` (8 CSVs covering ModApte / ModHayes / ModLewis splits, 13
columns each: `text, text_type, topics, lewis_split, cgis_split, old_id,
new_id, places, people, orgs, exchanges, date, title`). The default binary
triage task is **`earn` vs `not earn`**, but every part of the pipeline now
accepts an arbitrary positive/negative label so you can repeat the experiment
for any topic in the corpus (`trade`, `acq`, `crude`, `money-fx`, ...).

### Step 1 — Prepare the data

```bash
# auto-extracts data/archive.zip into data/archive/ on first run
python scripts/prepare_reuters.py --split ModApte --label earn
```

Useful options:

```bash
# inspect available topics (descending frequency) before picking one
python scripts/prepare_reuters.py --split ModApte --list-topics 20

# different positive class
python scripts/prepare_reuters.py --split ModApte --label trade

# different ModX split
python scripts/prepare_reuters.py --split ModLewis --label acq
```

Outputs (under `data_use_cases/`):

- `reuters_modapte_earn_train.csv` — pool (`id, title, description, label, label_earn`)
- `reuters_modapte_earn_validation.csv` — gold validation (`id, title, description, label`)
- `reuters_modapte_earn_smoke_train.csv` / `_smoke_validation.csv` — small subsets for quick runs

### Step 2 — Smoke-test the Qwen labeler (optional, ~30s on CPU)

```bash
# downloads Qwen2.5-0.5B-Instruct on first run
python scripts/smoke_test_qwen_label.py
```

Set `QWEN_MODEL_DIR=Qwen/Qwen2.5-1.5B-Instruct` (or any local path) for higher
accuracy.

### Step 3 — Run LTS with Qwen as labeler

```bash
# point at a local Qwen checkpoint or HF id
export QWEN_MODEL_DIR="Qwen/Qwen2.5-1.5B-Instruct"

# fast smoke run (≈500-row pool, 200-row validation, 10 rounds)
python main_cluster.py \
  -sample_size 50 \
  -filename "data_use_cases/reuters_modapte_earn_smoke_train" \
  -val_path "data_use_cases/reuters_modapte_earn_smoke_validation.csv" \
  -balance True \
  -sampling "thompson" \
  -filter_label False \
  -model_finetune "bert-base-uncased" \
  -labeling "qwen" \
  -model "text" \
  -baseline 0.5 \
  -metric "f1" \
  -cluster_size 5
```

For a different positive class (e.g. `trade`), pass it explicitly so the prompt
and the {0,1} mapping match what `prepare_reuters.py` produced:

```bash
python main_cluster.py \
  -sample_size 200 \
  -filename "data_use_cases/reuters_modapte_trade_train" \
  -val_path "data_use_cases/reuters_modapte_trade_validation.csv" \
  -balance True -sampling "thompson" -filter_label False \
  -model_finetune "bert-base-uncased" -labeling "qwen" \
  -model "text" -baseline 0.5 -metric "f1" -cluster_size 10 \
  -positive_label "trade" -negative_label "not trade" \
  -task_description "the title of a Reuters news article. Label 1 ({pos}) if the article is primarily about international trade, tariffs, exports, imports, or trade policy. Label 2 ({neg}) for any other topic."
```

You can equivalently set those three knobs via env vars
(`LTS_POSITIVE_LABEL`, `LTS_NEGATIVE_LABEL`, `LTS_TASK_DESCRIPTION`) and
optionally point `LTS_EXAMPLES_JSON` at a JSON file of `{"title", "label"}`
in-context examples.

### Notes / tips

- Before re-running on a new pool, delete `positive_data.csv` (carry-over from a
  previous failed iteration) and the `models/` / `log/` / `results/` outputs to
  start clean.
- `main_cluster.py` caches LDA clusters in `<filename>_lda.csv`; delete it to
  re-cluster after changing `--label` or split.
- Use `-labeling file` instead of `-labeling qwen` to run an oracle baseline
  with the gold `label` column — useful for measuring how much accuracy LTS
  loses by relying on Qwen pseudo-labels.
- Qwen2.5-0.5B is fine for a smoke run; for paper-quality results use
  Qwen2.5-1.5B-Instruct or larger.

---

## 🧮 Use Case 5: Reuters multi-class triage (Reuters-21578 + Qwen)

The pipeline also supports **multi-class triage**: instead of one binary task,
train a single classifier that maps each article to one of N Reuters topics
(plus an optional `other` catch-all). Binary mode (above) is unchanged and
still the default; multi-class is opt-in via `--task-type multiclass` and
`-task_type multiclass`.

Reuters articles can be tagged with multiple topics, so the prep script picks
the **first matching label in your priority order**. Articles with no matching
topic are placed in `--other-label other` (or dropped via `--no-other`).

### Step 1 — Prepare a multi-class pool

```bash
# default top-5 + 'other'
python scripts/prepare_reuters.py \
  --split ModApte \
  --task-type multiclass \
  --labels "earn,acq,trade,crude,money-fx" \
  --other-label "other"
```

Outputs (under `data_use_cases/`):

- `reuters_modapte_multiclass_6cls_train.csv` — pool (`id, title, description, label, label_text`)
- `reuters_modapte_multiclass_6cls_validation.csv` — gold validation
- `reuters_modapte_multiclass_6cls_smoke_train.csv` / `_smoke_validation.csv`

The script prints the resolved class list and a one-liner you can copy into the
`main_cluster.py` invocation:

```text
[prepare_reuters] use this with main_cluster.py:
    -task_type "multiclass" -class_labels "earn,acq,trade,crude,money-fx,other"
```

### Step 2 — Run LTS in multi-class mode

```bash
export QWEN_MODEL_DIR="Qwen/Qwen2.5-1.5B-Instruct"

python main_cluster.py \
  -sample_size 200 \
  -filename "data_use_cases/reuters_modapte_multiclass_6cls_train" \
  -val_path "data_use_cases/reuters_modapte_multiclass_6cls_validation.csv" \
  -balance False \
  -sampling "thompson" \
  -filter_label False \
  -model_finetune "bert-base-uncased" \
  -labeling "qwen" \
  -model "text" \
  -baseline 0.5 \
  -metric "f1" \
  -cluster_size 10 \
  -task_type "multiclass" \
  -class_labels "earn,acq,trade,crude,money-fx,other" \
  -task_description "the title (and possibly a short excerpt) of a Reuters news article. Pick the single best matching topic from the list of labels."
```

### What changes vs binary mode

- **BERT** is configured with `num_labels = len(class_labels)` (here 6) instead
  of 2. Saved checkpoints and `update_model` reload at the same width.
- The Qwen prompt asks for **one** label from the list; the parser
  case-insensitively matches the longest label that appears in the model's
  output, and falls back to the last class (typically `other`) on garbage.
- **Metrics** include both **`eval_f1` (weighted)** and **`eval_f1_macro`**.
  Weighted F1 is what the bandit reward uses (matches binary mode); macro F1 is
  the better number to report when classes are imbalanced.
- **Balance / positive-carry-over** logic from binary mode is **disabled**: it
  hard-codes `label == 1` vs `label == 0` semantics. `-balance True` is silently
  ignored, and `positive_data.csv` is not written.
- The weighted-loss `MyTrainer` path is **not used** in multi-class (the
  built-in HF `Trainer` with uniform cross-entropy is used instead).

### Coexisting with your binary results

Multi-class outputs use a **different filename stem**
(`reuters_modapte_multiclass_*`), so existing binary artifacts
(`reuters_modapte_earn_*` CSVs, `*_model_results.json`, saved BERT checkpoints,
`positive_data.csv`) are **not overwritten**. You can keep both sets side by
side and compare them in your write-up.

---

📫 Contact
For questions or feedback, please open an issue or reach out via the contact information provided in the paper.


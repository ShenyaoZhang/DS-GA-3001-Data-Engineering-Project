# Emotions `joy` Reproducibility Guide

This folder contains the reproducible code, notebook, prompts, and run configuration for the Emotions `joy` experiment in the DS-GA 3001 Data Engineering final project.

## Experiment Goal

This experiment evaluates whether the Lean-to-Sample (LTS) framework can improve minority-class discovery on an imbalanced text classification task.

- **Dataset:** `dair-ai/emotion` (HuggingFace)
- **Positive class:** `joy`
- **Negative class:** all other emotions (sadness, love, anger, fear, surprise)
- **Sampling strategies compared:** random sampling vs. Thompson sampling
- **Pseudo-labeling model:** `Qwen/Qwen2.5-3B-Instruct`
- **Downstream classifier:** `bert-base-uncased`

## Folder Structure

```text
emotions_rec/
├── README.md
├── notebooks/
│   └── emotions_rec_repro.ipynb
├── src/
│   ├── preprocessing.py
│   ├── LDA.py
│   ├── labeling.py
│   ├── fine_tune.py
│   ├── main_cluster.py
│   ├── random_sampling.py
│   └── thompson_sampling.py
├── prompts/
│   └── few_shot_examples_joy.json
├── data/
│   ├── README.md
│   ├── raw/
│   └── processed/
└── run_configs/
    ├── random_run.txt
    └── thompson_run.txt
```

## Dataset Source

The `dair-ai/emotion` dataset is loaded directly from HuggingFace:

```python
from datasets import load_dataset
dataset = load_dataset("dair-ai/emotion")
```

No manual download is required. The dataset has 16,000 training, 2,000 validation, and 2,000 test examples across 6 emotion classes: sadness, joy, love, anger, fear, surprise.

## Recommended Environment

This experiment was tested using **Google Colab** with **Google Drive mounted**.

You will also need:

- a Hugging Face account
- a valid Hugging Face token
- access to the model `Qwen/Qwen2.5-3B-Instruct`

## Reproducing the Experiment in Colab

### Step 1: Open the notebook

Open:

`emotions_rec/notebooks/emotions_rec_repro.ipynb`

in Google Colab.

### Step 2: Run the repository setup cells

The notebook is already configured to clone or pull this repository and switch to the `emotions-rec` branch:

```python
REPO_URL = "https://github.com/ShenyaoZhang/DS-GA-3001-Data-Engineering-Project.git"
BRANCH = "emotions-rec"
```

These setup cells will:

- mount Google Drive
- clone or pull the repository into Drive
- change into the `emotions_rec` experiment folder
- create the expected data directories if needed

### Step 3: Install dependencies

The notebook installs the required packages from the repository root `requirements.txt`.

### Step 4: Run preprocessing cells

Run the preprocessing cells to:

- load the `dair-ai/emotion` dataset from HuggingFace
- create binary labels for `joy` vs. rest
- generate train / validation / test CSV files in LTS format

These files are written to:

`emotions_rec/data/processed/`

Expected processed files include:

- `train_inner_emotions_joy.csv`
- `val_emotions_joy.csv`
- `test_emotions_joy.csv`

### Step 5: Run the few-shot example cell

This auto-selects representative joy and non-joy examples from the training split and writes the prompt file:

`emotions_rec/prompts/few_shot_examples_joy.json`

### Step 6: Run source verification and syntax checks

The notebook checks that all required Python files exist in `src/` and can be compiled successfully.

### Step 7: Run the experiment

Run either the random sampling or Thompson sampling experiment cell.

The run configuration files are also provided in:

`emotions_rec/run_configs/`

### Step 8: Review generated outputs

Generated artifacts are written to multiple locations:

- **processed CSV / JSON outputs:** `data/processed/`
- **training checkpoints:** `results/`
- **saved model(s):** `models/`
- **state files:** experiment root, such as `selected_ids.txt`

## Run Configurations

### Random sampling

See:

`run_configs/random_run.txt`

### Thompson sampling

See:

`run_configs/thompson_run.txt`

## Important Notes

- The notebook is designed to be run from the repository-based Colab workflow, not from a private local folder path.
- The source code in `src/` is the authoritative implementation.
- The notebook should be treated as an experiment runner, not as the place where the Python modules are authored.
- Some generated files such as checkpoints and intermediate state files are not tracked by Git and are expected to be created during runtime.

## Expected Main Outputs

After a successful run, you should see files such as:

- `train_inner_emotions_joy_lda.csv`
- `train_inner_emotions_joy_data_labeled.csv`
- `train_inner_emotions_joy_training_data.csv`
- `train_inner_emotions_joy_model_results.json`

in:

`emotions_rec/data/processed/`

You should also see:

- checkpoint folders in `results/`
- saved fine-tuned models in `models/`

## Contact / Team Context

This folder contains only the Emotions `joy` experiment. Other datasets used in the full project report, such as Reuters and 20 Newsgroups, are documented separately in their corresponding sections of the team report.

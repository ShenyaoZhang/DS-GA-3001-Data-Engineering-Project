# 20 Newsgroups `rec.autos` Reproducibility Guide

This folder contains the reproducible code, notebook, prompts, and run configuration for the 20 Newsgroups `rec.autos` experiment in the DS-GA 3001 Data Engineering final project.

## Experiment Goal

This experiment evaluates whether the Lean-to-Sample (LTS) framework can improve minority-class discovery on an imbalanced text classification task.

- Dataset: 20 Newsgroups
- Positive class: `rec.autos`
- Negative class: all other newsgroups
- Sampling strategies compared: random sampling vs. Thompson sampling
- Pseudo-labeling model: `Qwen/Qwen2.5-3B-Instruct`
- Downstream classifier: `bert-base-uncased`

## Folder Structure

20news_rec_autos/
├── README.md
├── notebooks/
│   └── 20newsgroups_rec_autos_repro.ipynb
├── src/
│   ├── preprocessing.py
│   ├── LDA.py
│   ├── labeling.py
│   ├── fine_tune.py
│   ├── main_cluster.py
│   ├── random_sampling.py
│   └── thompson_sampling.py
├── prompts/
│   └── few_shot_examples_rec_autos.json
├── data/
│   ├── README.md
│   ├── raw/
│   └── processed/
└── run_configs/
    ├── random_run.txt
    └── thompson_run.txt

## Dataset Source

20 Newsgroups dataset source:
https://www.kaggle.com/datasets/crawford/20-newsgroups

Download the dataset and place the raw `.txt` source files into:
20news_rec_autos/data/raw/

## Recommended Environment

This experiment was tested using Google Colab with Google Drive mounted.

You will also need:
- a Hugging Face account
- a valid Hugging Face token
- access to the model `Qwen/Qwen2.5-3B-Instruct`

## Reproducing the Experiment in Colab

### Step 1: Open the notebook

Open:
20news_rec_autos/notebooks/20newsgroups_rec_autos_repro.ipynb

in Google Colab.

### Step 2: Run the repository setup cells

The notebook is already configured to clone or pull this repository and switch to the `20news-rec-autos` branch:

REPO_URL = "https://github.com/ShenyaoZhang/DS-GA-3001-Data-Engineering-Project.git"
BRANCH = "20news-rec-autos"

These setup cells will:
- mount Google Drive
- clone or pull the repository into Drive
- change into the `20news_rec_autos` experiment folder
- create the expected data directories if needed

### Step 3: Place raw data in the correct folder

Put the raw 20 Newsgroups `.txt` files into:
20news_rec_autos/data/raw/

### Step 4: Install dependencies

The notebook installs the required packages from the repository root `requirements.txt`.

### Step 5: Run preprocessing cells

Run the preprocessing cells to:
- parse the raw text files
- deduplicate examples
- create binary labels for `rec.autos`
- generate train / validation / test CSV files

These files are written to:
20news_rec_autos/data/processed/

Expected processed files include:
- `train_inner_20news_rec_autos.csv`
- `val_20news_rec_autos.csv`
- `test_20news_rec_autos.csv`

### Step 6: Run the few-shot example cell

This creates or verifies the prompt file:
20news_rec_autos/prompts/few_shot_examples_rec_autos.json

### Step 7: Run source verification and syntax checks

The notebook checks that all required Python files exist in `src/` and can be compiled successfully.

### Step 8: Run the experiment

Run either the random sampling or Thompson sampling experiment cell.

The run configuration files are also provided in:
20news_rec_autos/run_configs/

### Step 9: Review generated outputs

Generated artifacts are written to multiple locations:
- processed CSV / JSON outputs: `data/processed/`
- training checkpoints: `results/`
- saved model(s): `models/`
- state files: experiment root (for example `selected_ids.txt`)

## Run Configurations

### Random sampling

See:
run_configs/random_run.txt

### Thompson sampling

See:
run_configs/thompson_run.txt

## Important Notes

- The notebook is designed to be run from the repository-based Colab workflow, not from a private local folder path.
- The source code in `src/` is the authoritative implementation.
- The notebook should be treated as an experiment runner, not as the place where the Python modules are authored.
- Some generated files such as checkpoints and intermediate state files are not tracked by Git and are expected to be created during runtime.

## Expected Main Outputs

After a successful run, you should see files such as:
- `train_inner_20news_rec_autos_lda.csv`
- `train_inner_20news_rec_autos_data_labeled.csv`
- `train_inner_20news_rec_autos_training_data.csv`
- `train_inner_20news_rec_autos_model_results.json`

in:
20news_rec_autos/data/processed/

You should also see:
- checkpoint folders in `results/`
- saved fine-tuned models in `models/`

## Contact / Team Context

This folder contains only the 20 Newsgroups `rec.autos` experiment. Other datasets used in the full project report, such as Reuters and Emotions, are documented separately in their corresponding sections of the team report.

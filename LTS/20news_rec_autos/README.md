# 20 Newsgroups rec.autos experiment

This folder contains my 20 Newsgroups experiment for the `rec.autos` target class.

## Final setup
- Target class: `rec.autos`
- Labeling model: `Qwen/Qwen2.5-3B-Instruct`
- Labeling method: generation-based
- Prompt: shortened rec.autos prompt
- Few-shot file: `few_shot_examples_rec_autos.json`

## Best Thompson result
- `eval_f1_pos = 0.7192`
- `eval_precision_pos = 0.7836`
- `eval_recall_pos = 0.6646`

## Comparison to random sampling
Random sampling produced slightly lower positive-class F1 and lower recall than Thompson sampling, so Thompson was the better sampling strategy in this setup.

## Contents
- `prompts/labeling.py`: final labeling code
- `prompts/few_shot_examples_rec_autos.json`: final curated few-shot examples
- `notebooks/`: notebook used for the experiment

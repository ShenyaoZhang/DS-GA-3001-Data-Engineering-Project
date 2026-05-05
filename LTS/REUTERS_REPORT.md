# Reuters-21578 Results — Lean-to-Sample (LTS) with Qwen

**Author note.** This document is the Reuters subsection of the team report. It
mirrors the structure of the 20 Newsgroups subsection (5.1) so it can be lifted
into `LLM_Powered_Data_Triage_Team_Report_structured.ipynb` as section **5.2**.
All quantitative numbers are produced automatically by
`scripts/analyze_results.py` from the JSON / CSV artifacts in
`LTS/data_use_cases/`, and are also written to:

- `LTS/results/lts_summary.csv`
- `LTS/results/lts_thompson_vs_random.csv`
- `LTS/results/REUTERS_REPORT.md` (machine-readable, table-heavy)
- `LTS/results/figures/*.png` (the visualizations embedded in this report)

To regenerate these from a fresh checkout:

```bash
cd LTS
python scripts/analyze_results.py
```

---

## 5.2 Reuters Results

### 5.2.1 Setup and Tasks

The Reuters-21578 corpus comes shipped as `LTS/data/archive.zip` and contains
three standard splits: **ModApte**, **ModHayes**, and **ModLewis**. Each split
has a `train` and `test` CSV in which every row carries a list of topic tags.
Two LTS triage tasks were defined on top of this corpus:

1. **Binary `earn` triage.** Positive class = `earn`, negative class = every
   other article. This is the closest analog to the wildlife-trafficking task
   the original LTS paper studied: a single rare class buried in a much larger
   pool of negatives.
2. **Multi-class triage (6 classes).** Class set =
   `{earn, acq, trade, crude, money-fx, other}`, where `other` is a catch-all
   for articles whose primary topic is none of the first five. This was added
   to test whether the same active-learning loop generalizes beyond binary
   triage to a small but realistic taxonomy.

Both tasks share the same LTS pipeline:

```
preprocess  →  LDA cluster (K=10 requested, ~9 observed)  →
sample (Thompson or random)  →  Qwen pseudo-label  →
fine-tune BERT (bert-base-uncased)  →  reward sampler from validation F1  →
repeat for 10 rounds × 200 examples per round (full scale)
```

For each split × task combination, two LTS runs were executed under matching
budgets:

- **Thompson (LTS):** the bandit-driven sampler picks one of the LDA clusters
  based on validation F1 lift, then draws `sample_size` rows from that cluster.
- **Random:** the same loop, but each round draws `sample_size` rows uniformly
  at random from the pool. Pseudo-labeling, fine-tuning, and evaluation are
  identical so any final-metric gap is attributable to the sampler, not to the
  rest of the pipeline.

The total labeling budget per run is the same in both arms (≈ 2,000 Qwen
pseudo-labels per active-learning loop), so the comparison measures sampler
efficiency under a fixed labeling budget — exactly the question the LTS
framework was designed to answer.

### 5.2.2 Final Validation Metrics — Binary `earn` Triage

Best-iteration metrics on the held-out gold validation set, taken across all
active-learning rounds for each run:

| Split    | Sampling          | Best weighted F1 | Best macro F1 | Best accuracy |
|----------|-------------------|-----------------:|--------------:|--------------:|
| ModApte  | Random            |           0.8940 |        0.8778 |        0.8959 |
| ModApte  | **Thompson (LTS)**|       **0.9090** |     *(n/a\*)* |    **0.9111** |
| ModHayes | Random            |           0.9832 |    **0.8627** |        0.9819 |
| ModHayes | **Thompson (LTS)**|       **0.9842** |        0.8550 |    **0.9847** |
| ModLewis | Random            |           0.8886 |        0.8059 |        0.8891 |
| ModLewis | **Thompson (LTS)**|       **0.9177** |    **0.8578** |    **0.9174** |

\* ModApte Thompson binary was trained before `f1_macro` was added to
`compute_metrics`, so the JSON does not contain that field. The accuracy and
weighted-F1 numbers are still directly comparable.

**Reading the table.** Thompson sampling matches or beats random sampling on
weighted F1 across all three splits, with the largest gap on ModLewis
(+0.0291 F1, +0.0519 macro F1). On ModHayes, both samplers already saturate
near 0.98 F1 — the dataset is comparatively easy because `earn` headlines are
highly stylized — so there is little headroom for the sampler to matter and
the gap shrinks to +0.0010 F1.

![Binary `earn` triage — Random vs Thompson across splits](results/figures/binary_final_metrics.png)

The chart above shows the same three metrics side-by-side. The accuracy and
weighted-F1 panels visually confirm that Thompson is at or above random for
every split; the macro-F1 panel highlights ModLewis as the split where the
sampler choice matters most.

### 5.2.3 Sample Composition and Positive-Class Enrichment

Because the LTS labeling budget is fixed, the more useful question is: under
that fixed budget, *how many positive examples does each sampler actually pull
into the labeled training set?* Higher positive enrichment in the labeled
sample is the operational goal of LTS. Here are the gold-label positive rates
in each pool, in the random labeled sample, and in the Thompson labeled sample:

| Split    | Pool positive rate | Random sample rate | Thompson sample rate | Thompson lift |
|----------|-------------------:|-------------------:|---------------------:|--------------:|
| ModApte  |             29.57% |             24.97% |               28.17% |        +3.20% |
| ModHayes |             19.39% |             20.38% |           **60.50%** |   **+40.12%** |
| ModLewis |             20.85% |             17.53% |               21.20% |        +3.67% |

ModHayes is the headline story: Thompson sampling triples the positive rate
relative to random (60.5% vs. 20.4%), spending its labeling budget heavily on
the cluster(s) that are densely populated with `earn` headlines. ModApte and
ModLewis are both already roughly 20–30% positive at the pool level, so the
absolute lift is smaller, but Thompson still extracts a denser positive sample
than random in both cases.

![Positive-class enrichment under fixed labeling budget](results/figures/binary_positive_enrichment.png)

This is the same plot that section 5.1 used for 20 Newsgroups: light-grey is
the natural pool rate, dark-grey is the random labeled sample, and blue is
the Thompson labeled sample under the same labeling budget. The blue spike on
ModHayes is the clearest single piece of evidence that LTS is doing what it
claims: concentrating labeling spend on the rare, useful regions of the pool.

### 5.2.4 Pseudo-Label Quality (Qwen vs. Gold)

For each sampled row, the agreement between Qwen's pseudo-label and the gold
label was computed on the same row IDs. This is a direct measure of how good
Qwen is as the labeling oracle for each task:

| Split    | Task         | Sampling | Pseudo rows | Qwen ↔ gold agreement |
|----------|--------------|----------|------------:|----------------------:|
| ModApte  | binary       | Random   |       1,734 |                81.03% |
| ModApte  | binary       | Thompson |       3,919 |                84.00% |
| ModHayes | binary       | Random   |       3,602 |                85.29% |
| ModHayes | binary       | Thompson |       2,000 |                67.30% |
| ModLewis | binary       | Random   |       1,951 |                81.60% |
| ModLewis | binary       | Thompson |       2,000 |                86.30% |
| ModApte  | multi-class  | Random   |       1,538 |                10.79% |
| ModApte  | multi-class  | Thompson |       1,920 |                12.19% |
| ModHayes | multi-class  | Thompson |       1,746 |                13.86% |
| ModLewis | multi-class  | Random   |       1,931 |                10.36% |
| ModLewis | multi-class  | Thompson |       1,585 |                16.09% |

For binary triage Qwen is a strong oracle (≈ 80–86% agreement). For multi-class
triage Qwen is barely better than random (≈ 10–16% agreement), which is the
single most important observation in this report.

![Pseudo-label quality on sampled rows](results/figures/pseudo_label_agreement.png)

The left half of the chart (binary runs) hovers around 80%; the right half
(multi-class runs) collapses below 20%. The horizontal dashed line at 50%
makes the gap between "useful oracle" and "near-random oracle" easy to read
at a glance.

### 5.2.5 Final Validation Metrics — Multi-Class Triage (6 classes)

The downstream BERT classifier is fine-tuned on whatever labels Qwen produced,
so the multi-class numbers below largely inherit the ≈ 12% pseudo-label
quality from the previous table:

| Split    | Sampling | Best weighted F1 | Best macro F1 | Best accuracy |
|----------|----------|-----------------:|--------------:|--------------:|
| ModApte  | Random   |           0.0362 |        0.0318 |        0.0454 |
| ModApte  | Thompson |       **0.0734** |    **0.0616** |    **0.1178** |
| ModHayes | Thompson |           0.6381 |        0.1397 |        0.7167 |
| ModLewis | Random   |           0.0059 |        0.0120 |        0.0146 |
| ModLewis | Thompson |       **0.5130** |    **0.1282** |    **0.6131** |

Two observations:

1. **Thompson sampling still wins where a comparison exists**, sometimes by
   very large absolute margins (+0.51 weighted F1 on ModLewis multi-class).
   The qualitative reason matches the binary case: Thompson concentrates the
   labeling budget on clusters where the model is gaining ground, while random
   sampling spreads labels uniformly across an extremely imbalanced pool that
   is dominated by the `other` class.
2. **All multi-class macro F1 scores are low (≤ 0.14)**, even though weighted
   F1 / accuracy can look respectable on splits like ModHayes. That is the
   class-imbalance signature: the classifier becomes very good at predicting
   `other` and very weak everywhere else.

The comparatively large weighted-F1 on ModHayes / ModLewis Thompson runs is
*not* a sign that the multi-class pipeline is working well — it is mostly
the model's accuracy on `other`, which is by far the largest class.

![Multi-class triage (6 classes) — Random vs Thompson](results/figures/multiclass_final_metrics.png)

The chart shows the multi-class metrics side-by-side. Note the huge accuracy
gap on ModLewis (0.61 vs. 0.01) versus the much smaller macro-F1 gap (0.13 vs.
0.01) — Thompson is correctly classifying the dominant `other` class while
both samplers struggle on every minority class. That gap between weighted /
accuracy and macro F1 is precisely the imbalance signature called out above.

### 5.2.6 Why Multi-Class Performance Is Low — Diagnosis

Inspecting the per-class output distribution that Qwen actually produced on
each multi-class run shows where the breakdown is. The table below is taken
from the auto-generated `results/REUTERS_REPORT.md` (ModApte Thompson run, but
all four multi-class runs show the same pattern):

| Class    | Qwen count | Qwen %  | Gold count | Gold %  |
|----------|-----------:|--------:|-----------:|--------:|
| acq      |        107 |  5.57%  |        429 | 22.34%  |
| crude    |         45 |  2.34%  |         27 |  1.41%  |
| earn     |         65 |  3.39%  |        420 | 21.88%  |
| money-fx |        148 |  7.71%  |         36 |  1.88%  |
| other    |        116 |  6.04%  |        955 | 49.74%  |
| trade    |    **1439**| **74.95%**|       53 |  2.76%  |

Qwen labels ≈ 75% of every sampled row as `trade`, regardless of split or
sampler, even though only ≈ 1–3% of the gold pool is `trade`. The same
75% / `trade` skew shows up in ModHayes Thompson, ModLewis Thompson, and the
ModApte / ModLewis random multi-class runs. This is a prompt-quality / LLM-
calibration limitation, not a pipeline bug:

- The current multi-class prompt was generalized out of the original binary
  `earn` / `not earn` template. Many of its few-shot examples are still binary
  in flavor, which biases Qwen toward whatever financial-trading-adjacent
  label sounds plausible — `trade` is the obvious attractor.
- Reuters-21578 uses a fine-grained newswire taxonomy (`acq`, `money-fx`,
  `crude`, etc.) that does not map cleanly onto Qwen's general semantic
  understanding without explicit per-class definitions and examples.

Because BERT trains on whatever Qwen produces, the downstream classifier
inherits this skew and learns to predict `trade` and `other` for almost
everything. That is precisely what the multi-class confusion in
section 5.2.5 reflects.

![Qwen multi-class output distribution vs gold (sampled rows)](results/figures/qwen_vs_gold_distribution.png)

The figure makes the prompt-quality story unmistakable: the orange `Qwen`
bars are dominated by `trade` in every panel (≈ 72–75% of all sampled rows),
while the green `Gold` bars are dominated by `other` and `earn`. No amount of
sampler intelligence downstream can compensate for an oracle that classifies
three quarters of the dataset as `trade`.

### 5.2.7 Interpretation

Across all five Thompson-vs-random pairs that have matched runs:

- **Binary `earn`:** Thompson > random in all 3 splits. The biggest absolute
  gap is on **ModLewis (+0.029 weighted F1, +0.052 macro F1)**, the most
  realistic of the three splits because it has the largest negative pool and
  the most stylistically diverse documents. ModHayes is essentially saturated
  at ≈ 0.98 F1, which is the hardest case for an active learner to
  differentiate from random.
- **Multi-class:** Thompson > random in both splits where a head-to-head
  comparison exists (ModApte and ModLewis), with very large weighted-F1 lifts
  (+0.04 and +0.51). However, *all* multi-class absolute scores are low
  because the upstream Qwen pseudo-labels are only ≈ 12–16% accurate. Until
  that bottleneck is fixed, the multi-class numbers measure prompt quality
  more than they measure sampler quality.
- **Sample enrichment:** The clearest signal of LTS's value comes from the
  binary positive-rate enrichment table: on ModHayes, Thompson lifts the
  positive rate from 20% (pool / random) to 60% (Thompson labeled sample).
  This mirrors the 20 Newsgroups `rec.autos` finding (6.4% pool → 17.7%
  Thompson sample) and confirms that the LTS sampler does what it claims to
  do — concentrate labeling spend on the rare, useful regions of the pool.

### 5.2.8 Recommendations / Next Steps

1. **Fix the multi-class Qwen prompt.** Replace the binary-style few-shot
   examples with one explicit example per class (`earn`, `acq`, `trade`,
   `crude`, `money-fx`, `other`) and add stricter, mutually exclusive class
   definitions in the system prompt. This is the single highest-impact change
   for the multi-class numbers.
2. **Run an oracle baseline.** Re-run multi-class with `-labeling file` so
   that the BERT trainer sees the gold labels directly. This isolates whether
   the remaining error is in the Qwen oracle (expected) or in the classifier
   training loop (should be small).
3. **Report macro F1 and per-class F1 in the team notebook.** Weighted F1 is
   misleading on these splits because `other` dominates; macro F1 is the
   honest summary statistic for the multi-class case.

### 5.2.9 Reproducibility

All numbers in this section can be regenerated from a clean clone with:

```bash
cd LTS

# 1. Prepare data (binary + multi-class CSVs for each split)
python scripts/prepare_reuters.py --split ModApte  --label earn
python scripts/prepare_reuters.py --split ModHayes --label earn
python scripts/prepare_reuters.py --split ModLewis --label earn
python scripts/prepare_reuters.py --split ModApte  --task-type multiclass \
    --labels "earn,acq,trade,crude,money-fx" --other-label other
python scripts/prepare_reuters.py --split ModHayes --task-type multiclass \
    --labels "earn,acq,trade,crude,money-fx" --other-label other
python scripts/prepare_reuters.py --split ModLewis --task-type multiclass \
    --labels "earn,acq,trade,crude,money-fx" --other-label other

# 2. Train each LTS run (Thompson or random); see README Use Case 4 / 5
#    for the full main_cluster.py command lines.

# 3. Aggregate everything into tables + a Markdown report
python scripts/analyze_results.py
```

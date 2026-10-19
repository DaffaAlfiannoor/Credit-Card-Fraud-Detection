# Credit Card Fraud Detection

Fraud detection on a severely imbalanced dataset: only **0.172%** of transactions are
fraudulent. This project trains five tree-based models — RandomForest, AdaBoost,
CatBoost, XGBoost and LightGBM — and then takes a hard look at two things that are
routinely done wrong on data like this:

1. **ROC-AUC computed from class labels is misleading**, and it silently makes good
   models look bad.
2. **Class weighting does not automatically help** — on this dataset it actively hurts.

| | |
|---|---|
| Best model by PR-AUC on the test set | **CatBoost — 0.851** |
| Best model by ROC-AUC on the test set | CatBoost — 0.978 |
| Best model by F1 on the test set | CatBoost — 0.843 |
| Class weighting | **Harmful** (LightGBM PR-AUC 0.689 → 0.298) |
| SMOTE / undersampling | Not used — see [Why not SMOTE](#why-not-smote) |
| Primary metrics | PR-AUC, recall, precision, F1 at a tuned threshold |

---

## Table of Contents

- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Notebook Walkthrough](#notebook-walkthrough)
- [Methodology](#methodology)
  - [Data Splitting](#data-splitting)
  - [Why ROC-AUC from Class Labels Is Misleading](#why-roc-auc-from-class-labels-is-misleading)
  - [Threshold Selection and Metrics](#threshold-selection-and-metrics)
  - [Why Not SMOTE](#why-not-smote)
- [Results](#results)
  - [Head-to-Head on a Single Test Set](#head-to-head-on-a-single-test-set)
  - [LightGBM with 5-Fold Cross-Validation](#lightgbm-with-5-fold-cross-validation)
  - [Does Class Weighting Help?](#does-class-weighting-help)
  - [Where the Models Disagree](#where-the-models-disagree)
- [Key Findings](#key-findings)
- [How to Run](#how-to-run)
- [Limitations](#limitations)
- [References](#references)

---

## Dataset

`creditcard.csv` — **284,807 rows × 31 columns**, no missing values.

| Column | Description |
|---|---|
| `Time` | Seconds elapsed since the first transaction in the dataset (covers ~48 hours) |
| `V1`–`V28` | Principal components from a PCA transformation; original feature names are withheld for confidentiality |
| `Amount` | Transaction amount |
| `Class` | Target — `0` = legitimate, `1` = fraud |

Class distribution:

| Class | Count | Share |
|---|---|---|
| Legitimate (`0`) | 284,315 | 99.828% |
| Fraud (`1`) | 492 | 0.172% |

The dataset is the well-known *Credit Card Fraud Detection* dataset published by the
Machine Learning Group at Université Libre de Bruxelles (ULB):

**https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud**

The CSV is **not** committed to this repository (see `.gitignore`). Download it from
the Kaggle link above and place it at the repository root before running the notebook.

---

## Repository Structure

```
.
├── notebook.ipynb    # full analysis: EDA → modeling → evaluation
├── creditcard.csv    # dataset (not committed, download from Kaggle)
├── catboost_info/    # CatBoost training logs, generated automatically
└── README.md
```

---

## Notebook Walkthrough

The notebook is organized into the following sections:

1. **Read Data** — load `creditcard.csv` and inspect the first rows.
2. **Check the Data** — shape, descriptive statistics, missing-value check, and a bar
   chart of the class distribution.
3. **Exploration Data Analysis**
   - *Transaction in time* — density of transaction time per class, plus hourly
     aggregates of transaction count, total amount, mean, median, min and max.
   - *Transactions amount* — box plots comparing amounts between classes, and a
     scatter of fraudulent amounts over time.
   - *Features correlation* — Pearson correlation heatmap over all 31 columns, plus
     regression plots for the feature/amount pairs that show notable correlation.
   - *Features density plot* — kernel density estimates for `V1`–`V28` split by class,
     which shows how separable the PCA components are.
4. **Predictive Models** — define predictors, split the data, and train five models:
   RandomForest, AdaBoost, CatBoost, XGBoost, LightGBM.
5. **Training and validation using cross-validation** — LightGBM with 5-fold KFold.
6. **Comparing models on one test set** — all five models scored on the same test set
   with the same metrics and a threshold tuned on validation.
7. **Does class weighting help?** — a controlled experiment on `scale_pos_weight`.
8. **Kesimpulan** — the conclusion, written in Indonesian.

> Note: the narrative sections are mixed-language — most analysis is in English, the
> final conclusion is in Indonesian. Section 8 is kept as written in the notebook.

---

## Methodology

### Data Splitting

The data is split once, with `random_state = 2018`:

```python
train_df, test_df  = train_test_split(data_df, test_size=0.20, random_state=2018, shuffle=True)
train_df, valid_df = train_test_split(train_df, test_size=0.20, random_state=2018, shuffle=True)
```

| Set | Rows | Share |
|---|---|---|
| Train | 182,276 | 64% |
| Validation | 45,569 | 16% |
| Test | 56,962 | 20% |

The training set contains only about **290 fraudulent transactions (0.16%)**. The
validation set is used exclusively to choose the decision threshold; the **test set is
never used to make any modeling decision**.

### Why ROC-AUC from Class Labels Is Misleading

For RandomForest, AdaBoost and CatBoost the notebook first reports AUC computed from
`predict` (hard 0/1 labels), then recomputes it from `predict_proba`. The gap is large
and systematic:

| Model | AUC from labels | AUC from probabilities |
|---|---|---|
| RandomForest | 0.853 | 0.910 |
| AdaBoost | 0.814 | 0.983 |
| CatBoost | 0.858 | 0.984 |

AUC computed from labels only ever sees two discrete values, so **all ranking
information is discarded**. Every true positive is tied with every other true positive,
and every false positive with every false positive. The resulting number depends on the
operating point, not on how well the model separates the classes.

The practical consequence is severe: AdaBoost looks like the *worst* of the three
(0.814) when measured from labels, but is essentially tied with CatBoost (0.983 vs
0.984) when measured properly. Any ranking built on label-based AUC is meaningless.

### Threshold Selection and Metrics

The default 0.5 threshold is arbitrary. Instead, the threshold is chosen from the
precision-recall curve on the **validation set** (the one that maximizes F1) and then
applied unchanged to the test set:

```python
def pick_threshold(y_true, proba):
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return thresholds[f1[:-1].argmax()]
```

Accuracy is deliberately **not** reported. With a 0.16% fraud rate, a model that never
predicts fraud already reaches ~99.8% accuracy. That number says nothing about fraud
detection and invites exactly the wrong conclusion.

### Why Not SMOTE

The notebook does not use SMOTE or undersampling. With only ~290 fraudulent
transactions in the training set, SMOTE would synthesize hundreds of thousands of
rows from 290 original points. Validation metrics computed on such data no longer
reflect real traffic, and the synthetic points occupy the same convex neighborhoods
as the originals — so the model learns to reproduce the interpolation, not the
decision boundary.

The chosen approach is: **train on the data as-is, then tune the decision threshold
and report PR-AUC.** For a screening task this is both simpler and more honest than
resampling.

---

## Results

### Head-to-Head on a Single Test Set

Every model trained on the same training set, scored on the same test set, with a
threshold tuned on the same validation set:

| Model | Threshold | PR-AUC | ROC-AUC | Recall | Precision | F1 |
|---|---|---|---|---|---|---|
| **CatBoost** | 0.653 | **0.851** | 0.978 | 0.75 | 0.962 | **0.843** |
| RandomForest | 0.590 | 0.841 | 0.944 | 0.74 | 0.949 | 0.831 |
| XGBoost | 0.299 | 0.822 | 0.972 | 0.77 | 0.865 | 0.815 |
| AdaBoost | 0.458 | 0.751 | 0.971 | 0.76 | 0.731 | 0.745 |
| LightGBM | 0.998 | 0.348 | 0.947 | 0.80 | 0.364 | 0.500 |

Reading the table:

- **CatBoost wins on every metric that matters here** — PR-AUC, F1 and precision. At its
  threshold, 96.2% of flagged transactions are genuinely fraudulent, and it catches 75%
  of the fraud present in the test set.
- **RandomForest is a close second.** Its ROC-AUC is the lowest of the five (0.944) yet
  its PR-AUC is the second highest (0.841). This is the clearest illustration that
  ROC-AUC and PR-AUC rank models differently.
- **XGBoost has better ROC-AUC than RandomForest (0.972 vs 0.944) but worse PR-AUC**
  (0.822 vs 0.841). ROC-AUC weights performance evenly across the whole false-positive
  range, most of which is operationally irrelevant. PR-AUC only cares about the top of
  the ranked list — the part that actually gets reviewed.
- **LightGBM is the cautionary tale.** Trained with `scale_pos_weight=150` and stopped
  at iteration 29, its probabilities are badly miscalibrated. The tuned threshold
  jumped to 0.998, and at that threshold recall is 0.80 but precision is only 0.364 —
  meaning roughly two out of three transactions flagged as fraud are actually legitimate.

### LightGBM with 5-Fold Cross-Validation

Cross-validated LightGBM, with the test prediction averaged over the five fold models:

| Fold | Validation AUC |
|---|---|
| 1 | 0.998 |
| 2 | 0.966 |
| 3 | 0.943 |
| 4 | 0.993 |
| 5 | 0.985 |
| **OOF (full train)** | **0.970** |
| **Test (average of 5 models)** | **0.968** |

The spread across folds is large — 0.943 to 0.998, standard deviation ≈ 0.020 — because
each fold's validation split contains only 47 to 77 fraudulent transactions. A single
hard-to-classify fraud case moves fold AUC by more than a percentage point. The OOF
score of 0.970 should be read with that variance in mind; it is not a precise estimate.

Cross-validated LightGBM (0.968 on test) clearly beats the single-split LightGBM
(0.947) and lands close to XGBoost (0.972), which confirms that the earlier poor
LightGBM result was caused by the `scale_pos_weight=150` setting, not by the algorithm.

### Does Class Weighting Help?

A controlled comparison on the same split: identical LightGBM hyperparameters, three
values of `scale_pos_weight`, evaluated on the validation set.

| Configuration | Best iteration | PR-AUC | ROC-AUC |
|---|---|---|---|
| No weighting | 156 | **0.689** | 0.973 |
| `scale_pos_weight=150` | 29 | 0.298 | 0.958 |
| `scale_pos_weight=627` | 1 | 0.360 | 0.964 |

Class weighting **hurts**, badly.

- **No weighting** gives the best PR-AUC (0.689) and trains for 156 iterations.
- **Weight 150** (an arbitrary guess) drops PR-AUC to 0.298 and stops after 29 iterations.
- **Weight 627** — the *actual* ratio of legitimate to fraudulent transactions in the
  training set — drops PR-AUC to 0.360 and stops after **one** iteration. The model
  effectively never trains.

The mechanism: a large `scale_pos_weight` makes the gradient contribution of the
positive class dominate. The validation loss is already worse than its starting value
at the first iteration, so early stopping fires immediately. A bigger weight is not a
better detector — it is just a broken training signal.

Note that the PR-AUC of 0.689 here differs from the 0.348 in the comparison table
because they are measured on different sets: this one on the validation set, the other
on the test set using a threshold derived from a weighted model. Both point in the same
direction: weighting degrades ranking quality.

### Where the Models Disagree

Some observations that are worth recording because they shape how the numbers should
be read:

- **PR-AUC and ROC-AUC rank the five models differently.** CatBoost tops both, but
  RandomForest beats XGBoost on PR-AUC while losing on ROC-AUC. If the deliverable is a
  ranked queue for human review, PR-AUC is the metric to optimize.
- **Precision at the tuned threshold is high across the top three** (0.865–0.962),
  which means the alert volume stays manageable. That matters operationally: a queue
  where 96% of items are true fraud is workable; one where 36% are is not.
- **Recall plateaus around 0.75–0.80** for every model. Roughly one in four frauds is
  missed by all of them at their F1-optimal thresholds. Pushing recall higher requires
  accepting a large precision drop — a business decision, not a modeling one.

---

## Key Findings

1. **Never measure AUC from class labels on imbalanced data.** The gap can exceed 0.13
   (AdaBoost: 0.814 → 0.983) and can invert the ranking of models. Always use
   `predict_proba`.
2. **PR-AUC is more relevant than ROC-AUC** when what matters is the quality of the top
   of the ranked list rather than the entire curve. ROC-AUC is dominated by the easy
   negative examples that make up 99.8% of the data.
3. **Class weighting is not a free win.** On LightGBM here it cut PR-AUC from 0.689 to
   0.298 and, at the true class ratio, reduced training to a single iteration.
4. **Threshold tuning is where the real gains are.** Moving the threshold off 0.5
   improved the usable operating point more than any hyperparameter change.
5. **CatBoost is the best model for this dataset**, leading on PR-AUC (0.851), F1
   (0.843) and precision (0.962), with RandomForest a close second.
6. **Report the confusion-matrix consequences, not accuracy.** Recall 0.75 with
   precision 0.96 means one in four frauds slips through, while 1 in 25 flagged
   transactions is a false alarm — a far better outcome than mass false positives that
   block legitimate customers.

---

## How to Run

The notebook was executed with **Python 3.11** in a conda environment named `py311-ml`.

```bash
# 1. Create and activate the environment
conda create -n py311-ml python=3.11
conda activate py311-ml

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn plotly scikit-learn \
            catboost lightgbm xgboost jupyter

# 3. Download creditcard.csv from Kaggle and place it in the repository root
#    https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

# 4. Launch the notebook
jupyter notebook notebook.ipynb
```

All randomness is controlled by `RANDOM_STATE = 2018`, so results should be
reproducible. Training CatBoost and the 5-fold LightGBM loop are the slowest steps;
expect several minutes on a laptop CPU.

A note on the environment: the LightGBM version used prints an `LGBMDeprecationWarning`
about `eval_set` being deprecated in favour of `eval_X` / `eval_y`. It is harmless and
does not affect results.

---

## Limitations

- **A single train/validation/test split.** No repeated cross-validation was run for
  the final comparison, so the test-set numbers carry an uncertainty that is not
  quantified. With ~98 frauds in the test set, one transaction moves recall by about
  one percentage point.
- **Small effective sample size.** The 5-fold CV results show fold AUCs ranging from
  0.943 to 0.998. That spread is a property of the data (47–77 frauds per fold), not a
  flaw in the model, but it means point estimates are noisy.
- **`Time` is used as a raw feature.** It is seconds since the first transaction, not a
  derived feature like hour-of-day or day-of-week. The hourly aggregates in the EDA
  suggest time-of-day carries signal that the models do not exploit directly.
- **The dataset is already PCA-transformed and anonymized.** Feature-level
  interpretation is impossible beyond `Amount` and `Time`, which limits explainability
  and any attempt to derive business rules from feature importance.
- **No calibration analysis.** The LightGBM discussion notes that weighted models
  produce miscalibrated probabilities, but no reliability curve or calibration step
  (Platt scaling, isotonic regression) is applied. If probabilities are to be used
  directly rather than as a ranking, calibration should be added.
- **No cost-sensitive evaluation.** False positives and false negatives have very
  different business costs, but the threshold was chosen by maximizing F1, which treats
  them symmetrically.

---

## References

- **Dataset** — Machine Learning Group, Université Libre de Bruxelles (ULB).
  *Credit Card Fraud Detection.* Kaggle.
  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- Dal Pozzolo, A., Caelen, O., Le Borgne, Y.-A., Waterschoot, S., & Bontempi, G.
  (2014). *Learned lessons in credit card fraud detection from a practitioner
  perspective.* Expert Systems with Applications, 41(10), 4915–4928.
- Saito, T., & Rehmsmeier, M. (2015). *The Precision-Recall Plot Is More Informative
  than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets.*
  PLoS ONE, 10(3), e0118432.
- Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). *SMOTE:
  Synthetic Minority Over-sampling Technique.* Journal of Artificial Intelligence
  Research, 16, 321–357.

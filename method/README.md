# Method: multi-arm uplift with cost and capacity

This document describes the approach behind the case study in
[`../case_study_beer`](../case_study_beer): what is estimated, how, and how the
estimates become decisions. It explains the method in words, formulas and
pseudocode. The implementation itself (a Python package) is not published.

## 1. The problem

A business has a "do nothing" option (control, arm 0) and K possible actions
(arms 1..K). Each unit (a customer, an outlet) gets exactly one arm per decision
cycle. Each arm has a cost, and some arms are limited (only so many rep visits per
region per month). The question for every unit is:

> Which arm gives the highest expected net value for this unit, and does the
> resulting allocation respect the limits?

Answering it needs the effect of every arm on every unit, not one average effect.
An arm can help one kind of unit and lose money on another.

## 2. Notation

| Symbol | Meaning |
|---|---|
| `x` | features of a row (unit and cycle) |
| `y` | outcome of the row, binary or continuous |
| `mu_k(x)` | expected outcome under arm k: `E[y | arm = k, x]` |
| `tau_k(x)` | effect of arm k against control: `mu_k(x) - mu_0(x)` |
| `e_k(x)` | propensity: `P(arm = k | x)` in the historical data |
| `V` | money per unit of outcome (for example gross margin per case) |
| `c_k` | cost of arm k per row |

Every effect is measured against control. The difference between two treatments
is the difference of two effects, with the uncertainty of both.

## 3. Data the method expects

* Randomised data is the normal and safest case.
* Mixed data also works: some rows randomised, the rest assigned by a known
  business rule, provided the inputs of the rule are among the features. The
  propensity model can then learn the rule, and the doubly robust estimates
  correct for it.
* Where an arm was never given to some kind of row, there is no evidence of its
  effect there. The method detects this (low propensity) and never chooses the arm
  for those rows; it cannot create the missing evidence.
* When the same unit appears in many cycles, all its rows stay together in every
  split (train / test, cross-fitting folds, bootstrap).

## 4. Pipeline

```
historical rows (x, arm, y)
  -> feature filter          drop unusable features, with a reason for each
  -> tuning                  choose learner and model complexity on held-out units
  -> nuisance models         e_k(x) and mu_k(x), cross-fitted, shared by all learners
  -> final stage per arm     tau_k(x) for k = 1..K
  -> policy                  one arm per row: best net value, under shares or capacities
  -> evaluation              doubly robust value of any policy, with paired bootstrap CIs
  -> calibration (optional)  rescale effect sizes per arm, keeping their order
  -> robustness checks       overlap, clip, stability, bootstrap, placebo, A/A, overfit
  -> interpretation          importance with a noise floor, decision importance,
                             effects along features per group, one row explained
  -> scoring                 new rows -> effects -> decisions
```

## 5. Estimation

### 5.1 Shared, cross-fitted nuisance models

The two kinds of nuisance models are fitted once and shared by every learner:

```
split units into F folds (all rows of a unit in the same fold)
for each fold f:
    train on the other folds:
        propensity model:  one multinomial classifier for P(arm = k | x), all arms at once
        outcome models:    one regressor (or classifier) per arm, on the rows given that arm
    predict e_k(x) and mu_k(x) for the rows of fold f
```

Every row's nuisance predictions come from models that never saw it
(cross-fitting). Gradient-boosted trees are the default base model.

A feature can be marked "nuisance only" (for example the month index): it is used
by the outcome and propensity models to absorb seasonality and drift, but hidden
from the effect models, because the effects must hold in future cycles.

### 5.2 Doubly robust scores

For every row and arm:

```
gamma_k = mu_k + 1[arm = k] * (y - mu_k) / max(e_k, clip)
```

The mean of `gamma_k` over rows estimates the average outcome under arm k, and
stays unbiased if either the outcome model or the propensity model is right
("doubly robust"). `clip` (0.01) stops a row with a tiny propensity from getting
a huge weight. These scores drive the DR learner, every policy value, the
calibration and the robustness checks.

### 5.3 One final stage per arm

Each arm k = 1..K gets its own effect estimate against control, using the shared
nuisances. Several standard meta-learners are available:

| Learner | Effect of arm k |
|---|---|
| DR learner | regress `gamma_k - gamma_0` on `x` |
| T-learner | `mu_k(x) - mu_0(x)`, no extra model |
| S-learner | one model on `[x, arm]`, predicted with each arm switched on |
| X-learner | impute effects on the rows of arm k and of control, regress each, blend with `e_k / (e_0 + e_k)` |
| R-learner | residual-on-residual regression on the rows of arms 0 and k |

Because the nuisances and folds are shared, comparing learners is fair and cheap:
only the final stages differ. The per-arm decomposition keeps each arm inspectable
on its own.

### 5.4 Tuning

Candidates (learner, tree complexity, feature-filter thresholds) are compared on
units they were not fitted on, and never grade themselves:

```
split training units into tune-train and validation (whole units)
referee = DR scores gamma_k cross-fitted on the validation rows only (fixed for all candidates)
for each candidate:
    fit on tune-train
    choose an arm for every validation row (the policy of section 6)
    score = mean over validation rows of ( V * gamma[row, chosen arm] - c[chosen arm] )
best = highest score
tied = candidates whose paired-bootstrap difference to the best includes 0
winner = the simplest tied candidate (fewest features, then fewest leaves, then a preferred learner order)
```

Grid, random or Bayesian (TPE) search can generate the candidates.

## 6. From effects to decisions

The net value of arm k for a row is

```
value_k = V * tau_k(x) - c_k          (control: 0)
```

An arm whose estimated propensity for the row is below a floor (0.02) is not
eligible: there is no evidence for it there. Three strategies:

```
best:       arm = argmax over eligible arms of value_k   (control if none is positive)

top_share:  given shares s_k (in a filling order, scarcest arm first)
            for each arm k in order:
                among rows not yet assigned, sort by value_k, highest first
                give arm k to the top s_k * (all rows)     [only rows with value_k > min_gain]
            remaining rows -> control

capacity:   the same filling, with a NUMBER of rows per arm,
            applied separately in every capacity bucket (for example region x month)
```

With `min_gain = 0`, a share or a capacity is a maximum: a row never gets an arm
that is expected to lose money. The shares of a `top_share` policy can be chosen by
a sweep: every combination on a grid is valued (section 7) and the simplest
combination tied with the best is selected.

## 7. Evaluation

A policy is valued with doubly robust scores, never with the model's own
predicted effects:

```
value(policy) = mean over rows of ( V * gamma[row, chosen arm] - c[chosen arm] )
```

* On held-out units, a separate **referee** learner is cross-fitted on those rows
  only, and its scores `gamma` value the decisions of the model trained elsewhere.
* Confidence intervals come from a bootstrap that resamples whole units.
* Two policies are compared with a **paired** difference on the same bootstrap
  samples, which is much tighter than comparing two separate intervals.
* Baselines are always included: everyone gets control, the best single arm for
  everyone, and the business rule (when the data records it).
* Per arm, an **uplift curve** sorts rows by predicted effect and shows the gain of
  treating the top q% only; the **Qini** is the area between that curve and random
  targeting.

## 8. Calibration

A learner can rank rows well and still get effect sizes wrong, which matters when
the policy compares `V * tau_k - c_k` across arms. Per arm:

```
bin rows by predicted effect
in each bin: observed effect = mean(gamma_k - gamma_0)
slope = regression of observed on predicted     (1 = calibrated, < 1 = predictions too spread out)
optional: isotonic mapping predicted -> observed, fitted out of fold (keeps the order)
```

Calibration is applied only if it improves the decisions on held-out rows.

## 9. Robustness checks

| Check | How | Passes when |
|---|---|---|
| Overlap | distribution of estimated propensities per arm | low-propensity regions are known and excluded by the floor |
| Clip sensitivity | policy value under several clips | value and gains barely move |
| Stability | policy gain per month and per segment | no group where the model loses to the best single arm |
| Bootstrap | refit on subsamples of units, compare decisions | high agreement with the original arm per row |
| Placebo | refit with the arm column shuffled (19 times) | real gain above every shuffled gain (p <= 0.05) |
| A/A | refit with an outcome the arm cannot affect (a pre-treatment outcome) | gain CI includes 0 |
| Overfit | same metrics in sample, out of fold and on held-out units | out of fold and held out agree |

## 10. Interpretation

Ordered as "which features matter", then "how they drive the effect":

* **Importance** per arm: mean absolute tree SHAP contribution to `tau_k`.
* **Noise floor**: refit once with shuffled copies of real features ("shadow"
  features). A feature is a driver only if it beats the best shadow by a margin.
* **Decision importance**: shuffle one feature, decide again, and measure how many
  rows switch arm and how much DR value is lost.
* **Effects along a feature**, per group (for example per channel): partial
  dependence, SHAP, or binned with the DR observed effect and its CI overlaid, so
  the model's curve can be checked against the data. Two-feature heatmaps show
  interactions.
* **One row explained**: the net value gap between the chosen arm and the runner-up,
  split into feature contributions, the part common to all rows, and the cost
  difference.

## 11. Design choices

| Choice | Alternative | Reason |
|---|---|---|
| Per-arm decomposition on shared nuisances | Native multi-arm estimators | Reuses well-understood binary meta-learners, each arm can be inspected alone, learners are compared on identical nuisances. |
| Doubly robust scores everywhere (learning, evaluation, calibration, checks) | Inverse propensity weighting or outcome models alone | Unbiased if either model is right; required on mixed randomised and rule-based data. |
| Referee scores for evaluation and tuning | Evaluate with the model's own predictions | A model graded on its own predictions always looks good. |
| Folds, splits and bootstrap by unit | Row-level splits | The same unit appears every cycle; row-level splits leak. |
| Time as a nuisance-only feature | Time in every model | Effects must hold in future cycles. |
| Propensity floor in the policy | Trust every estimate | No evidence, no decision. |
| Capacity per arm, per group and per period, filled scarcest arm first | Best arm only | Real treatments are limited in absolute numbers per region and month. |
| `min_gain = 0` for capacity and shares in the case study | Fill every share or capacity | A limit is a maximum, not a quota. |
| Simplest-tied winner in tuning and in the share sweep | Highest score | The highest score among many noisy candidates is partly luck. |
| Fallbacks warn loudly | Silent fallback | A silent fallback produces numbers that look valid and are not. |

## 12. Limitations

* Training is single-machine; scoring can be distributed.
* On non-randomised data the estimates are only as good as the propensity model
  and the overlap.
* Effects are measured against control and assumed not to spill over between
  units or across cycles.
* There is no monitoring after deployment: keep a random holdout and re-evaluate.

# Multi-arm uplift with cost and capacity: method and case study

Which of several actions should each customer or outlet get, when every action
has a cost, some actions are limited, and the same action helps some units and
loses money on others?

This repository shows one way to answer that question:

| Folder | What it holds |
|---|---|
| [`method/`](method) | The method in words, formulas and pseudocode: shared cross-fitted nuisance models, doubly robust scores, one effect model per arm, policies under cost and capacity, honest evaluation, robustness checks and interpretation. |
| [`case_study_beer/`](case_study_beer) | A demo of the full workflow on synthetic beer retail data: the data rules and generator, an executed notebook with every output, and two PDF reports (business and technical). |
| [`case_study_loan_reminders/`](case_study_loan_reminders) | An exploratory study of two real randomized experiments (payment reminders at a consumer finance lender): which treatment wins, in which segments, and which to drop before the next pilot. Results are masked; a report and the masked numbers behind its figures. No code or data from that project. |

The modelling library used in the case study is private and not included. The
notebook is published already executed, so every table and figure can be read
without running anything. The synthetic data generator and its tests run on their
own.

**The beer case study is fully synthetic.** Names, costs, capacities and figures are
illustrative and do not describe any real company or market. The loan reminder case study
reports real experiments with every identifying detail removed and the results masked.

## Quick look

* Open the [project page](https://n2tp1311.github.io/AutoUpliftMultiArms/) for an overview of the method and both case studies.
* Read [`case_study_beer/case_study_business.pdf`](case_study_beer/case_study_business.pdf) for the decision and the money, and [`case_study_beer/case_study_technical.pdf`](case_study_beer/case_study_technical.pdf) for the full method and every check.
* Open [`case_study_beer/03_case_study_beer.ipynb`](case_study_beer/03_case_study_beer.ipynb) on GitHub to see the code steps and their outputs.
* Read [`method/README.md`](method/README.md) for how the approach works.

## Generate the data

```bash
python3.9 -m venv .venv
source .venv/bin/activate
pip install -r case_study_beer/requirements.txt
cd case_study_beer
pytest                                   # checks the generator against data_rules.md
python -m data_gen.generate --out data/  # writes the panel, the ground truth and metadata as CSV / JSON
```

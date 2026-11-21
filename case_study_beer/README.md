# Case study: which trade action for which beer outlet?

A worked case study of the `auto_uplift_multi_arms` library on **synthetic** beer
retail data. Every name, cost, capacity and figure is illustrative.

A beer brand can give each of 20,000 outlets one trade action per month
(`control`, `zalo_message`, `price_promo`, `display_kit`, `rep_visit`). The case
study estimates the effect of each action on cases ordered for every outlet-month,
chooses the action with the best net margin under regional capacity limits, and
compares that choice with the current business-as-usual rule and with the true
best action (known here because the data is simulated).

## What is in this folder

| File | What it is |
|---|---|
| [`case_study_business.pdf`](case_study_business.pdf) | Business report (8 pages): the decision, what the model is worth, what is shown and what is not, the pilot and how large a gain it can detect, risks, FAQ. |
| [`case_study_technical.pdf`](case_study_technical.pdf) | Technical report: data, ground truth, glossary, method, every result and robustness check, model against the truth, recommendation, limitations, how to reproduce, FAQ. |
| [`03_case_study_beer.ipynb`](03_case_study_beer.ipynb) | The executed notebook, with every output saved. Parts 1 to 14 match the sections of the technical report. |
| [`case_study_helpers.py`](case_study_helpers.py) | Helper code the notebook imports (data loading, split, exports, comparisons with the truth). Needs the private library. |
| [`data_rules.md`](data_rules.md) | Specification of the synthetic data: arms, costs, columns, assignment rule, true effects. |
| [`data_gen/`](data_gen) | The data generator (`make_beer_promo_data`) and a command-line entry point. |
| [`tests/`](tests) | Tests that the generated data follows `data_rules.md`. |
| [`outputs/`](outputs) | Figures, tables and `case_study_numbers.json` written by the notebook run; both reports are built from them. |
| [`report_src/`](report_src) | Report templates (`business.md.j2`, `technical.md.j2`) and the shared LaTeX preamble. |
| [`build_case_study.py`](build_case_study.py) | Runs the notebook (optional) and builds both PDFs. |
| [`case_study_review.md`](case_study_review.md) | The final cold-reader review of the two reports and the notebook. |
| [`requirements.txt`](requirements.txt) | Packages for the data generator, its tests and the report build. |

## Reproduce

The notebook uses a private modelling library (`auto_uplift_multi_arms`) that is
not part of this repository, so the notebook cannot be re-executed from here. It
is published already executed: every output, table and figure is saved in it and
in `outputs/`. What runs without the library:

```bash
python3.9 -m venv .venv            # at the repository root
source .venv/bin/activate
pip install -r case_study_beer/requirements.txt
cd case_study_beer
pytest                              # data generator tests
python -m data_gen.generate --out data/   # write the data as CSV / JSON
python build_case_study.py          # rebuild both PDFs from outputs/
```

The report build needs pandoc and tectonic on the PATH (macOS: `brew install pandoc tectonic`);
tectonic downloads the LaTeX packages it needs on the first run.

With the library installed (owner's machine), `python build_case_study.py --execute`
reruns the notebook and then rebuilds both PDFs. The full run takes about 22 minutes
on a laptop.

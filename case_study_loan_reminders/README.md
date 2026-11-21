# Case study: payment reminders before the due date (real experiment, masked)

Two randomized experiments at a consumer finance lender tested different reminder plans before a loan
instalment falls due: one for customers facing their first instalment (4 arms), one for customers on
their second instalment or later (5 arms). Each was modelled with the multi-arm uplift method described
in [`../method/README.md`](../method/README.md), in a full version with every arm and a reduced version
with only the leading arms.

The study is exploratory: it looks for the superior treatment, the segments where it is superior, and
the treatments that lose to another in every segment. Those are removed so the next pilot stays clean
(9 plans narrowed to 3).

**The results are real but masked.** The company, country, currency, dates, internal names, customer
counts and absolute payment rates are removed. Arms, features and products have generic names, and
effects are rounded to 0.1 percentage points. No code, notebook or data from the original project is
included. The full list of changes is at the end of the report.

## Main findings

- **First instalment:** one plan (late schedule, tracked message first) is best for every product and
  customer group: +1.0 pp on-time payment over the old control, +0.7 pp over the next best arm.
- **Second instalment onward:** a second reminder adds +1.3 to +1.4 pp; email adds nothing once there are
  two reminders.
- **Personalisation:** in all four model versions, assigning arms per customer is within 0.1 pp of giving
  everyone the best single arm. The models are useful for describing who responds, not for targeting.

## Files

| File | What it is |
|---|---|
| [`case_study_loan_reminders.pdf`](case_study_loan_reminders.pdf) | The report (business summary, method, results, lessons). |
| [`report.md`](report.md) | The same report in Markdown. |
| [`masked_results.json`](masked_results.json) | Every number shown in the figures, after masking. |
| [`make_figures.py`](make_figures.py) | Draws `figures/*.png` from `masked_results.json`. |
| [`build_report.py`](build_report.py) | Builds the PDF from `report.md` (needs pandoc and tectonic). |

```bash
python make_figures.py && python build_report.py
```

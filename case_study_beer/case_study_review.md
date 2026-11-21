# Review record: beer trade case study

This file records how the three published outputs of the case study were reviewed before release:

| Output | File |
|---|---|
| Business report | `case_study_business.pdf` |
| Technical report | `case_study_technical.pdf` |
| Executed notebook | `03_case_study_beer.ipynb` |

## Method

Each round used three independent cold readers, one per output, who had not seen the earlier rounds:

- **Business report:** read as a trade marketing manager with no statistics background.
- **Technical report:** read as a data scientist who knows causal inference and uplift modelling.
- **Notebook:** read as a data scientist on GitHub. The notebook was read, not run, because it needs the private
  library.

Every reader checked the numbers against `outputs/case_study_numbers.json` and `outputs/tables/*.csv`, and
against the other two outputs. Findings were rated high, medium or low. An output counts as clean when a round
finds no high or medium issue in it. The final verdicts: notebook and technical report clean in round 6,
business report clean in round 8. The low findings of those clean rounds were fixed too. Every finding was either fixed or recorded below as accepted. After each
round the notebook was rerun where code changed, and both PDFs were rebuilt from the saved outputs.

## Rounds

| Round | Business report | Technical report | Notebook |
|---|---|---|---|
| 1 to 3 | not clean | not clean | not clean |
| 4 | not clean (3 medium) | not clean (2 medium) | not clean (3 medium) |
| 5 | not clean (3 medium) | not clean (2 medium) | not clean (1 medium) |
| 6 | not clean (1 medium) | **clean** (8 low, fixed) | **clean** (5 low, fixed) |
| 7 | not clean (1 medium) | | |
| 8 | **clean** (5 low, fixed) | | |

## Main corrections

These changed what a reader would conclude, not just the wording.

- **Spending.** An early draft said the model's gains came with lower spending on actions. The opposite is
  true. Without limits the model spends about 184,000 VND per outlet-month against 94,000 for BAU. Within the
  limits, the model with the promo rule spends about 109,000 against 63,000. The gains are net of this. The
  extra pilot budget is now stated in the recommendation itself: about 1,242 million VND over 3 months.
- **Pilot power.** A 3-month pilot with the adjusted analysis detects the simulated gain only about 60 times in
  100. The reports used to suggest a clearer answer. They now state the power, the switch rule and its error
  rates: it switches about 9 times in 10 if the gain is real and about 2 times in 10 if it is not. The decision
  analysis is now the adjusted comparison that the power table assumes, not DR scoring.
- **Learner choice.** The tuner chose `dr`, but a paired out-of-fold test favours `x` significantly
  (46,844 VND, CI 2,436 to 87,706). Both reports and the notebook now say the choice is not settled.
- **Illustrative choices.** Two choices were made with the simulated truth in view: the brand share cutoff
  (60%) and the fill order (rep visits before display kits). Both are now disclosed as illustrative. The fill
  order was also checked without the truth, and the difference is not significant (-11,526 VND, CI -26,138
  to 2,271).
- **Why the capped comparison misses.** The miss was first blamed on the capped policy's promos. The report's
  own numbers contradict that. The cause is a shift in the "all control" baseline that affects policies
  treating almost every row. The reports now say so.
- **Claims without evidence.** The reports no longer state effects in supermarkets, where rep visits and
  display kits were never tried, as facts. Statements about A/A, overfit and clip checks now match what the
  outputs actually show.
- **A failed check now reaches business readers.** In bars and restaurants the model does significantly worse
  than giving every outlet a rep visit (-82,000 and -60,000 VND per outlet-month). The technical report
  showed this all along; the business report now does too. The cost of the limits is also stated against both
  versions of the capped model, with and without the promo rule (about 110,000 and 90,000 VND per outlet-month).
- **Definitions.** Several table columns were undefined or wrongly defined (for example `ess_share`), and the
  units of the ranking scores were missing. These were corrected and legends added.

## Known and accepted

These were raised and deliberately left as they are.

- **Library plot cosmetics.** Long decimals in some plot titles and an overlapping subtitle. These are logged in
  `library/docs/known_issues.md` and were not changed, because the library is out of scope for this release.
- **Repeated weak-overlap warnings.** These are expected: modern trade never received display kits or rep
  visits. They are explained in notebook Part 5.
- **The notebook cannot be rerun without the private library.** It is published executed.
- **Layout.** Rotated headers on landscape tables, unnumbered short tables, and a half-empty page before a
  figure in the business report.
- **`zalo_message` / "App message".** This may name a real messaging product. The owner decides.
- **The cost of the random 10% pilot group is not estimated.** The reports say so.

## Checks run on the final version

- **Leak scan** of the repository against the reference material: only generic Python and library API names
  match.
- **Text scans:** no company names and no absolute paths in the case study, the method note or the PDFs.
- **Style:** no em dashes in the PDFs, the notebook or the templates.
- **Tests:** `pytest` passes (67 passed, 1 skipped).
- **Build:** both PDFs build with no LaTeX warnings.

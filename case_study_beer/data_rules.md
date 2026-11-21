# Synthetic data rules: beer retail demo

This document is the specification of the data used in the case study. The code
that implements it is `data_gen/beer_promo.py` (beer panel) on top of
`data_gen/synthetic.py` (shared building blocks and a generic panel).
`tests/test_beer_data.py` checks that the generated data follows these rules.

**Everything here is synthetic and illustrative.** No real brand, company, outlet,
cost, capacity or market figure is used. The numbers were chosen so that the
business story is easy to follow, not to match any real market.

Generate the default data:

```bash
cd case_study_beer
python -m data_gen.generate --out data/
```

```python
from data_gen import make_beer_promo_data
df, truth, meta = make_beer_promo_data(n_outlets=20_000, n_months=6, randomised_share=0.7,
                                       outcome="volume", random_state=42)
```

The same arguments always give the same data (fixed `random_state`, NumPy `default_rng`).

## 1. Decision and arms

A beer brand sells to retail outlets. Every month it chooses, per outlet, one
trade action (arm). Each outlet gets exactly one arm per month.

| Code | Label | What it physically is | Cost per outlet-month (VND) | Capacity |
|---|---|---|---|---|
| 0 | `control` | No action. | 0 | none |
| 1 | `zalo_message` | Digital reminder on a messaging app with a link to order. | 2,000 | none |
| 2 | `price_promo` | Buy 10 cases, get 1 free, for orders this month. | 120,000 | none |
| 3 | `display_kit` | Cooler sticker, standee and menu boards installed at the outlet. | 400,000 | 6% of the region's outlets per month |
| 4 | `rep_visit` | Sales rep visit with merchandising and order taking. | 150,000 | 8% of the region's outlets per month |

* The promo cost is a fixed average per outlet. In reality it scales with the
  volume ordered. The policy takes one fixed cost per arm, so this is a
  simplification of the demo, not of the method.
* The caps are a planning constraint for the months after the data. The simulated
  history ignores them: the BAU rule and the trade test assign arms without any cap,
  so BAU gives more display kits and rep visits than the caps allow. Capped policies
  are therefore compared with BAU cut down to the caps.
* Capacity per region and month is `share x number of outlets in the region`,
  rounded to the nearest 10 (minimum 10). With the default data:

| Region | display_kit | rep_visit |
|---|---|---|
| North | 360 | 480 |
| Central | 180 | 240 |
| Highlands | 120 | 160 |
| South | 360 | 490 |
| Mekong | 180 | 240 |

* Money: `value_per_outcome` = 60,000 VND gross margin per case
  (`outcome="volume"`), or 600,000 VND per outlet that places an order
  (`outcome="reorder"`). Net value of an arm for a row is
  `value_per_outcome x tau_k - cost_k`, in VND per outlet-month.

## 2. Grain and period

* One row = one outlet in one month. Default: 20,000 outlets x 6 months = 120,000 rows.
* Months: January to June 2026 (`month` holds the first day of the month).
  February 2026 is the Tet month (lunar new year). May and June start the rainy
  season in most regions.
* Three extra months before the panel (October to December 2025) are simulated
  only to build the history features. They are not returned, and no trade action
  happens in them.
* Outlets repeat every month and can get a different arm each month. This is why
  the case study uses `group_col="outlet_id"` for the folds.

## 3. Columns

| Column | Type | Meaning | Unit | Missing |
|---|---|---|---|---|
| `outlet_id` | int | Outlet identifier, 1..n_outlets. | id | no |
| `month` | date | First day of the month. | date | no |
| `region` | category (5) | North, Central, Highlands, South, Mekong. Shares 30/15/10/30/15%. | | no |
| `province` | category (20) | Province code inside the region (`north_01` ... ). North 5, Central 4, Highlands 3, South 5, Mekong 3. | | no |
| `channel` | category (5) | `on_trade_bar` 18%, `on_trade_restaurant` 22%, `off_trade_grocery` 32%, `off_trade_minimart` 18%, `modern_trade` 10%. | | no |
| `outlet_size` | category (3) | small / medium / large; the mix depends on the channel (section 4). | | no |
| `n_coolers` | int | Coolers at the outlet, 0 to 6. | count | no |
| `seating_capacity` | float | Seats; on-trade only. | seats | yes, for every off-trade and modern trade outlet (about 60%) |
| `vol_last_1m` | float | Cases ordered last month. | cases | no |
| `vol_avg_3m` | float | Mean cases ordered over the last 3 months. | cases | no |
| `vol_trend_3m` | float | `(vol last month - vol 3 months ago) / (vol_avg_3m + 1)`, between about -3 and 3. Negative = falling. | ratio | no |
| `days_since_last_order` | int | Days from the last order to the start of the month. | days | no |
| `order_frequency_3m` | int | Number of orders over the last 3 months. | orders | no |
| `brand_share_est` | float | Estimated share of our brand in the outlet's beer volume: true share + noise (sd 0.05), clipped to 0.01..0.99. | share 0..1 | yes, 8% (outlet not surveyed) |
| `competitor_promo_active` | 0/1 | A competitor runs a promo at the outlet this month (20%; 35% in the Tet month). | flag | no |
| `tet_month` | 0/1 | Month of the lunar new year. | flag | no |
| `rainy_season` | 0/1 | Rainy season in the outlet's region this month. | flag | no |
| `avg_temperature` | float | Monthly mean temperature of the region, plus a province offset and noise. | degrees C | no |
| `distance_to_distributor_km` | float | Distance to the serving distributor. | km | no |
| `credit_days` | int | Payment terms: 0, 7, 14 or 30 days (large outlets at least 14). | days | no |
| `legacy_program_flag` | 0/1 | Near-constant column: 1 for 0.4% of outlets. Planted for `FeatureFilter`. | flag | no |
| `vol_avg_3m_reported` | float | Near-duplicate of `vol_avg_3m` (x (1 + noise sd 2%)). Planted for `FeatureFilter`. | cases | no |
| `noise_feature` | float | Pure noise, N(0, 1), unrelated to anything. Planted for the noise floor. | | no |
| `arm` | int | Arm the outlet received this month (0..4). | code | no |
| `rule_arm` | int | Arm the business-as-usual (BAU) rule would give (section 6). | code | no |
| `in_trade_test` | 0/1 | 1 if the outlet-month was in the randomised trade test. Design information, not a feature. | flag | no |
| `cases_ordered` | float | Outcome for `outcome="volume"`: cases ordered this month, 1 decimal; 0 when no order. | cases | no |
| `reordered` | 0/1 | Outcome for `outcome="reorder"`: 1 if the outlet ordered this month. | flag | no |

`arm`, `rule_arm`, `in_trade_test`, `cases_ordered` and `reordered` must not be
used as features.

## 4. Outlet attributes (fixed over the months)

* Size mix P(small, medium, large) by channel: bar 45/35/20%, restaurant 40/40/20%,
  grocery 65/28/7%, minimart 40/45/15%, modern trade 10/40/50%.
* Coolers: Poisson with mean `0.4 + 0.9 x size_index` for off-trade and modern
  trade, `0.3 + 0.5 x size_index` for on-trade (size_index 0 small, 1 medium,
  2 large), capped at 6.
* Seating (on-trade): 20 / 50 / 120 seats for small / medium / large, times a lognormal factor (sd 0.3).
* Base cases per order month (median, small / medium / large): bar 25/45/90,
  restaurant 20/40/80, grocery 12/22/40, minimart 15/28/50, modern trade 60/110/200;
  times an outlet factor, lognormal with sd 0.35.
* Latent growth per month (log scale): N(0, 0.06) per outlet. Not observed; the
  model sees it only through `vol_trend_3m`.
* True brand share: Beta(4, 4) per outlet (mean 0.5). The model sees only `brand_share_est`.

## 5. Outcome without any action (control)

Every month an outlet orders with probability `q0` and, when it orders, buys a
number of cases with mean `m0`.

```
q0 = sigmoid( base_logit(channel) + outlet noise N(0, 0.6)
              - 0.015 x (days_since_last_order - 15)
              - 0.02  x (distance_to_distributor_km - 8)
              + 0.4   x tet_month )
base_logit: bar 2.0, restaurant 1.8, grocery 1.4, minimart 1.7, modern trade 2.8

m0 = base_cases x exp(growth x month_index)
     x season x (0.92 if competitor_promo_active) x (1 + 0.004 x credit_days)
season = 1.45 in the Tet month
         x exp(0.02 x (avg_temperature - 25))
         x rainy factor (0.90 for on-trade, 0.97 otherwise, in the rainy season)
```

Observed outcome under the arm actually received (`q`, `m` include the arm's effect, section 7):

```
ordered       ~ Bernoulli(q)
cases_ordered = Gamma(shape 8, mean m) if ordered else 0      (right-skewed, rounded to 0.1)
reordered     = ordered
orders in the month = 1 + Poisson(m / 20) if ordered else 0   (feeds order_frequency_3m)
```

History features of month t are computed from the simulated outcomes of months
t-1, t-2, t-3, so a trade action in one month changes the history seen in the
next months. The effect of an action itself lasts only the month it is given.

## 6. How arms were assigned (mixed data)

Each outlet-month is in the randomised trade test with probability
`randomised_share` (default 0.7).

* **Trade test rows** get an arm uniformly at random among the arms allowed for
  the outlet. Modern trade outlets (key accounts, central buying) can only get
  `control`, `zalo_message` or `price_promo` in the test: display kits and rep
  visits are not allowed in their stores. So for modern trade the propensity of
  `display_kit` and `rep_visit` is exactly 0. This is the planted lack of
  overlap: the model has no evidence there and the policy's `propensity_floor`
  must keep those arms out.
* **Other rows** follow the business-as-usual (BAU) rule, using observed features only.
  The first line that matches decides:

| # | Condition | BAU arm |
|---|---|---|
| 1 | modern trade and `brand_share_est` < 0.4 | `price_promo` |
| 2 | modern trade (other) | `control` |
| 3 | on-trade and size large | `rep_visit` |
| 4 | on-trade and size medium and `vol_trend_3m` < -0.15 | `rep_visit` |
| 5 | off-trade and `n_coolers` >= 2 | `display_kit` |
| 6 | `vol_trend_3m` < -0.2 | `price_promo` |
| 7 | `days_since_last_order` > 45 | `zalo_message` |
| 8 | otherwise | `control` |

(A missing `brand_share_est` counts as not low.) The rule's arm is stored in
`rule_arm` for every row, also for test rows, so the BAU policy can be evaluated.
The BAU rule does not respect the capacities of section 1.

True propensity of arm k for a row:
`e_k = randomised_share x 1/(number of allowed arms) x [k allowed] + (1 - randomised_share) x [k == rule_arm]`.
With the default 0.7, every allowed arm has at least 0.14 (0.233 in modern trade).
Because BAU rows follow the rule, the arm is confounded with size, channel,
trend, coolers and recency: a naive comparison of outcomes by arm is biased.

## 7. True effects (ground truth)

Each arm changes the order probability by `dq_k` and the cases per order by
`dm_k`: `q_k = q0 + dq_k`, `m_k = m0 + dm_k`. The true effects are exact
expectations:

* `outcome="volume"`: `tau_k = q_k x m_k - q0 x m0` (cases per outlet-month)
* `outcome="reorder"`: `tau_k = q_k - q0` (probability of ordering)

Notation: `size_f` = 0 small, 0.5 medium, 1 large; `falling` = min(1, max(0, -vol_trend_3m));
`share` = true brand share; `comp` = competitor_promo_active; `days` = days_since_last_order.

| Arm | Effect on order probability `dq` | Effect on cases per order `dm` | Business meaning |
|---|---|---|---|
| `zalo_message` | `0.20 x (1 - q0) x sigmoid((days - 40) / 8)` | 0 | Brings lapsed outlets back; close to zero for regular buyers. Cheap, so it often wins on net value for small outlets. |
| `price_promo` | `0.05 x (1 - q0)` | `12 x (1 - share)^2 x (1 + 0.5 x comp) x (0.7 + 0.7 x size_f)` | Wins volume from competitors where our share is low, more when a competitor promo runs. At high share it mostly subsidises cases that would be bought anyway: its net value turns negative above a share of about 0.65. |
| `display_kit` | 0 | off-trade with >= 1 cooler: `(9 + 4.5 x min(n_coolers, 3)) x (0.8 + 0.4 x size_f)`; off-trade without cooler 0.75; restaurant 1.5; bar 0.45; modern trade 2.25 | Works in groceries and minimarts that have coolers; little effect in bars. |
| `rep_visit` | `0.15 x (1 - q0)`, 0 in modern trade | `base(channel) x (1 + 2 x size_f) x (1 + 2.5 x falling)`; base: bar 3.0, restaurant 2.7, off-trade 1.5, modern trade 0.3 | Best for large on-trade outlets with a falling trend (size x trend interaction); weak in modern trade, where store staff cannot order. |

* In the Tet month every `dq` and `dm` is multiplied by 1.4.
* `dq` is capped so that `q_k` stays below 1.
* The effect structure uses the true brand share and the true recency, while the
  model sees `brand_share_est` (noisy, 8% missing). A perfect model is therefore
  not possible: this is intended.

`truth` (second output) holds, per row: `tau_1..tau_4`, `mu_0..mu_4` (true
expected outcome per arm), `e_0..e_4` (true propensity), `net_value_0..net_value_4`
(VND, with the costs and value of section 1) and `true_best_arm` (arm with the
highest true net value; control when none beats it).

Real projects do not have this table. It exists here only so the case study can
judge the model against the truth.

## 8. Generic generator

`make_multi_arm_data` in `data_gen/synthetic.py` builds a smaller generic panel
with generic names (`customer_id`, `date`, `x1`..`xN`, `cat_high`, `segment`,
`arm`, `rule_arm`, `y`), a binary or continuous outcome, a `randomised_share` and
heterogeneous effects (arm 1 grows with `x1` and is harmful for `x1 < -1`; arm 2
has opposite signs by segment; arms 3+ react to `x3` in one segment). Its
docstring lists the exact formulas. It is used for quick experiments; the case
study uses the beer panel.

## 9. What the generator guarantees (checked by tests)

* Deterministic for a given `random_state`; default size runs in a few seconds.
* The true best arm differs by channel.
* `price_promo` has negative mean true net value for outlets with high brand share.
* Modern trade has zero propensity for `display_kit` and `rep_visit`.
* Every arm is harmful (negative net value) for some rows, and helpful for others.
* `rep_visit` effect grows with size and with a falling trend.
* Effects are larger in the Tet month.
* `FeatureFilter` has a near-constant column, a near-duplicate and a pure-noise column to find.

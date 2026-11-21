"""Beer retail demo panel: one row = one outlet in one month.

A beer brand decides every month, per retail outlet, one trade action (arm):

    0 control        no action
    1 zalo_message   digital reminder with an order link
    2 price_promo    buy 10 cases get 1 free
    3 display_kit    cooler sticker, standee, menu boards       (capacity per region per month)
    4 rep_visit      sales rep visit with merchandising         (capacity per region per month)

All names, numbers, costs and capacities are illustrative. The data is synthetic.
The full rules (features, effects, assignment) are written in ``data_rules.md``;
every constant below is listed there, and ``tests/test_beer_data.py`` checks that
the generated data follows them.

The generator is built on the generic building blocks in ``synthetic.py``:
``mix_assignment`` for the trade test mixed with the business-as-usual rule and
``true_best_arm`` for the true best arm on net value.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .synthetic import mix_assignment, true_best_arm

__all__ = ["make_beer_promo_data", "ARMS", "ARM_LABELS", "COST_VND", "VALUE_PER_CASE_VND", "VALUE_PER_REORDER_VND",
           "CHANNELS", "REGIONS", "true_effects"]

# ── arms, money ─────────────────────────────────────────────────────────────
ARMS = {
    0: ("control", "No trade action this month."),
    1: ("zalo_message", "Digital reminder on a messaging app with a link to order."),
    2: ("price_promo", "Buy 10 cases, get 1 free, for orders this month."),
    3: ("display_kit", "Cooler sticker, standee and menu boards installed at the outlet."),
    4: ("rep_visit", "Sales rep visit with merchandising and order taking."),
}
ARM_LABELS = {k: v[0] for k, v in ARMS.items()}
COST_VND = {0: 0, 1: 2_000, 2: 120_000, 3: 400_000, 4: 150_000}  # per outlet per month
VALUE_PER_CASE_VND = 60_000        # gross margin per case (outcome="volume")
VALUE_PER_REORDER_VND = 600_000    # value of one outlet placing an order (outcome="reorder")
CAPACITY_SHARE = {3: 0.06, 4: 0.08}  # capacity per region-month = share x outlets in the region, rounded to 10

# ── outlets ─────────────────────────────────────────────────────────────────
REGIONS = {"North": (0.30, 5), "Central": (0.15, 4), "Highlands": (0.10, 3), "South": (0.30, 5), "Mekong": (0.15, 3)}
CHANNELS = {"on_trade_bar": 0.18, "on_trade_restaurant": 0.22, "off_trade_grocery": 0.32,
            "off_trade_minimart": 0.18, "modern_trade": 0.10}
ON_TRADE = ("on_trade_bar", "on_trade_restaurant")
OFF_TRADE = ("off_trade_grocery", "off_trade_minimart")
SIZES = ("small", "medium", "large")
SIZE_PROBS = {  # P(small, medium, large) by channel
    "on_trade_bar": (0.45, 0.35, 0.20), "on_trade_restaurant": (0.40, 0.40, 0.20),
    "off_trade_grocery": (0.65, 0.28, 0.07), "off_trade_minimart": (0.40, 0.45, 0.15),
    "modern_trade": (0.10, 0.40, 0.50),
}
BASE_CASES = {  # median cases per order month for (small, medium, large)
    "on_trade_bar": (25, 45, 90), "on_trade_restaurant": (20, 40, 80), "off_trade_grocery": (12, 22, 40),
    "off_trade_minimart": (15, 28, 50), "modern_trade": (60, 110, 200),
}
BASE_ORDER_LOGIT = {"on_trade_bar": 2.0, "on_trade_restaurant": 1.8, "off_trade_grocery": 1.4,
                    "off_trade_minimart": 1.7, "modern_trade": 2.8}
# monthly mean temperature by region, months 1..12 (illustrative)
TEMP = {
    "North":     (17, 18, 21, 25, 28, 30, 30, 29, 28, 25, 22, 19),
    "Central":   (21, 22, 24, 27, 29, 30, 30, 30, 28, 26, 24, 22),
    "Highlands": (20, 21, 23, 24, 24, 23, 23, 23, 23, 22, 21, 20),
    "South":     (27, 28, 29, 30, 29, 28, 28, 28, 28, 28, 27, 27),
    "Mekong":    (26, 27, 28, 29, 29, 28, 28, 28, 28, 28, 27, 26),
}
RAINY_MONTHS = {"North": range(5, 10), "Central": range(9, 13), "Highlands": range(5, 12),
                "South": range(5, 12), "Mekong": range(5, 12)}
TET_MONTH = {2025: 1, 2026: 2, 2027: 2, 2028: 1}  # calendar month of the lunar new year (Tet)

N_PRE_MONTHS = 3       # simulated before the panel to build the history features; not returned
TET_EFFECT_MULT = 1.4  # every arm effect is 40% larger in the Tet month
GAMMA_SHAPE = 8.0      # cases | order ~ Gamma(shape, mean m): right-skewed, CV about 0.35


def _logit(p):
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def true_effects(ctx: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Effect of each arm on the order probability (dq) and on cases per order (dm).

    ``ctx`` holds, per row: channel, size_f (0 small, 0.5 medium, 1 large), n_coolers,
    falling (max(0, -vol_trend_3m) capped at 1), share (true brand share),
    competitor_promo (0/1), days_since_last_order, tet (0/1), q0 (control order probability).

    Returns
    -------
    dq : array (n, 5)   order probability under arm k minus under control (column 0 is 0)
    dm : array (n, 5)   cases per order under arm k minus under control (column 0 is 0)
    The arm then has q_k = q0 + dq_k and m_k = m0 + dm_k.
    """
    ch, size_f, q0 = ctx["channel"], ctx["size_f"], ctx["q0"]
    n = len(ch)
    off = np.isin(ch, OFF_TRADE)
    mt = ch == "modern_trade"
    bar = ch == "on_trade_bar"
    rest = ch == "on_trade_restaurant"
    tet = np.where(ctx["tet"] == 1, TET_EFFECT_MULT, 1.0)
    dq = np.zeros((n, 5))
    dm = np.zeros((n, 5))

    # 1 zalo_message: reminds lapsed outlets to order; close to zero for regular buyers
    dq[:, 1] = 0.20 * (1 - q0) * _sigmoid((ctx["days_since_last_order"] - 40.0) / 8.0)

    # 2 price_promo: wins volume from competitors where our share is low; for loyal,
    # high-share outlets it mostly subsidises cases that would be bought anyway
    size_promo = 0.7 + 0.7 * size_f  # 0.7 small, 1.05 medium, 1.4 large
    dm[:, 2] = 12.0 * (1 - ctx["share"]) ** 2 * (1 + 0.5 * ctx["competitor_promo"]) * size_promo
    dq[:, 2] = 0.05 * (1 - q0)

    # 3 display_kit: works in off-trade outlets that have coolers; little effect in bars
    coolers = np.minimum(ctx["n_coolers"], 3)
    dm[:, 3] = np.select(
        [off & (coolers >= 1), off, rest, bar, mt],
        [(9.0 + 4.5 * coolers) * (0.8 + 0.4 * size_f), 0.75, 1.5, 0.45, 2.25], 0.0)

    # 4 rep_visit: best for large on-trade outlets with a falling trend (size x trend
    # interaction); weak in modern trade (central buying, store staff cannot order)
    base_rep = np.select([bar, rest, off, mt], [3.0, 2.7, 1.5, 0.3], 0.0)
    dm[:, 4] = base_rep * (1 + 2 * size_f) * (1 + 2.5 * ctx["falling"])
    dq[:, 4] = np.where(mt, 0.0, 0.15 * (1 - q0))

    dq *= tet[:, None]
    dm *= tet[:, None]
    dq = np.minimum(dq, (1 - q0)[:, None] * 0.95)  # keep q_k below 1
    return dq, dm


def _bau_rule(channel, size, n_coolers, trend, days, share_est) -> np.ndarray:
    """Business-as-usual rule: first matching line wins (see data_rules.md)."""
    on = np.isin(channel, ON_TRADE)
    off = np.isin(channel, OFF_TRADE)
    mt = channel == "modern_trade"
    low_share = np.nan_to_num(share_est, nan=1.0) < 0.4
    conds = [
        mt & low_share,                                 # central promo for key accounts with low share
        mt,                                             # other key accounts: no action
        on & (size == "large"),                         # reps visit the largest on-trade outlets
        on & (size == "medium") & (trend < -0.15),      # and medium on-trade outlets that are falling
        off & (n_coolers >= 2),                         # display kits go to off-trade outlets with coolers
        trend < -0.2,                                   # promos go to outlets with a falling trend
        days > 45,                                      # reminders go to lapsed outlets
    ]
    return np.select(conds, [2, 0, 4, 4, 3, 2, 1], 0).astype(int)


def make_beer_promo_data(
    n_outlets: int = 20_000,
    n_months: int = 6,
    randomised_share: float = 0.7,
    outcome: str = "volume",
    random_state: int = 42,
    start_month: str = "2026-01",
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Synthetic beer retail panel with a mixed trade test and known true effects.

    Parameters
    ----------
    n_outlets : int, default 20_000
    n_months : int, default 6
        Months in the panel. The default (January to June 2026) covers the Tet
        month (February) and the start of the rainy season (May, June).
    randomised_share : float, default 0.7
        Share of outlet-months in the randomised trade test. The rest follow the
        business-as-usual (BAU) rule. Inside the test, modern trade outlets can only
        get control, zalo_message or price_promo.
    outcome : {"volume", "reorder"}, default "volume"
        "volume": ``cases_ordered`` (cases in the month, continuous, zeros allowed).
        "reorder": ``reordered`` (1 if the outlet ordered at least once in the month).
        Both columns are always in the frame; this picks the truth and the money.
    random_state : int, default 42
    start_month : str, default "2026-01"

    Returns
    -------
    df : DataFrame
        One row per outlet-month: ids, features, ``arm``, ``rule_arm`` (BAU arm),
        ``in_trade_test``, ``cases_ordered``, ``reordered``.
    truth : DataFrame, same index as df
        tau_1..tau_4 (true effect on the chosen outcome), mu_0..mu_4 (true expected
        outcome per arm), e_0..e_4 (true propensity), net_value_0..net_value_4 (VND),
        true_best_arm.
    meta : dict
        arm_labels, arm_descriptions, cost, value_per_outcome, capacity (per region and
        month for display_kit and rep_visit), capacity_by, period, outcome_col,
        outcome_unit, months, tet_month, and a few counts.

    Examples
    --------
    >>> df, truth, meta = make_beer_promo_data(n_outlets=2_000, n_months=3, random_state=0)
    >>> meta["arm_labels"][4]
    'rep_visit'
    """
    if outcome not in ("volume", "reorder"):
        raise ValueError(f"outcome must be 'volume' or 'reorder'; got '{outcome}'")
    if n_months < 1:
        raise ValueError("n_months must be at least 1")
    rng = np.random.default_rng(random_state)
    n = n_outlets

    # ── static outlet attributes ────────────────────────────────────────────
    reg_names = list(REGIONS)
    region = rng.choice(reg_names, n, p=[REGIONS[r][0] for r in reg_names])
    province = np.empty(n, dtype=object)
    for r in reg_names:
        m = region == r
        province[m] = [f"{r.lower()}_{i:02d}" for i in rng.integers(1, REGIONS[r][1] + 1, m.sum())]
    ch_names = list(CHANNELS)
    channel = rng.choice(ch_names, n, p=[CHANNELS[c] for c in ch_names])
    size = np.empty(n, dtype=object)
    for c in ch_names:
        m = channel == c
        size[m] = rng.choice(SIZES, m.sum(), p=SIZE_PROBS[c])
    size_idx = np.select([size == "small", size == "medium"], [0, 1], 2)
    size_f = size_idx / 2.0
    on = np.isin(channel, ON_TRADE)
    off = np.isin(channel, OFF_TRADE)
    n_coolers = np.where(off | (channel == "modern_trade"), rng.poisson(0.4 + 0.9 * size_idx),
                         rng.poisson(0.3 + 0.5 * size_idx)).astype(int)
    n_coolers = np.minimum(n_coolers, 6)
    seating = np.where(on, np.round(np.array([20, 50, 120])[size_idx] * rng.lognormal(0, 0.3, n)), np.nan)
    base_cases = np.array([BASE_CASES[c][s] for c, s in zip(channel, size_idx)]) * rng.lognormal(0, 0.35, n)
    order_logit = np.array([BASE_ORDER_LOGIT[c] for c in channel]) + rng.normal(0, 0.6, n)
    trend = rng.normal(0, 0.06, n)                      # latent log growth per month
    share = rng.beta(4, 4, n)                           # true brand share of the outlet's beer volume
    share_est = np.clip(share + rng.normal(0, 0.05, n), 0.01, 0.99).round(2)
    share_est[rng.random(n) < 0.08] = np.nan            # not surveyed
    distance = np.round(rng.lognormal(np.log(8), 0.6, n), 1)
    credit_days = rng.choice([0, 7, 14, 30], n, p=[0.35, 0.30, 0.20, 0.15])
    credit_days = np.where(size == "large", np.maximum(credit_days, 14), credit_days)
    legacy_flag = (rng.random(n) < 0.004).astype(int)  # near-constant column
    prov_temp = {p: rng.normal(0, 0.8) for p in np.unique(province)}
    prov_temp_off = np.array([prov_temp[p] for p in province])

    # ── months ──────────────────────────────────────────────────────────────
    all_months = pd.period_range(pd.Period(start_month, "M") - N_PRE_MONTHS, periods=N_PRE_MONTHS + n_months, freq="M")
    V = np.zeros((len(all_months), n))       # cases ordered
    ORD = np.zeros((len(all_months), n))       # number of orders in the month
    last_order_day = np.full(n, -rng.integers(1, 30, n).astype(float))  # days before the first simulated month
    rows, truths = [], []
    capacity = {}
    for t, per in enumerate(all_months):
        mo, yr = per.month, per.year
        in_panel = t >= N_PRE_MONTHS
        tet = np.full(n, int(TET_MONTH.get(yr, 2) == mo))
        rainy_by_region = {r: int(mo in RAINY_MONTHS[r]) for r in reg_names}
        rainy = np.array([rainy_by_region[r] for r in region])
        temp = np.array([TEMP[r][mo - 1] for r in region]) + prov_temp_off + rng.normal(0, 0.5, n)
        comp = (rng.random(n) < np.where(tet == 1, 0.35, 0.20)).astype(int)
        month_start_day = 30.0 * t
        days_since = np.maximum(1.0, month_start_day - last_order_day).round()

        # history features (from the simulated months before t)
        if t >= 3:
            v1, v2, v3 = V[t - 1], V[t - 2], V[t - 3]
        else:  # pre-period warm-up: history from the latent level
            v1 = v2 = v3 = base_cases * _sigmoid(order_logit)
        vol_avg_3m = (v1 + v2 + v3) / 3.0
        vol_trend_3m = (v1 - v3) / (vol_avg_3m + 1.0)
        freq_3m = ORD[t - 3:t].sum(axis=0) if t >= 3 else np.full(n, 3.0)

        # control expected outcome
        season = (1.45 if tet[0] else 1.0) * np.exp(0.02 * (temp - 25.0)) * np.where(rainy == 1, np.where(on, 0.90, 0.97), 1.0)
        m0 = base_cases * np.exp(trend * (t - N_PRE_MONTHS)) * season * np.where(comp == 1, 0.92, 1.0) \
            * (1 + 0.004 * credit_days)
        q0 = _sigmoid(order_logit - 0.015 * (days_since - 15.0) - 0.02 * (distance - 8.0) + 0.4 * tet)

        if in_panel:
            ctx = dict(channel=channel, size_f=size_f, n_coolers=n_coolers,
                       falling=np.clip(-vol_trend_3m, 0, 1), share=share, competitor_promo=comp,
                       days_since_last_order=days_since, tet=tet, q0=q0)
            dq, dm = true_effects(ctx)
            q = q0[:, None] + dq
            m = m0[:, None] + dm
            rule = _bau_rule(channel, size, n_coolers, vol_trend_3m, days_since, share_est)
            allowed = np.ones((n, 5), bool)
            allowed[channel == "modern_trade", 3:] = False
            arm, randomised, e = mix_assignment(rule, 5, randomised_share, rng, allowed)
        else:
            q, m = q0[:, None], m0[:, None]
            arm = np.zeros(n, int)

        qa = q[np.arange(n), arm]
        ma = m[np.arange(n), arm]
        ordered = rng.random(n) < qa
        cases = np.where(ordered, rng.gamma(GAMMA_SHAPE, ma / GAMMA_SHAPE), 0.0).round(1)
        n_orders = np.where(ordered, 1 + rng.poisson(ma / 20.0), 0)
        V[t], ORD[t] = cases, n_orders
        order_day = month_start_day + rng.integers(0, 30, n)
        last_order_day = np.where(ordered, order_day, last_order_day)

        if not in_panel:
            continue
        month_ts = per.to_timestamp()
        frame = pd.DataFrame({
            "outlet_id": np.arange(1, n + 1), "month": month_ts, "region": region, "province": province,
            "channel": channel, "outlet_size": size, "n_coolers": n_coolers, "seating_capacity": seating,
            "vol_last_1m": v1.round(1), "vol_avg_3m": vol_avg_3m.round(1), "vol_trend_3m": vol_trend_3m.round(3),
            "days_since_last_order": days_since.astype(int), "order_frequency_3m": freq_3m.astype(int),
            "brand_share_est": share_est, "competitor_promo_active": comp, "tet_month": tet,
            "rainy_season": rainy, "avg_temperature": temp.round(1), "distance_to_distributor_km": distance,
            "credit_days": credit_days, "legacy_program_flag": legacy_flag,
            "vol_avg_3m_reported": (vol_avg_3m * (1 + rng.normal(0, 0.02, n))).round(1),
            "noise_feature": rng.normal(0, 1, n).round(3),
            "arm": arm, "rule_arm": rule, "in_trade_test": randomised.astype(int),
            "cases_ordered": cases, "reordered": ordered.astype(int),
        })
        if outcome == "volume":
            mu = q * m
            value = VALUE_PER_CASE_VND
        else:
            mu = q
            value = VALUE_PER_REORDER_VND
        tau = mu[:, 1:] - mu[:, [0]]
        best, net = true_best_arm(tau, COST_VND, value)
        tr = pd.DataFrame({f"tau_{k}": tau[:, k - 1] for k in range(1, 5)})
        for k in range(5):
            tr[f"mu_{k}"] = mu[:, k]
        for k in range(5):
            tr[f"e_{k}"] = e[:, k]
        for k in range(5):
            tr[f"net_value_{k}"] = net[:, k]
        tr["true_best_arm"] = best
        rows.append(frame)
        truths.append(tr)

    df = pd.concat(rows, ignore_index=True)
    truth = pd.concat(truths, ignore_index=True)
    for c in ("region", "province", "channel"):
        df[c] = df[c].astype(str)
    df["outlet_size"] = pd.Categorical(df["outlet_size"], categories=list(SIZES), ordered=True).astype(str)

    outlets_per_region = pd.Series(region).value_counts()
    for k, s in CAPACITY_SHARE.items():
        capacity[k] = {r: int(max(10, round(s * outlets_per_region.get(r, 0), -1))) for r in reg_names}
    months = sorted(df["month"].unique())
    meta = {
        "arm_labels": dict(ARM_LABELS),
        "arm_descriptions": {k: v[1] for k, v in ARMS.items()},
        "cost": dict(COST_VND),
        "value_per_outcome": VALUE_PER_CASE_VND if outcome == "volume" else VALUE_PER_REORDER_VND,
        "outcome": outcome,
        "outcome_col": "cases_ordered" if outcome == "volume" else "reordered",
        "outcome_unit": "cases per outlet-month" if outcome == "volume" else "share of outlets that ordered",
        "capacity": capacity,
        "capacity_by": "region",
        "period": "M",
        "date_col": "month",
        "id_col": "outlet_id",
        "months": [pd.Timestamp(m).strftime("%Y-%m") for m in months],
        "tet_month": [pd.Timestamp(m).strftime("%Y-%m") for m in months
                      if TET_MONTH.get(pd.Timestamp(m).year, 2) == pd.Timestamp(m).month],
        "n_outlets": n,
        "n_rows": len(df),
        "randomised_share_target": randomised_share,
        "randomised_share_realised": float(df["in_trade_test"].mean()),
        "random_state": random_state,
        "illustrative": "Synthetic data. All names, costs, capacities and figures are illustrative.",
    }
    return df, truth, meta

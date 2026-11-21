"""Helpers for the beer case study notebook (03_case_study_beer.ipynb).

The notebook keeps the story; the plumbing lives here: loading the synthetic data,
the outlet-level split, the data dictionary, exporting figures / tables / numbers
for the PDF, and the comparisons with the true effects (possible only because the
data is synthetic).

Every exported file goes to ``outputs/``:
    outputs/figures/<name>.png        figures embedded in the PDF
    outputs/tables/<name>.csv         tables embedded in the PDF
    outputs/case_study_numbers.json   every number quoted in the PDF text
    outputs/captions.json             caption and producing step of every figure and table
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                      # data_gen
from data_gen.beer_promo import make_beer_promo_data  # noqa: E402

import auto_uplift_multi_arms as aum  # noqa: E402

# ── settings of the run (quoted in the PDF method section) ──────────────────
RANDOM_STATE = 42          # data generator
SPLIT_SEED = 7             # outlet-level train / test split
TEST_SHARE = 0.30          # share of outlets held out for the test
N_FOLDS = 3                # cross-fitting folds (grouped by outlet)
BASE_PARAMS = {"n_estimators": 300, "learning_rate": 0.05, "early_stopping": True,
               "n_jobs": 5, "verbose": -1}
REFEREE_PARAMS = {"n_estimators": 300, "learning_rate": 0.05, "early_stopping": True,
                  "num_leaves": 15, "min_child_samples": 1000, "n_jobs": 5, "verbose": -1}
PROPENSITY_CLIP = 0.01     # evaluator: floor on e_k in the DR scores
PROPENSITY_FLOOR = 0.02    # policy: an arm with e_k below this is never chosen for the row

# columns that are not features: identifiers, design information, outcomes
NOT_FEATURES = ["outlet_id", "month", "arm", "rule_arm", "in_trade_test", "cases_ordered", "reordered"]
NUISANCE_ONLY = ["month_index"]   # time: helps the outcome and propensity models, not the effect

COLUMN_INFO = {  # meaning and unit of every column (same text as data_rules.md, section 3)
    "outlet_id": ("Outlet identifier.", "id"),
    "month": ("First day of the month.", "date"),
    "region": ("Region: North, Central, Highlands, South, Mekong.", ""),
    "province": ("Province code inside the region (20 levels).", ""),
    "channel": ("Outlet type: bar, restaurant, grocery, minimart, modern trade.", ""),
    "outlet_size": ("small / medium / large.", ""),
    "n_coolers": ("Coolers at the outlet.", "count"),
    "seating_capacity": ("Seats; on-trade only, missing elsewhere.", "seats"),
    "vol_last_1m": ("Cases ordered last month.", "cases"),
    "vol_avg_3m": ("Mean cases ordered over the last 3 months.", "cases"),
    "vol_trend_3m": ("(last month - 3 months ago) / (vol_avg_3m + 1). Negative = falling.", "ratio"),
    "days_since_last_order": ("Days from the last order to the start of the month.", "days"),
    "order_frequency_3m": ("Orders over the last 3 months.", "orders"),
    "brand_share_est": ("Estimated share of our brand in the outlet's beer volume (noisy).", "share 0-1"),
    "competitor_promo_active": ("A competitor runs a promo at the outlet this month.", "flag"),
    "tet_month": ("Month of the lunar new year (Tet).", "flag"),
    "rainy_season": ("Rainy season in the outlet's region this month.", "flag"),
    "avg_temperature": ("Monthly mean temperature.", "deg C"),
    "distance_to_distributor_km": ("Distance to the serving distributor.", "km"),
    "credit_days": ("Payment terms.", "days"),
    "legacy_program_flag": ("Near-constant flag (planted for the feature filter).", "flag"),
    "vol_avg_3m_reported": ("Near-copy of vol_avg_3m (planted for the feature filter).", "cases"),
    "noise_feature": ("Pure noise (planted for the noise floor).", ""),
    "arm": ("Arm the outlet received this month (0-4).", "code"),
    "rule_arm": ("Arm the business-as-usual rule would give.", "code"),
    "in_trade_test": ("1 if the row was in the randomised trade test.", "flag"),
    "cases_ordered": ("OUTCOME: cases ordered this month (0 if no order).", "cases"),
    "reordered": ("1 if the outlet ordered this month (not used here).", "flag"),
    "month_index": ("Months since January 2026 (0-5), added for the nuisance models.", "months"),
}


# ── data ────────────────────────────────────────────────────────────────────
def load_data(random_state: int = RANDOM_STATE):
    """The default beer panel, its ground truth and metadata, plus month_index."""
    df, truth, meta = make_beer_promo_data(n_outlets=20_000, n_months=6, randomised_share=0.7,
                                           outcome="volume", random_state=random_state)
    df["month_index"] = (df["month"].dt.year - 2026) * 12 + df["month"].dt.month - 1
    return df, truth, meta


def features_of(df: pd.DataFrame):
    return [c for c in df.columns if c not in NOT_FEATURES]


def split_outlets(df: pd.DataFrame, test_share: float = TEST_SHARE, seed: int = SPLIT_SEED) -> np.ndarray:
    """Boolean mask of test rows: whole outlets (all their months) go to one side."""
    ids = df["outlet_id"].unique()
    rng = np.random.default_rng(seed)
    test_ids = rng.choice(ids, int(round(test_share * len(ids))), replace=False)
    return df["outlet_id"].isin(test_ids).to_numpy()


def make_dataset(d: pd.DataFrame, meta: Dict, features, outcome: str = "cases_ordered") -> aum.UpliftDataset:
    return aum.UpliftDataset(d.reset_index(drop=True), treatment="arm", outcome=outcome, features=list(features),
                             treatment_labels=meta["arm_labels"], date_col="month", rule_arm_col="rule_arm",
                             id_cols=["outlet_id"], suppress_warnings=True)


def data_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    """Every column: name, type, meaning, unit, missing rate, example value."""
    rows = []
    ex = df.iloc[0]
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            typ, example = "date", ex[c].strftime("%Y-%m-%d")
        elif s.dtype == object:
            typ, example = f"category ({s.nunique()})", ex[c]
        elif set(s.dropna().unique()) <= {0, 1}:
            typ, example = "0/1", int(ex[c])
        elif pd.api.types.is_integer_dtype(s):
            typ, example = "int", int(ex[c])
        else:
            typ, example = "float", ("missing" if pd.isna(ex[c]) else round(float(ex[c]), 2))
        meaning, unit = COLUMN_INFO.get(c, ("", ""))
        rows.append({"column": c, "type": typ, "meaning": meaning, "unit": unit,
                     "missing_%": round(100 * s.isna().mean(), 1), "example": example})
    return pd.DataFrame(rows)


def rows_per_arm(df: pd.DataFrame, meta: Dict) -> pd.DataFrame:
    g = df.groupby("arm")
    out = pd.DataFrame({
        "arm": g.size().index,
        "label": [meta["arm_labels"][a] for a in g.size().index],
        "rows": g.size().to_numpy(),
        "share_of_rows_%": (100 * g.size() / len(df)).round(1).to_numpy(),
        "from_trade_test_%": (100 * g["in_trade_test"].mean()).round(1).to_numpy(),
        "mean_cases_ordered": g["cases_ordered"].mean().round(2).to_numpy(),
    })
    return out


def naive_vs_true(df: pd.DataFrame, truth: pd.DataFrame, meta: Dict) -> pd.DataFrame:
    """Mean outcome difference arm k minus control, on all rows and on trade-test rows only,
    next to the true average effect: shows the bias the BAU rule creates."""
    rows = []
    t = df["in_trade_test"] == 1
    for k in range(1, 5):
        naive_all = df.loc[df.arm == k, "cases_ordered"].mean() - df.loc[df.arm == 0, "cases_ordered"].mean()
        naive_rct = (df.loc[t & (df.arm == k), "cases_ordered"].mean()
                     - df.loc[t & (df.arm == 0), "cases_ordered"].mean())
        allowed = df["channel"] != "modern_trade" if k in (3, 4) else np.ones(len(df), bool)
        naive_rct_allowed = (df.loc[t & allowed & (df.arm == k), "cases_ordered"].mean()
                             - df.loc[t & allowed & (df.arm == 0), "cases_ordered"].mean())
        rows.append({"arm": k, "label": meta["arm_labels"][k],
                     "naive_difference_all_rows": round(naive_all, 2),
                     "difference_trade_test_rows": round(naive_rct if k in (1, 2) else naive_rct_allowed, 2),
                     "true_average_effect": round(truth.loc[allowed, f"tau_{k}"].mean(), 2)})
    return pd.DataFrame(rows)


# ── ground truth summaries ──────────────────────────────────────────────────
def true_effects_by(df: pd.DataFrame, truth: pd.DataFrame, meta: Dict, by: str = "channel") -> pd.DataFrame:
    """Mean true effect (cases) and mean true net value (VND) of every arm per group."""
    t = truth.assign(**{by: df[by].to_numpy()})
    rows = []
    for g, d in t.groupby(by):
        r = {by: g, "rows": len(d)}
        for k in range(1, 5):
            r[f"tau_{meta['arm_labels'][k]}"] = round(d[f"tau_{k}"].mean(), 2)
        for k in range(1, 5):
            r[f"net_{meta['arm_labels'][k]}"] = int(round(d[f"net_value_{k}"].mean(), -2))
        rows.append(r)
    return pd.DataFrame(rows)


def true_best_mix(df: pd.DataFrame, truth: pd.DataFrame, meta: Dict, by: str = "channel") -> pd.DataFrame:
    """Share of rows (%) whose true best arm is k, per group and overall."""
    lab = truth["true_best_arm"].map(meta["arm_labels"])
    tab = pd.crosstab(df[by], lab, normalize="index").mul(100).round(1)
    tab.loc["all rows"] = lab.value_counts(normalize=True).mul(100).round(1)
    cols = [meta["arm_labels"][k] for k in range(5) if meta["arm_labels"][k] in tab.columns]
    return tab[cols].reset_index().rename(columns={by: by})


def promo_by_share(df: pd.DataFrame, truth: pd.DataFrame, bins=(0, .2, .3, .4, .5, .6, .7, .8, 1)) -> pd.DataFrame:
    """True price_promo effect and net value by bands of brand_share_est."""
    labels = [f"{a:.1f} to {b:.1f}" for a, b in zip(bins[:-1], bins[1:])]
    band = pd.cut(df["brand_share_est"], bins=list(bins), include_lowest=True, labels=labels)
    t = truth.assign(band=band.astype(str).to_numpy())
    t = t[df["brand_share_est"].notna().to_numpy()]
    g = t.groupby("band", sort=False)
    out = pd.DataFrame({"rows": g.size(), "true_effect_cases": g["tau_2"].mean().round(2),
                        "true_net_value_vnd": g["net_value_2"].mean().round(-2).astype(int),
                        "share_rows_promo_loses_%": (100 * g["net_value_2"].apply(lambda s: (s < 0).mean())).round(1)})
    return out.reindex([o for o in labels if o in out.index]).reset_index().rename(columns={"band": "brand_share_est"})


# ── policies against the truth ──────────────────────────────────────────────
def true_net(truth_rows: pd.DataFrame) -> np.ndarray:
    return truth_rows[[f"net_value_{k}" for k in range(5)]].to_numpy()


def true_value(net: np.ndarray, arms: np.ndarray) -> float:
    """Mean true net gain over all-control (VND per outlet-month) of an arm per row."""
    return float(net[np.arange(len(arms)), np.asarray(arms, int)].mean())


def policy_table(ev: aum.MultiArmEvaluator, specs: Dict[str, np.ndarray], net: np.ndarray, n_outlets_total: int) -> pd.DataFrame:
    """Estimated (DR, test rows) and true net gain over all-control for every policy.

    est_gain        DR estimate of value(policy) - value(all_control), VND per outlet-month
    ci_low/high     95% paired bootstrap CI (whole outlets resampled)
    true_gain       the same quantity computed from the true effects
    gap_to_oracle   true gain of the oracle minus true gain of the policy
    monthly_total   est_gain x n_outlets_total: VND per month for the whole outlet base
    """
    oracle = net.max(1).mean()
    rows = []
    for name, spec in specs.items():
        arms = np.asarray(spec, int)                 # one arm per test row
        d, lo, hi = ev.paired_difference(spec, "all_control")
        tv = true_value(net, arms)
        shares = {f"%_{lab}": round(100 * float((arms == k).mean()), 1) for k, lab in ARM_SHORT.items()}
        rows.append({"policy": name, "est_gain": round(d), "ci_low": round(lo), "ci_high": round(hi),
                     "true_gain": round(tv), "gap_to_oracle": round(oracle - tv),
                     "monthly_total_mVND": round(d * n_outlets_total / 1e6), **shares})
    return pd.DataFrame(rows)


ARM_SHORT = {0: "control", 1: "zalo", 2: "promo", 3: "display", 4: "rep"}


def agreement_by(df_test: pd.DataFrame, chosen: np.ndarray, true_best: np.ndarray, by: str = "channel") -> pd.DataFrame:
    """Share of rows (%) where the policy picks the true best arm, per group and overall."""
    ok = (np.asarray(chosen) == np.asarray(true_best))
    t = pd.DataFrame({by: df_test[by].to_numpy(), "ok": ok})
    g = t.groupby(by)["ok"]
    out = pd.DataFrame({by: g.size().index, "rows": g.size().to_numpy(),
                        "agreement_%": (100 * g.mean()).round(1).to_numpy()})
    return pd.concat([out, pd.DataFrame([{by: "all rows", "rows": len(t), "agreement_%": round(100 * ok.mean(), 1)}])],
                     ignore_index=True)


def allocation(df_test: pd.DataFrame, chosen: np.ndarray, meta: Dict, by: str = "channel") -> pd.DataFrame:
    """Share of rows (%) given each arm, per group."""
    lab = pd.Series(np.asarray(chosen)).map(meta["arm_labels"])
    tab = pd.crosstab(df_test[by].to_numpy(), lab.to_numpy(), normalize="index").mul(100).round(1)
    tab.index.name = by
    tab.columns.name = None
    cols = [meta["arm_labels"][k] for k in range(5)]
    return tab.reindex(columns=cols, fill_value=0.0).reset_index()


def model_vs_truth(frame: pd.DataFrame, truth_rows: pd.DataFrame, df_test: pd.DataFrame, meta: Dict,
                   by: str = "channel") -> pd.DataFrame:
    """Per arm and group: mean predicted vs mean true effect (cases), bias and correlation."""
    rows = []
    for k in range(1, 5):
        p, t = frame[f"tau_{k}"].to_numpy(), truth_rows[f"tau_{k}"].to_numpy()
        for g in list(pd.unique(df_test[by])) + ["all rows"]:
            m = np.ones(len(p), bool) if g == "all rows" else (df_test[by].to_numpy() == g)
            rows.append({"arm": meta["arm_labels"][k], by: g, "rows": int(m.sum()),
                         "mean_predicted": round(float(p[m].mean()), 2), "mean_true": round(float(t[m].mean()), 2),
                         "bias": round(float(p[m].mean() - t[m].mean()), 2),
                         "corr": round(float(np.corrcoef(p[m], t[m])[0, 1]), 3) if m.sum() > 2 else np.nan})
    return pd.DataFrame(rows)


def capacity_for(df_part: pd.DataFrame, df_all: pd.DataFrame, meta: Dict) -> Dict[int, Dict[str, int]]:
    """Scale the per-region monthly capacities of the whole outlet base to a subset of outlets."""
    n_all = df_all.groupby("region")["outlet_id"].nunique()
    n_part = df_part.groupby("region")["outlet_id"].nunique()
    return {int(k): {r: int(round(c * n_part.get(r, 0) / n_all[r])) for r, c in caps.items()}
            for k, caps in meta["capacity"].items()}


def bau_within_caps(df_part: pd.DataFrame, caps: Dict[int, Dict[str, int]], seed: int = 0) -> np.ndarray:
    """The BAU rule cut down to the capacity limits: a fair baseline for the capacity policy.

    In each region and month, when the rule gives a capped arm to more rows than the cap,
    a random subset of cap rows keeps the arm and the other rows get control (0). This is
    how a team without a model would ration: the rule picks who qualifies, chance picks who
    gets the scarce action.
    """
    arms = df_part["rule_arm"].to_numpy().copy()
    rng = np.random.default_rng(seed)
    month = pd.to_datetime(df_part["month"]).dt.to_period("M").astype(str).to_numpy()
    region = df_part["region"].to_numpy()
    for k, by_region in caps.items():
        for r, cap in by_region.items():
            for m in np.unique(month):
                idx = np.flatnonzero((arms == k) & (region == r) & (month == m))
                if len(idx) > cap:
                    arms[rng.choice(idx, size=len(idx) - cap, replace=False)] = 0
    return arms


def pilot_power(df: pd.DataFrame, value_per_outcome: float, outlets_per_side: int,
                months=(3, 6), alpha_z: float = 1.96, power_z: float = 0.84) -> pd.DataFrame:
    """Minimum detectable difference of a randomised pilot that splits whole outlets into two sides.

    Uses only observed data (no truth): the outcome of an outlet is its mean gross margin per month
    over the pilot window. The standard error of the difference of the two side means is
    sqrt(2) * sd / sqrt(outlets_per_side); the minimum detectable effect (MDE) at 5% two-sided and
    80% power is (alpha_z + power_z) * se. "adjusted" regresses the outcome on features known at
    the start of the window (vol_avg_3m, order_frequency_3m, n_coolers, channel), as a pilot
    analysis would; its residual sd replaces sd. Arm costs are left out (they are fixed per arm,
    so they shift a side mean without adding noise).
    """
    month_list = sorted(pd.to_datetime(df["month"]).unique())
    rows = []
    for m in months:
        if m > len(month_list):
            continue
        start = month_list[len(month_list) - m]
        win = df[pd.to_datetime(df["month"]) >= start]
        y = win.groupby("outlet_id")["cases_ordered"].mean() * value_per_outcome
        pre = df[pd.to_datetime(df["month"]) == start].set_index("outlet_id")[
            ["vol_avg_3m", "order_frequency_3m", "n_coolers", "channel"]]
        X = pd.get_dummies(pre.loc[y.index], columns=["channel"], drop_first=True).astype(float)
        X.insert(0, "const", 1.0)
        beta, *_ = np.linalg.lstsq(X.to_numpy(), y.to_numpy(), rcond=None)
        resid = y.to_numpy() - X.to_numpy() @ beta
        for label, sd in (("raw", float(y.std())), ("adjusted", float(resid.std()))):
            se = np.sqrt(2) * sd / np.sqrt(outlets_per_side)
            rows.append({"months": m, "analysis": label, "outlets_per_side": outlets_per_side,
                         "sd_per_outlet": round(sd), "se_difference": round(se),
                         "mde_80_power": round((alpha_z + power_z) * se),
                         "r2_adjustment": round(1 - (resid.var() / y.var()), 3) if label == "adjusted" else np.nan})
    return pd.DataFrame(rows)


# ── export for the PDF ──────────────────────────────────────────────────────
class Exporter:
    """Saves figures, tables and numbers of the run; the PDF build reads them back."""

    def __init__(self, out: Path = HERE / "outputs"):
        self.out = Path(out)
        (self.out / "figures").mkdir(parents=True, exist_ok=True)
        (self.out / "tables").mkdir(parents=True, exist_ok=True)
        self.numbers: Dict[str, object] = {}
        self.captions: Dict[str, Dict[str, str]] = {}
        self.t0 = time.time()

    def fig(self, fig, name: str, caption: str, step: str, dpi: int = 130):
        fig.savefig(self.out / "figures" / f"{name}.png", dpi=dpi, bbox_inches="tight")
        self.captions[f"fig:{name}"] = {"caption": caption, "step": step}
        return fig

    def table(self, df: pd.DataFrame, name: str, caption: str, step: str, show: bool = True):
        df.to_csv(self.out / "tables" / f"{name}.csv", index=False)
        self.captions[f"tab:{name}"] = {"caption": caption, "step": step}
        return df if show else None

    def num(self, **kv):
        for k, v in kv.items():
            if isinstance(v, (np.integer,)):
                v = int(v)
            elif isinstance(v, (np.floating,)):
                v = float(v)
            elif isinstance(v, np.bool_):
                v = bool(v)
            self.numbers[k] = v

    def environment(self, packages: Iterable[str] = ("numpy", "pandas", "scikit-learn", "lightgbm", "optuna",
                                                       "scipy", "matplotlib")):
        from importlib.metadata import version
        env = {"python": platform.python_version(), "machine": platform.machine(),
               "auto_uplift_multi_arms": aum.__version__}
        for p in packages:
            try:
                env[p] = version(p)
            except Exception as e:  # report, do not hide
                env[p] = f"unknown ({type(e).__name__})"
        self.numbers["environment"] = env
        return env

    def save(self):
        self.numbers["runtime_seconds"] = round(time.time() - self.t0)
        (self.out / "case_study_numbers.json").write_text(json.dumps(self.numbers, indent=2, default=str),
                                                          encoding="utf-8")
        (self.out / "captions.json").write_text(json.dumps(self.captions, indent=2), encoding="utf-8")
        return self.out / "case_study_numbers.json"


def fmt_vnd(x: Optional[float]) -> str:
    return f"{x:,.0f} VND"


def money_labels(fig) -> None:
    """Rewrite numbers printed with decimals in a figure's titles and text labels as whole numbers
    with thousands separators (VND). The library prints them with its `decimals` setting."""
    import re
    def fmt(m):
        return f"{float(m.group(0)):+,.0f}" if m.group(0)[0] in "+-" else f"{float(m.group(0)):,.0f}"
    pat = re.compile(r"[+-]?\d{3,}\.\d+")                 # only large numbers: money, not shares or ratios
    for ax in fig.axes:
        for t in ax.texts:
            t.set_text(pat.sub(fmt, t.get_text()))
        for loc in ("left", "center", "right"):
            ax.set_title(pat.sub(fmt, ax.get_title(loc=loc)), loc=loc)
    if getattr(fig, "_suptitle", None) is not None:
        fig._suptitle.set_text(pat.sub(fmt, fig._suptitle.get_text()))


def zoom_to_points(fig, pad: float = 0.15) -> None:
    """Limit each panel to the range of its plotted points and their error bars (same range on x and y,
    so the diagonal stays the 45-degree line). Used where a fitted line spans a much wider range than
    the data. Points are scatter offsets, marker-only lines (errorbar dots) and error-bar segments."""
    from matplotlib.collections import LineCollection, PathCollection
    for ax in fig.axes:
        pts = []
        for c in ax.collections:
            if isinstance(c, PathCollection) and len(c.get_offsets()):
                pts.append(np.asarray(c.get_offsets(), dtype=float))
            elif isinstance(c, LineCollection):
                pts += [np.asarray(seg, dtype=float) for seg in c.get_segments() if len(seg)]
        for ln in ax.lines:
            if ln.get_marker() not in (None, "None", "", " ") and ln.get_linestyle() in ("None", "", " "):
                pts.append(np.column_stack([ln.get_xdata(), ln.get_ydata()]).astype(float))
        pts = [p for p in pts if p.ndim == 2 and p.shape[1] == 2 and len(p)]
        if not pts:
            continue
        xy = np.vstack(pts)
        lo, hi = np.nanmin(xy), np.nanmax(xy)
        span = max(hi - lo, 1.0)
        ax.set_xlim(lo - pad * span, hi + pad * span)
        ax.set_ylim(lo - pad * span, hi + pad * span)

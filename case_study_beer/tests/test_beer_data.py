"""The generated data follows data_rules.md."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_gen import make_beer_promo_data, make_multi_arm_data  # noqa: E402
from data_gen.beer_promo import COST_VND, VALUE_PER_CASE_VND  # noqa: E402


@pytest.fixture(scope="module")
def beer():
    df, truth, meta = make_beer_promo_data()
    return df, truth, meta, pd.concat([df, truth], axis=1)


def test_shape_grain_and_columns(beer):
    df, truth, meta, _ = beer
    assert len(df) == 20_000 * 6 == meta["n_rows"]
    assert not df.duplicated(["outlet_id", "month"]).any()
    assert df.groupby("outlet_id")["month"].nunique().eq(6).all()
    assert meta["months"] == ["2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06"]
    assert meta["tet_month"] == ["2026-02"]
    assert len(truth) == len(df)
    for k in range(1, 5):
        assert f"tau_{k}" in truth
    assert df["province"].nunique() == 20
    assert df["region"].nunique() == 5
    assert set(df["channel"]) == {"on_trade_bar", "on_trade_restaurant", "off_trade_grocery",
                                  "off_trade_minimart", "modern_trade"}


def test_deterministic():
    a = make_beer_promo_data(n_outlets=1_000, n_months=3, random_state=7)
    b = make_beer_promo_data(n_outlets=1_000, n_months=3, random_state=7)
    pd.testing.assert_frame_equal(a[0], b[0])
    pd.testing.assert_frame_equal(a[1], b[1])
    c = make_beer_promo_data(n_outlets=1_000, n_months=3, random_state=8)
    assert not a[0]["cases_ordered"].equals(c[0]["cases_ordered"])


def test_missing_values_where_documented(beer):
    df = beer[0]
    on = df["channel"].str.startswith("on_trade")
    assert df.loc[on, "seating_capacity"].notna().all()
    assert df.loc[~on, "seating_capacity"].isna().all()
    assert 0.05 < df["brand_share_est"].isna().mean() < 0.11
    other = df.drop(columns=["seating_capacity", "brand_share_est"])
    assert other.notna().all().all()


def test_true_best_arm_differs_by_channel(beer):
    d, labels = beer[3], beer[2]["arm_labels"]
    top = d.groupby("channel")["true_best_arm"].agg(lambda s: s.value_counts().idxmax()).map(labels)
    assert top["on_trade_bar"] == "rep_visit"
    assert top["modern_trade"] == "price_promo"
    assert top.nunique() >= 2
    minimart = d[d["channel"] == "off_trade_minimart"]["true_best_arm"].value_counts(normalize=True)
    assert minimart.get(3, 0) > 0.2  # display_kit wins often in minimarts


def test_price_promo_loses_money_at_high_brand_share(beer):
    d = beer[3]
    high = d["brand_share_est"] > 0.7
    low = d["brand_share_est"] < 0.3
    assert d.loc[high, "net_value_2"].mean() < 0
    assert d.loc[low, "net_value_2"].mean() > 0


def test_modern_trade_has_no_overlap_for_display_and_rep(beer):
    d = beer[3]
    mt = d["channel"] == "modern_trade"
    assert (d.loc[mt, ["e_3", "e_4"]] == 0).all().all()
    assert not d.loc[mt, "arm"].isin([3, 4]).any()
    assert (d.loc[~mt, [f"e_{k}" for k in range(5)]].min() >= 0.14 - 1e-12).all()


def test_propensities_and_assignment(beer):
    df, _, meta, d = beer
    e = d[[f"e_{k}" for k in range(5)]].to_numpy()
    np.testing.assert_allclose(e.sum(axis=1), 1.0)
    assert abs(meta["randomised_share_realised"] - 0.7) < 0.01
    bau = df["in_trade_test"] == 0
    assert (df.loc[bau, "arm"] == df.loc[bau, "rule_arm"]).all()
    # within the test, arms are close to uniform outside modern trade
    test = df[(df["in_trade_test"] == 1) & (df["channel"] != "modern_trade")]
    shares = test["arm"].value_counts(normalize=True)
    assert ((shares - 0.2).abs() < 0.01).all()


def test_bau_rule_lines(beer):
    df = beer[0]
    on = df["channel"].str.startswith("on_trade")
    assert (df.loc[on & (df["outlet_size"] == "large"), "rule_arm"] == 4).all()
    mt_low = (df["channel"] == "modern_trade") & (df["brand_share_est"] < 0.4)
    assert (df.loc[mt_low, "rule_arm"] == 2).all()
    off2 = df["channel"].str.startswith("off_trade") & (df["n_coolers"] >= 2)
    assert (df.loc[off2, "rule_arm"] == 3).all()


def test_every_arm_helps_some_rows_and_hurts_others(beer):
    d = beer[3]
    for k in range(1, 5):
        assert (d[f"net_value_{k}"] > 0).mean() > 0.05, k
        assert (d[f"net_value_{k}"] < 0).mean() > 0.01, k


def test_rep_visit_size_by_trend_interaction(beer):
    d = beer[3]
    on = d[d["channel"].str.startswith("on_trade")]
    falling = on["vol_trend_3m"] < -0.5
    rising = on["vol_trend_3m"] > 0.15
    large, small = on["outlet_size"] == "large", on["outlet_size"] == "small"
    gap_large = on.loc[large & falling, "tau_4"].mean() - on.loc[large & rising, "tau_4"].mean()
    gap_small = on.loc[small & falling, "tau_4"].mean() - on.loc[small & rising, "tau_4"].mean()
    assert gap_large > 2 * gap_small > 0


def test_effects_larger_in_tet_month(beer):
    d = beer[3]
    tet = d["tet_month"] == 1
    for k in range(1, 5):
        assert d.loc[tet, f"tau_{k}"].mean() > d.loc[~tet, f"tau_{k}"].mean(), k


def test_net_value_and_best_arm_consistent(beer):
    d = beer[3]
    for k in range(1, 5):
        np.testing.assert_allclose(d[f"net_value_{k}"], VALUE_PER_CASE_VND * d[f"tau_{k}"] - COST_VND[k])
    nv = d[[f"net_value_{k}" for k in range(5)]].to_numpy()
    assert (nv.argmax(axis=1) == d["true_best_arm"].to_numpy()).all()


def test_planted_feature_filter_targets(beer):
    df = beer[0]
    assert df["legacy_program_flag"].mean() < 0.01
    assert df["vol_avg_3m"].corr(df["vol_avg_3m_reported"]) > 0.99
    assert abs(df["noise_feature"].corr(df["cases_ordered"])) < 0.02


def test_outcome_volume_and_reorder():
    df_v, tr_v, meta_v = make_beer_promo_data(n_outlets=2_000, n_months=2, random_state=3)
    df_r, tr_r, meta_r = make_beer_promo_data(n_outlets=2_000, n_months=2, random_state=3, outcome="reorder")
    pd.testing.assert_frame_equal(df_v, df_r)  # same draws, only the truth and money change
    assert meta_v["outcome_col"] == "cases_ordered" and meta_r["outcome_col"] == "reordered"
    assert (df_v["cases_ordered"] >= 0).all() and (df_v["cases_ordered"] == 0).mean() > 0.05
    assert set(df_r["reordered"].unique()) <= {0, 1}
    assert tr_r[[f"tau_{k}" for k in range(1, 5)]].abs().max().max() <= 1
    assert ((df_v["cases_ordered"] > 0) == (df_v["reordered"] == 1)).all()


def test_true_effect_matches_simulated_outcomes_on_test_rows(beer):
    """Difference in means on randomised rows recovers the true average effect."""
    d = beer[3]
    t = d[(d["in_trade_test"] == 1) & (d["channel"] != "modern_trade")]
    y0 = t.loc[t["arm"] == 0, "cases_ordered"].mean()
    for k in range(1, 5):
        est = t.loc[t["arm"] == k, "cases_ordered"].mean() - y0
        true = t[f"tau_{k}"].mean()
        assert abs(est - true) < 1.5, (k, est, true)


def test_invalid_arguments():
    with pytest.raises(ValueError):
        make_beer_promo_data(n_outlets=100, outcome="margin")
    with pytest.raises(ValueError):
        make_beer_promo_data(n_outlets=100, randomised_share=1.5)


def test_generic_generator():
    df, truth = make_multi_arm_data(n_rows=4_000, n_arms=3, random_state=1)
    assert {"customer_id", "date", "x1", "x2", "segment", "arm", "y"} <= set(df.columns)
    assert df["customer_id"].duplicated().any()
    assert (truth["tau_1"] < 0).any() and (truth["tau_1"] > 0).any()
    seg_mean = pd.concat([df, truth], axis=1).groupby("segment", observed=True)["tau_2"].mean()
    assert seg_mean.min() < 0 < seg_mean.max()
    dfb, trb = make_multi_arm_data(n_rows=2_000, outcome_type="binary", randomised_share=0.5, random_state=2)
    assert set(dfb["y"].unique()) <= {0, 1}
    assert trb["e_0"].min() < trb["e_0"].max()

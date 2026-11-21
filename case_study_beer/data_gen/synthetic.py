"""Generic multi-arm synthetic data with known true effects.

Two layers:

* building blocks shared with the beer demo (``beer_promo.py``):
  ``mix_assignment`` (randomised trade test mixed with a business-as-usual rule),
  ``true_best_arm`` (best arm on true net value);
* ``make_multi_arm_data``: a small generic panel with generic feature names
  (``x1``, ``x2``, ``segment``, ``customer_id``), for quick experiments.

Every function is deterministic for a given ``random_state``. The true effects
returned are exact expectations, not noisy draws: for a continuous outcome
``tau_k = E[Y(k) | x] - E[Y(0) | x]``; for a binary outcome the same on the
probability scale.
"""
from __future__ import annotations

import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

__all__ = ["make_multi_arm_data", "mix_assignment", "true_best_arm"]


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def mix_assignment(
    rule_arm: np.ndarray,
    n_arms_total: int,
    randomised_share: float,
    rng: np.random.Generator,
    allowed: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assign one arm per row: a randomised test mixed with a business-as-usual rule.

    Each row is in the randomised test with probability ``randomised_share``. A
    test row gets an arm drawn uniformly among its allowed arms; any other row
    gets ``rule_arm``.

    Parameters
    ----------
    rule_arm : array of int, shape (n,)
        Arm the business-as-usual rule gives each row (0 = control).
    n_arms_total : int
        Number of arms including control (K + 1).
    randomised_share : float
        Probability that a row is in the randomised test, between 0 and 1.
    rng : numpy Generator
    allowed : bool array, shape (n, K + 1), optional
        Arms a row may receive in the test. None = all arms. Control must be allowed.

    Returns
    -------
    arm : array of int, shape (n,)
        Assigned arm.
    randomised : bool array, shape (n,)
        True when the row was in the randomised test.
    propensity : array, shape (n, K + 1)
        True assignment probability of every arm for every row.
    """
    n = len(rule_arm)
    if not 0.0 <= randomised_share <= 1.0:
        raise ValueError(f"randomised_share must be between 0 and 1; got {randomised_share}")
    if allowed is None:
        allowed = np.ones((n, n_arms_total), bool)
    if not allowed[:, 0].all():
        raise ValueError("control (arm 0) must be allowed for every row")
    p_test = allowed / allowed.sum(axis=1, keepdims=True)
    p_rule = np.zeros((n, n_arms_total))
    p_rule[np.arange(n), rule_arm] = 1.0
    propensity = randomised_share * p_test + (1.0 - randomised_share) * p_rule

    randomised = rng.random(n) < randomised_share
    u = rng.random(n)
    test_arm = (u[:, None] > np.cumsum(p_test, axis=1)).sum(axis=1)
    arm = np.where(randomised, test_arm, rule_arm).astype(int)
    return arm, randomised, propensity


def true_best_arm(
    tau: np.ndarray,
    cost: Optional[Dict[int, float]] = None,
    value_per_outcome: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Best arm per row on true net value, ``value_per_outcome * tau_k - cost_k``.

    Parameters
    ----------
    tau : array, shape (n, K)
        True effect of arms 1..K against control.
    cost : dict {arm: float}, optional
        Cost per row of each arm; missing arms cost 0.
    value_per_outcome : float, default 1.0

    Returns
    -------
    best_arm : array of int, shape (n,)
        0 when no arm beats control.
    net_value : array, shape (n, K + 1)
        Net value of every arm, control included (``-cost_0``).
    """
    cost = cost or {}
    n, k = tau.shape
    value = np.zeros((n, k + 1))
    value[:, 0] = -cost.get(0, 0.0)
    for a in range(1, k + 1):
        value[:, a] = value_per_outcome * tau[:, a - 1] - cost.get(a, 0.0)
    return value.argmax(axis=1), value


def make_multi_arm_data(
    n_rows: int = 10_000,
    n_arms: int = 3,
    n_features: int = 6,
    n_cat_levels: int = 12,
    n_groups: int = 4,
    n_customers: Optional[int] = None,
    n_periods: int = 4,
    start_date: str = "2026-01-01",
    outcome_type: str = "continuous",
    randomised_share: float = 1.0,
    effect_scale: float = 1.0,
    noise: float = 1.0,
    cost: Optional[Dict[int, float]] = None,
    value_per_outcome: float = 1.0,
    random_state: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generic multi-arm panel with heterogeneous, known effects.

    One row is one customer in one period. Customers repeat across periods and
    can get a different arm each period.

    Effect structure (``s`` = ``effect_scale``, ``g`` = segment index 0..n_groups-1):

    * arm 1: ``s * (0.5 + 0.5 * x1)``: grows with x1, negative for x1 < -1.
    * arm 2: ``s * (0.8 if segment is even else -0.3) + 0.2 * s * x2``: opposite signs by segment.
    * arm k >= 3: ``s * (0.3 * k - 0.6 * x3 * (g == k mod n_groups))``: one segment reacts to x3.

    Assignment: ``randomised_share`` of rows get a uniform random arm; the rest
    follow a rule on observed features (arm 1 when x1 > 0.5, else arm 2 when
    x2 > 0.5, else control), which confounds x1 and x2 with the arm.

    Parameters
    ----------
    n_rows : int, default 10_000
    n_arms : int, default 3
        Number of treatment arms K (control comes on top).
    n_features : int, default 6
        Numeric features x1..x{n_features}; at least 3.
    n_cat_levels : int, default 12
        Levels of the categorical feature ``cat_high`` (up to about 20).
    n_groups : int, default 4
        Levels of the grouping column ``segment`` (product-like).
    n_customers : int, optional
        Distinct customers; None = n_rows // n_periods.
    n_periods : int, default 4
        Monthly periods; the ``date`` column holds the first day of the month.
    start_date : str, default "2026-01-01"
    outcome_type : {"continuous", "binary"}, default "continuous"
    randomised_share : float, default 1.0
        Share of rows with a randomised arm. Below 1 the data is mixed.
    effect_scale : float, default 1.0
        Multiplies every effect; larger means a stronger signal.
    noise : float, default 1.0
        Outcome noise standard deviation (continuous outcome only).
    cost, value_per_outcome :
        Used only to compute ``true_best_arm`` in the truth table.
    random_state : int, default 0

    Returns
    -------
    df : DataFrame
        customer_id, date, x1..xN, cat_high, segment, arm, rule_arm, y.
    truth : DataFrame, same index as df
        tau_1..tau_K (true effects), mu_0..mu_K (true expected outcomes),
        e_0..e_K (true propensities), true_best_arm, randomised.

    Examples
    --------
    >>> df, truth = make_multi_arm_data(n_rows=4_000, n_arms=3, random_state=1)
    >>> df.shape[0], truth.filter(like="tau_").shape[1]
    (4000, 3)
    """
    if n_features < 3:
        raise ValueError("n_features must be at least 3")
    if n_arms < 1:
        raise ValueError("n_arms must be at least 1")
    if outcome_type not in ("continuous", "binary"):
        raise ValueError(f"outcome_type must be 'continuous' or 'binary'; got '{outcome_type}'")
    if n_cat_levels > 20:
        warnings.warn(f"n_cat_levels={n_cat_levels} is above the ~20 levels the library is designed for",
                      UserWarning, stacklevel=2)
    rng = np.random.default_rng(random_state)
    n_customers = n_customers or max(1, n_rows // n_periods)

    customer = rng.integers(0, n_customers, n_rows)
    period = rng.integers(0, n_periods, n_rows)
    dates = pd.date_range(start_date, periods=n_periods, freq="MS")
    # customer-level traits, so repeated rows of a customer are correlated
    cust_level = rng.normal(size=(n_customers, n_features))
    X = 0.6 * cust_level[customer] + 0.8 * rng.normal(size=(n_rows, n_features))
    seg_of_cust = rng.integers(0, n_groups, n_customers)
    g = seg_of_cust[customer]
    cat_high = rng.integers(0, n_cat_levels, n_rows)

    df = pd.DataFrame(X, columns=[f"x{i + 1}" for i in range(n_features)])
    df.insert(0, "customer_id", customer)
    df.insert(1, "date", dates[period])
    df["cat_high"] = pd.Categorical([f"c{v:02d}" for v in cat_high])
    df["segment"] = pd.Categorical([f"seg_{v}" for v in g])

    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    s = effect_scale
    lin = np.zeros((n_rows, n_arms))
    lin[:, 0] = s * (0.5 + 0.5 * x1)
    if n_arms >= 2:
        lin[:, 1] = s * np.where(g % 2 == 0, 0.8, -0.3) + 0.2 * s * x2
    for k in range(3, n_arms + 1):
        lin[:, k - 1] = s * (0.3 * k - 0.6 * x3 * (g == k % n_groups))
    base = 1.0 + 0.8 * x2 - 0.5 * X[:, 3 % n_features] + 0.1 * (cat_high % 5) + 0.3 * period

    if outcome_type == "continuous":
        mu = np.column_stack([base] + [base + lin[:, k] for k in range(n_arms)])
    else:
        z0 = 0.5 * (base - 1.0) - 0.3
        mu = np.column_stack([_sigmoid(z0)] + [_sigmoid(z0 + 0.6 * lin[:, k]) for k in range(n_arms)])
    tau = mu[:, 1:] - mu[:, [0]]

    rule = np.where(x1 > 0.5, 1, np.where((x2 > 0.5) & (n_arms >= 2), 2, 0)).astype(int)
    arm, randomised, e = mix_assignment(rule, n_arms + 1, randomised_share, rng)
    mu_obs = mu[np.arange(n_rows), arm]
    if outcome_type == "continuous":
        y = mu_obs + noise * rng.normal(size=n_rows)
    else:
        y = (rng.random(n_rows) < mu_obs).astype(int)
    df["arm"] = arm
    df["rule_arm"] = rule
    df["y"] = y

    best, _ = true_best_arm(tau, cost, value_per_outcome)
    truth = pd.DataFrame(index=df.index)
    for k in range(1, n_arms + 1):
        truth[f"tau_{k}"] = tau[:, k - 1]
    for k in range(n_arms + 1):
        truth[f"mu_{k}"] = mu[:, k]
    for k in range(n_arms + 1):
        truth[f"e_{k}"] = e[:, k]
    truth["true_best_arm"] = best
    truth["randomised"] = randomised
    return df, truth

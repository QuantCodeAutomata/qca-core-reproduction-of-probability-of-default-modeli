"""Tests for src/preprocessing.py."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import pytest

from src.calendar_utils import generate_banking_days, get_month_end_banking_days
from src.preprocessing import (
    aggregate_to_borrower_level,
    aggregate_to_household_level,
    apply_winsorization,
    assign_sampling_dates,
    compute_account_level_daily,
    create_default_labels,
    reconstruct_eod_states,
    winsorize_series,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_minimal_accounts():
    return pd.DataFrame(
        {
            "account_id": [1, 2],
            "borrower_id": [1, 1],
            "household_id": [10, 10],
            "account_type": [1, 2],
            "open_date": [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-01")],
            "credit_limit": [0.0, 1000.0],
        }
    )


def _make_minimal_balances():
    return pd.DataFrame(
        {
            "account_id": [1, 2],
            "date": [pd.Timestamp("2020-01-31"), pd.Timestamp("2020-01-31")],
            "balance": [5000.0, -300.0],
        }
    )


# ---------------------------------------------------------------------------
# reconstruct_eod_states
# ---------------------------------------------------------------------------

def test_eod_states_cadlag():
    """LOCF: once a balance observation exists, every subsequent banking day has
    a non-NaN balance (càdlàg / last-observation-carried-forward property)."""
    bd = generate_banking_days("2020-01-01", "2020-04-30")
    accounts = _make_minimal_accounts()
    balances = _make_minimal_balances()  # single observation on 2020-01-31

    eod = reconstruct_eod_states(accounts, balances, bd)

    first_obs = pd.Timestamp("2020-01-31").normalize()
    after_first = bd[bd >= first_obs]

    for acct_id in [1, 2]:
        acct_rows = eod[eod["account_id"] == acct_id]
        acct_dates = set(acct_rows["date"].dt.normalize())
        # Every banking day from first observation onwards must be present
        for d in after_first:
            assert d.normalize() in acct_dates, (
                f"Account {acct_id}: missing banking day {d.date()} after first observation"
            )
        # All balances after first observation must be non-NaN
        after_rows = acct_rows[acct_rows["date"].dt.normalize() >= first_obs]
        assert after_rows["balance"].notna().all(), (
            f"Account {acct_id}: NaN balance after first observation (LOCF violated)"
        )


def test_eod_states_covers_all_active_days():
    """eod_states must have exactly one row per (account, banking_day) pair
    where the account is active."""
    bd = generate_banking_days("2020-01-01", "2020-02-29")
    accounts = _make_minimal_accounts()
    balances = _make_minimal_balances()

    eod = reconstruct_eod_states(accounts, balances, bd)

    for acct_id in [1, 2]:
        n_rows = (eod["account_id"] == acct_id).sum()
        assert n_rows == len(bd), (
            f"Account {acct_id}: expected {len(bd)} rows, got {n_rows}"
        )


def test_eod_states_missing_columns_raises():
    """reconstruct_eod_states must raise ValueError for missing required columns."""
    bd = generate_banking_days("2020-01-01", "2020-01-31")
    bad_accounts = pd.DataFrame({"account_id": [1]})  # missing many columns
    balances = _make_minimal_balances()
    with pytest.raises(ValueError):
        reconstruct_eod_states(bad_accounts, balances, bd)


# ---------------------------------------------------------------------------
# compute_account_level_daily
# ---------------------------------------------------------------------------

def test_available_funds_formula(small_sample_data):
    """available_funds must equal balance + credit_limit for every row."""
    df = small_sample_data["account_daily"]
    diff = (df["available_funds"] - (df["balance"] + df["credit_limit"])).abs()
    assert diff.max() < 1e-9, (
        f"available_funds ≠ balance + credit_limit; max diff = {diff.max()}"
    )


def test_utilization_nonnegative(small_sample_data):
    """utilization must be ≥ 0 for every row."""
    df = small_sample_data["account_daily"]
    assert (df["utilization"] >= 0).all(), "Negative utilization found"


def test_overdraft_nonpositive(small_sample_data):
    """overdraft must be ≤ 0 for every row."""
    df = small_sample_data["account_daily"]
    assert (df["overdraft"] <= 0).all(), "Positive overdraft found"


def test_available_funds_formula_inline():
    """Verify available_funds formula on hand-crafted data."""
    eod = pd.DataFrame(
        {
            "account_id": [1, 2, 3],
            "borrower_id": [1, 1, 2],
            "household_id": [10, 10, 20],
            "date": [pd.Timestamp("2021-01-29")] * 3,
            "balance": [3000.0, -400.0, -1000.0],
            "credit_limit": [0.0, 500.0, 800.0],
            "account_type": [1, 2, 2],
        }
    )
    out = compute_account_level_daily(eod)
    expected_af = [3000.0, 100.0, -200.0]
    for i, exp in enumerate(expected_af):
        got = out.iloc[i]["available_funds"]
        assert abs(got - exp) < 1e-9, f"Row {i}: expected {exp}, got {got}"


def test_overdraft_min_zero():
    """overdraft = min(available_funds, 0); positive available_funds → overdraft=0."""
    eod = pd.DataFrame(
        {
            "account_id": [1],
            "borrower_id": [1],
            "household_id": [10],
            "date": [pd.Timestamp("2021-01-29")],
            "balance": [5000.0],
            "credit_limit": [0.0],
            "account_type": [1],
        }
    )
    out = compute_account_level_daily(eod)
    assert out.iloc[0]["overdraft"] == 0.0, "overdraft should be 0 when available_funds > 0"


# ---------------------------------------------------------------------------
# aggregate_to_borrower_level
# ---------------------------------------------------------------------------

def test_borrower_aggregation_sums():
    """Borrower-level aggregated values must equal the sum of constituent account values."""
    account_daily = pd.DataFrame(
        {
            "account_id": [1, 2, 3],
            "borrower_id": [7, 7, 7],
            "household_id": [20, 20, 20],
            "date": [pd.Timestamp("2021-06-30")] * 3,
            "balance": [2000.0, -600.0, 1500.0],
            "credit_limit": [0.0, 2000.0, 0.0],
            "available_funds": [2000.0, 1400.0, 1500.0],
            "overdraft": [0.0, 0.0, 0.0],
            "utilization": [0.0, 0.3, 0.0],
        }
    )
    borrower = aggregate_to_borrower_level(account_daily)
    row = borrower[borrower["borrower_id"] == 7].iloc[0]

    assert abs(row["balance"] - (2000.0 - 600.0 + 1500.0)) < 1e-9, "balance sum mismatch"
    assert abs(row["credit_limit"] - 2000.0) < 1e-9, "credit_limit sum mismatch"
    assert abs(row["available_funds"] - (2000.0 + 1400.0 + 1500.0)) < 1e-9, (
        "available_funds sum mismatch"
    )


def test_gearing_denominator_floor():
    """gearing = credit_limit / max(available_funds, 1000); floor at 1000 DKK."""
    account_daily = pd.DataFrame(
        {
            "account_id": [1],
            "borrower_id": [1],
            "household_id": [1],
            "date": [pd.Timestamp("2021-06-30")],
            "balance": [-900.0],
            "credit_limit": [1000.0],
            "available_funds": [100.0],  # < 1000 → denominator floor kicks in
            "overdraft": [-100.0],
            "utilization": [0.9],
        }
    )
    borrower = aggregate_to_borrower_level(account_daily)
    row = borrower.iloc[0]

    expected_gearing = 1000.0 / max(100.0, 1000.0)  # = 1.0
    assert abs(row["gearing"] - expected_gearing) < 1e-9, (
        f"Expected gearing={expected_gearing:.4f}, got {row['gearing']:.4f}"
    )


def test_gearing_no_floor_when_above_threshold():
    """When available_funds ≥ 1000, gearing denominator must not be floored."""
    account_daily = pd.DataFrame(
        {
            "account_id": [1],
            "borrower_id": [1],
            "household_id": [1],
            "date": [pd.Timestamp("2021-06-30")],
            "balance": [-500.0],
            "credit_limit": [5000.0],
            "available_funds": [4500.0],  # > 1000
            "overdraft": [0.0],
            "utilization": [0.1],
        }
    )
    borrower = aggregate_to_borrower_level(account_daily)
    row = borrower.iloc[0]

    expected_gearing = 5000.0 / 4500.0
    assert abs(row["gearing"] - expected_gearing) < 1e-9, (
        f"Expected gearing={expected_gearing:.4f}, got {row['gearing']:.4f}"
    )


# ---------------------------------------------------------------------------
# create_default_labels
# ---------------------------------------------------------------------------

def test_default_label_window():
    """Y=1 iff default_date ∈ (T, T+12 months]; defaults on or before T → Y=0."""
    T = pd.Timestamp("2020-06-30")
    sampling_df = pd.DataFrame(
        {
            "borrower_id": [1, 2, 3, 4],
            "sampling_date": [T] * 4,
            "sample_type": ["dev"] * 4,
        }
    )
    # Borrower 1: defaults on exactly T (not in open interval (T, …]) → Y=0
    # Borrower 2: defaults 6 months after T (within window) → Y=1
    # Borrower 3: defaults 13 months after T (after window) → Y=0
    # Borrower 4: no default → Y=0
    defaults_df = pd.DataFrame(
        {
            "borrower_id": [1, 2, 3],
            "default_date": [
                T,
                T + pd.DateOffset(months=6),
                T + pd.DateOffset(months=13),
            ],
        }
    )
    labeled = create_default_labels(sampling_df, defaults_df, horizon_months=12)

    labels = labeled.set_index("borrower_id")["default_label"]
    assert labels[1] == 0, "Default on T must not be labelled Y=1"
    assert labels[2] == 1, "Default 6 months after T must be labelled Y=1"
    assert labels[3] == 0, "Default 13 months after T must be labelled Y=0"
    assert labels[4] == 0, "Non-defaulting borrower must be labelled Y=0"


def test_default_label_window_horizon_boundary():
    """Default exactly at T + horizon_months must be included (closed upper bound)."""
    T = pd.Timestamp("2020-01-31")
    sampling_df = pd.DataFrame(
        {"borrower_id": [1], "sampling_date": [T], "sample_type": ["dev"]}
    )
    defaults_df = pd.DataFrame(
        {
            "borrower_id": [1],
            "default_date": [T + pd.DateOffset(months=12)],  # exactly at boundary
        }
    )
    labeled = create_default_labels(sampling_df, defaults_df, horizon_months=12)
    assert labeled.iloc[0]["default_label"] == 1, (
        "Default at exactly T+12 months must be Y=1 (closed upper bound)"
    )


def test_default_label_no_defaults():
    """All labels must be 0 when defaults_df is empty."""
    sampling_df = pd.DataFrame(
        {
            "borrower_id": [1, 2, 3],
            "sampling_date": [pd.Timestamp("2020-06-30")] * 3,
            "sample_type": ["dev"] * 3,
        }
    )
    labeled = create_default_labels(sampling_df, pd.DataFrame(columns=["borrower_id", "default_date"]))
    assert (labeled["default_label"] == 0).all(), "Labels must all be 0 with empty defaults"


# ---------------------------------------------------------------------------
# winsorize_series / apply_winsorization
# ---------------------------------------------------------------------------

def test_winsorization_thresholds():
    """winsorize_series must return lower < upper matching the requested percentiles."""
    rng = np.random.default_rng(7)
    series = pd.Series(rng.normal(size=500))
    lower, upper = winsorize_series(series, lower_pct=0.02, upper_pct=0.98)

    assert lower < upper, "Lower bound must be less than upper bound"
    assert abs(lower - float(series.quantile(0.02))) < 1e-9, "Lower bound mismatch"
    assert abs(upper - float(series.quantile(0.98))) < 1e-9, "Upper bound mismatch"


def test_winsorization_clips_correctly():
    """apply_winsorization must clip values outside [lower, upper]."""
    series = pd.Series(np.linspace(-10.0, 10.0, 100))
    lower, upper = winsorize_series(series, lower_pct=0.05, upper_pct=0.95)

    df = pd.DataFrame({"x": series})
    df_out = apply_winsorization(df, ["x"], {"x": (lower, upper)})

    assert df_out["x"].min() >= lower - 1e-12, "Value below lower bound after winsorization"
    assert df_out["x"].max() <= upper + 1e-12, "Value above upper bound after winsorization"


def test_winsorization_invalid_bounds_raises():
    """lower_pct ≥ upper_pct must raise ValueError."""
    series = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        winsorize_series(series, lower_pct=0.9, upper_pct=0.1)


def test_winsorization_untouched_values():
    """Values inside the winsorization range must remain unchanged."""
    series = pd.Series([0.0, 5.0, 10.0])
    lower, upper = winsorize_series(series, lower_pct=0.0, upper_pct=1.0)
    df = pd.DataFrame({"x": series})
    df_out = apply_winsorization(df, ["x"], {"x": (lower, upper)})
    pd.testing.assert_series_equal(df["x"], df_out["x"], check_names=False)


# ---------------------------------------------------------------------------
# assign_sampling_dates
# ---------------------------------------------------------------------------

def test_sampling_dates_in_window():
    """All assigned sampling dates must fall within the declared dev or OOT window."""
    bd = generate_banking_days("2015-01-01", "2025-12-31")
    me = get_month_end_banking_days("2015-01-01", "2025-12-31")

    borrower_ids = list(range(1, 31))
    sampling = assign_sampling_dates(
        borrower_ids,
        bd,
        me,
        dev_start="2017-08-31",
        dev_end="2019-12-31",
        oot_start="2021-01-31",
        oot_end="2024-12-31",
        seed=99,
    )

    dev_start = pd.Timestamp("2017-08-31")
    dev_end = pd.Timestamp("2019-12-31")
    oot_start = pd.Timestamp("2021-01-31")
    oot_end = pd.Timestamp("2024-12-31")

    for _, row in sampling.iterrows():
        d = pd.Timestamp(row["sampling_date"]).normalize()
        if row["sample_type"] == "dev":
            assert dev_start <= d <= dev_end, f"Dev date {d} out of range"
        else:
            assert oot_start <= d <= oot_end, f"OOT date {d} out of range"


def test_sampling_dates_one_per_borrower():
    """Each borrower must receive exactly one sampling date."""
    bd = generate_banking_days("2015-01-01", "2025-12-31")
    me = get_month_end_banking_days("2015-01-01", "2025-12-31")
    borrower_ids = list(range(1, 21))
    sampling = assign_sampling_dates(borrower_ids, bd, me, seed=0)
    assert len(sampling) == len(borrower_ids), (
        f"Expected {len(borrower_ids)} rows, got {len(sampling)}"
    )
    assert sampling["borrower_id"].nunique() == len(borrower_ids), (
        "Duplicate borrower_ids in sampling DataFrame"
    )


def test_sampling_dates_reproducible():
    """Same seed must produce identical results."""
    bd = generate_banking_days("2015-01-01", "2025-12-31")
    me = get_month_end_banking_days("2015-01-01", "2025-12-31")
    borrower_ids = list(range(1, 11))
    s1 = assign_sampling_dates(borrower_ids, bd, me, seed=123)
    s2 = assign_sampling_dates(borrower_ids, bd, me, seed=123)
    pd.testing.assert_frame_equal(s1.reset_index(drop=True), s2.reset_index(drop=True))

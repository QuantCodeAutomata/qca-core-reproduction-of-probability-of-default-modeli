"""Tests for src/calendar_utils.py."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import pytest

from src.calendar_utils import (
    generate_banking_days,
    get_banking_days_in_month,
    get_month_end_banking_days,
    get_relative_time_grid,
)


# ---------------------------------------------------------------------------
# generate_banking_days
# ---------------------------------------------------------------------------

def test_banking_days_no_weekends():
    """Banking days must not include Saturdays (5) or Sundays (6)."""
    bd = generate_banking_days("2020-01-01", "2020-12-31")
    weekdays = {d.dayofweek for d in bd}
    assert 5 not in weekdays, "Saturday found in banking days"
    assert 6 not in weekdays, "Sunday found in banking days"


def test_banking_days_range():
    """Every banking day must lie within the requested [start, end] window."""
    start, end = "2021-03-01", "2021-05-31"
    bd = generate_banking_days(start, end)
    assert len(bd) > 0, "Expected non-empty banking days for the given range"
    assert all(pd.Timestamp(start) <= d <= pd.Timestamp(end) for d in bd), (
        "Some banking days fall outside the requested range"
    )


def test_banking_days_end_before_start_raises():
    """Passing end_date < start_date must raise ValueError."""
    with pytest.raises(ValueError):
        generate_banking_days("2021-12-31", "2021-01-01")


def test_banking_days_unsupported_country_raises():
    """Unsupported country codes must raise NotImplementedError."""
    with pytest.raises(NotImplementedError):
        generate_banking_days("2021-01-01", "2021-12-31", country="US")


def test_banking_days_known_danish_holiday_excluded():
    """Christmas Day (25 Dec) must not appear in banking days."""
    bd = generate_banking_days("2022-12-01", "2022-12-31")
    christmas = pd.Timestamp("2022-12-25")
    assert christmas not in set(bd), "Christmas Day should not be a banking day"


def test_banking_days_known_new_years_day_excluded():
    """New Year's Day (1 Jan) must not appear in banking days."""
    bd = generate_banking_days("2022-01-01", "2022-01-31")
    new_year = pd.Timestamp("2022-01-01")
    assert new_year not in set(bd), "New Year's Day should not be a banking day"


# ---------------------------------------------------------------------------
# get_month_end_banking_days
# ---------------------------------------------------------------------------

def test_month_end_days_are_banking_days(banking_days):
    """Every month-end banking day must belong to the full banking-day index."""
    me = get_month_end_banking_days("2020-01-01", "2020-12-31")
    bd = generate_banking_days("2020-01-01", "2020-12-31")
    bd_set = set(bd)
    for ts in me.values:
        assert pd.Timestamp(ts).normalize() in bd_set, (
            f"Month-end day {ts} is not a banking day"
        )


def test_month_end_days_count():
    """There must be exactly one month-end banking day per calendar month."""
    me = get_month_end_banking_days("2020-01-01", "2020-12-31")
    assert len(me) == 12, f"Expected 12 month-end days, got {len(me)}"


def test_month_end_days_last_in_month():
    """Each month-end banking day must be the last banking day of its month."""
    bd = generate_banking_days("2020-01-01", "2020-12-31")
    me = get_month_end_banking_days("2020-01-01", "2020-12-31")
    for period, ts in me.items():
        p = pd.Period(period, freq="M")
        month_bdays = get_banking_days_in_month(p.year, p.month, bd)
        assert pd.Timestamp(ts).normalize() == month_bdays[-1], (
            f"Month-end for {p} is not the last banking day of the month"
        )


# ---------------------------------------------------------------------------
# get_banking_days_in_month
# ---------------------------------------------------------------------------

def test_banking_days_in_month(banking_days):
    """All returned days must be in the given year-month and be banking days."""
    days = get_banking_days_in_month(2021, 6, banking_days)
    assert len(days) > 0, "Expected non-empty list for June 2021"
    for d in days:
        assert d.year == 2021 and d.month == 6, (
            f"Day {d} is not in June 2021"
        )
        assert d in set(banking_days), f"Day {d} is not in the banking-days index"


def test_banking_days_in_month_invalid_month_raises(banking_days):
    """Month outside [1, 12] must raise ValueError."""
    with pytest.raises(ValueError):
        get_banking_days_in_month(2021, 0, banking_days)
    with pytest.raises(ValueError):
        get_banking_days_in_month(2021, 13, banking_days)


def test_banking_days_in_month_no_weekends(banking_days):
    """Days returned by get_banking_days_in_month must all be weekdays."""
    days = get_banking_days_in_month(2022, 4, banking_days)
    for d in days:
        assert d.dayofweek < 5, f"Weekend day {d} returned by get_banking_days_in_month"


# ---------------------------------------------------------------------------
# get_relative_time_grid
# ---------------------------------------------------------------------------

def test_relative_time_grid_domain(banking_days):
    """τ values must lie strictly in (0, M] for M in {3, 6, 9, 12}."""
    T = "2021-06-30"
    for M in [3, 6, 9, 12]:
        tau = get_relative_time_grid(T, M, banking_days)
        assert len(tau) > 0, f"Empty time grid for M={M}"
        assert np.all(tau > 0), f"Some τ ≤ 0 for M={M}"
        assert np.all(tau <= M), f"Some τ > M={M}"


def test_relative_time_grid_monotone(banking_days):
    """τ values must be strictly increasing."""
    tau = get_relative_time_grid("2021-06-30", 6, banking_days)
    assert len(tau) > 1, "Expected at least 2 τ values"
    diffs = np.diff(tau)
    assert np.all(diffs > 0), "τ values are not strictly increasing"


def test_relative_time_grid_month_ends_at_integers(banking_days):
    """The last banking day in each lookback month must map to an integer τ."""
    M = 4
    T = pd.Timestamp("2021-06-30").normalize()
    tau = get_relative_time_grid(T, M, banking_days)

    # For month m (1..M), the last banking day has j=k_m, giving τ = j/k_m + (m-1) = m
    # So τ values 1, 2, ..., M must appear (within floating-point tolerance)
    for m in range(1, M + 1):
        assert any(abs(t - m) < 1e-10 for t in tau), (
            f"Integer τ={m} not found in grid for M={M}"
        )


def test_relative_time_grid_invalid_M_raises(banking_days):
    """M < 1 must raise ValueError."""
    with pytest.raises(ValueError):
        get_relative_time_grid("2021-06-30", 0, banking_days)


def test_relative_time_grid_consistent_length(banking_days):
    """Grid length must equal the total number of banking days across M months."""
    M = 3
    T = pd.Timestamp("2021-06-30").normalize()
    tau = get_relative_time_grid(T, M, banking_days)

    # Count banking days manually across the M months before T
    total = 0
    for months_before in range(1, M + 1):
        raw_month = T.month - months_before
        t_year = T.year + (raw_month - 1) // 12
        t_month = (raw_month - 1) % 12 + 1
        total += len(get_banking_days_in_month(t_year, t_month, banking_days))

    assert len(tau) == total, (
        f"Expected {total} τ values, got {len(tau)}"
    )

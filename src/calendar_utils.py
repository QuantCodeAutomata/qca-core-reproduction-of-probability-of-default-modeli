"""
Banking-day calendar utilities for the Danish retail banking PD model.

All holiday rules are computed from first principles (no external holiday
libraries).  Easter is computed with the Meeus/Jones/Butcher algorithm.

Danish banking-day closures recognised
---------------------------------------
* New Year's Day            – 1 January
* Good Friday               – Easter − 2 days
* Easter Monday             – Easter + 1 day
* Store Bededag             – Easter + 19 days  (years < 2024; abolished 2024)
* Ascension Day             – Easter + 39 days
* Whit Monday               – Easter + 50 days
* Christmas Eve             – 24 December
* Christmas Day             – 25 December
* Boxing Day                – 26 December
* New Year's Eve            – 31 December
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Easter computation
# ---------------------------------------------------------------------------

def _easter_sunday(year: int) -> pd.Timestamp:
    """Return Easter Sunday for *year* (Meeus/Jones/Butcher algorithm).

    Parameters
    ----------
    year : int
        Gregorian calendar year (≥ 1900).

    Returns
    -------
    pd.Timestamp
        Midnight on Easter Sunday of that year.
    """
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    ll = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ll) // 451
    month = (h + ll - 7 * m + 114) // 31
    day = (h + ll - 7 * m + 114) % 31 + 1
    return pd.Timestamp(year=year, month=month, day=day)


def _dk_public_holidays(year: int) -> set:
    """Return the set of Danish public banking-holiday dates for *year*.

    Parameters
    ----------
    year : int
        Calendar year.

    Returns
    -------
    set of pd.Timestamp
        Each timestamp is normalised to midnight (time component = 0).
    """
    easter = _easter_sunday(year)
    # Offsets relative to Easter Sunday
    holidays: set = {
        pd.Timestamp(year, 1, 1),                    # New Year's Day
        easter - pd.Timedelta(days=2),               # Good Friday
        easter + pd.Timedelta(days=1),               # Easter Monday
        easter + pd.Timedelta(days=39),              # Ascension Day
        easter + pd.Timedelta(days=50),              # Whit Monday
        pd.Timestamp(year, 12, 24),                  # Christmas Eve
        pd.Timestamp(year, 12, 25),                  # Christmas Day
        pd.Timestamp(year, 12, 26),                  # Boxing Day
        pd.Timestamp(year, 12, 31),                  # New Year's Eve
    }
    # Store Bededag: 4th Friday after Maundy Thursday = Easter + 19 days
    # Abolished from 2024 onwards.
    if year < 2024:
        holidays.add(easter + pd.Timedelta(days=19))
    return holidays


# ---------------------------------------------------------------------------
# Core calendar functions
# ---------------------------------------------------------------------------

def generate_banking_days(
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    country: str = "DK",
) -> pd.DatetimeIndex:
    """Return all banking days (Mon–Fri, excluding public holidays) in a range.

    Parameters
    ----------
    start_date : str or pd.Timestamp
        First date of the range (inclusive), ISO format if string.
    end_date : str or pd.Timestamp
        Last date of the range (inclusive), ISO format if string.
    country : str
        Country code.  Only ``'DK'`` (Denmark) is currently supported.

    Returns
    -------
    pd.DatetimeIndex
        Sorted index of banking days (normalised to midnight).

    Raises
    ------
    NotImplementedError
        If *country* is not ``'DK'``.
    ValueError
        If *end_date* is before *start_date*.
    """
    if country != "DK":
        raise NotImplementedError(
            f"Country '{country}' is not supported.  Only 'DK' is implemented."
        )

    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if end < start:
        raise ValueError(f"end_date ({end.date()}) must be ≥ start_date ({start.date()}).")

    # All Mon–Fri in the range
    all_weekdays = pd.date_range(start=start, end=end, freq="B")

    # Collect holidays spanning the range (one call per year)
    years = range(start.year, end.year + 1)
    holiday_set: set = set()
    for y in years:
        holiday_set |= _dk_public_holidays(y)

    # Filter weekdays that are NOT in the holiday set
    mask = ~all_weekdays.isin(holiday_set)
    return all_weekdays[mask]


def get_month_end_banking_days(
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    country: str = "DK",
) -> pd.Series:
    """Return the last banking day of every calendar month in a date range.

    Parameters
    ----------
    start_date : str or pd.Timestamp
        Start of the range (inclusive).
    end_date : str or pd.Timestamp
        End of the range (inclusive).
    country : str
        Country code passed to :func:`generate_banking_days`.

    Returns
    -------
    pd.Series
        Index: ``pd.PeriodIndex`` with monthly frequency (``'M'``).
        Values: ``pd.Timestamp`` of the last banking day in each month.
    """
    banking_days = generate_banking_days(start_date, end_date, country=country)
    if len(banking_days) == 0:
        return pd.Series(dtype="datetime64[ns]")

    bd_series = pd.Series(banking_days, index=banking_days)
    bd_series.index = pd.PeriodIndex(bd_series.index, freq="M")

    # Last banking day per month
    month_end_bdays = bd_series.groupby(level=0).last()
    return month_end_bdays


def get_banking_days_in_month(
    year: int,
    month: int,
    banking_days: pd.DatetimeIndex,
) -> List[pd.Timestamp]:
    """Return all banking days in a specific calendar month.

    Parameters
    ----------
    year : int
        Calendar year.
    month : int
        Calendar month (1–12).
    banking_days : pd.DatetimeIndex
        Pre-computed index of valid banking days (from
        :func:`generate_banking_days`).

    Returns
    -------
    list of pd.Timestamp
        Banking days in chronological order, normalised to midnight.

    Raises
    ------
    ValueError
        If *month* is outside [1, 12].
    """
    if not 1 <= month <= 12:
        raise ValueError(f"month must be between 1 and 12, got {month}.")

    mask = (banking_days.year == year) & (banking_days.month == month)
    return list(banking_days[mask])


def get_relative_time_grid(
    T: str | pd.Timestamp,
    M: int,
    banking_days: pd.DatetimeIndex,
) -> np.ndarray:
    """Compute the relative-time grid for a functional predictor lookback window.

    Maps each banking day in the M-calendar-month lookback window before *T*
    to a scalar τ ∈ (0, M] via

        τ_j^(m)  =  j / k_m  +  (m − 1)

    where:
    * m = 1 is the calendar month M months before T (oldest),
      m = M is the calendar month 1 month before T (most recent).
    * j ∈ {1, …, k_m} is the ordinal index of the banking day within month m.
    * k_m is the total number of banking days in month m.

    Parameters
    ----------
    T : str or pd.Timestamp
        Sampling date (end-of-month banking day).  The lookback window covers
        the M calendar months strictly before the calendar month of *T*.
    M : int
        Lookback depth in calendar months (must be ≥ 1).
    banking_days : pd.DatetimeIndex
        Full set of valid banking days from :func:`generate_banking_days`.

    Returns
    -------
    np.ndarray of shape (n_points,) in float64
        Relative time values τ in ascending order.  Empty array if no banking
        days fall in the lookback window.

    Raises
    ------
    ValueError
        If *M* < 1.
    """
    if M < 1:
        raise ValueError(f"M must be at least 1, got {M}.")

    T_ts = pd.Timestamp(T).normalize()
    T_year, T_month = T_ts.year, T_ts.month

    tau_values: list[float] = []

    for m in range(1, M + 1):
        # m=1 → M months before T; m=M → 1 month before T
        months_before = M - m + 1
        raw_month = T_month - months_before
        target_year = T_year + (raw_month - 1) // 12  # floor division
        target_month = ((raw_month - 1) % 12) + 1

        month_bdays = get_banking_days_in_month(target_year, target_month, banking_days)
        k_m = len(month_bdays)
        if k_m == 0:
            # No banking days recorded for this month; skip gracefully
            continue

        for j in range(1, k_m + 1):
            tau_values.append(j / k_m + (m - 1))

    return np.array(tau_values, dtype=np.float64)

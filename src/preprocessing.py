"""
Data preprocessing for retail banking Probability of Default (PD) modelling.

Pipeline stages
---------------
1.  :func:`reconstruct_eod_states`       – LOCF balance reconstruction
2.  :func:`compute_account_level_daily`  – derived account variables
3.  :func:`aggregate_to_borrower_level`  – borrower-level daily aggregates
4.  :func:`aggregate_to_household_level` – household-level daily aggregates
5.  :func:`assign_sampling_dates`        – random dev / OOT split
6.  :func:`create_default_labels`        – 12-month default label
7.  :func:`winsorize_series`             – compute winsorization bounds
8.  :func:`apply_winsorization`          – apply pre-computed bounds
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. EOD state reconstruction (LOCF / càdlàg)
# ---------------------------------------------------------------------------

def reconstruct_eod_states(
    accounts_df: pd.DataFrame,
    balances_df: pd.DataFrame,
    banking_days: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Reconstruct end-of-banking-day account states via LOCF.

    For each account and each banking day on or after the account's
    ``open_date`` (and on or before ``close_date`` if that column exists),
    the balance is set to the most-recent observed balance on or before that
    banking day (last observation carried forward, càdlàg convention).

    Accounts with no balance observations on or before a given banking day
    receive ``NaN`` for that day's balance.

    Parameters
    ----------
    accounts_df : pd.DataFrame
        Account metadata with columns: account_id, borrower_id, household_id,
        account_type, open_date, credit_limit.  An optional ``close_date``
        column restricts the active window.
    balances_df : pd.DataFrame
        Sparse balance events with columns: account_id, date, balance.
    banking_days : pd.DatetimeIndex
        Complete sequence of valid banking days from
        :func:`src.calendar_utils.generate_banking_days`.

    Returns
    -------
    pd.DataFrame
        Columns: account_id, borrower_id, household_id, account_type, date,
        balance, credit_limit.  One row per (account, banking_day) pair where
        the account is active.

    Raises
    ------
    ValueError
        If required columns are absent from *accounts_df* or *balances_df*.
    """
    required_acct = {"account_id", "borrower_id", "household_id", "account_type",
                     "open_date", "credit_limit"}
    missing = required_acct - set(accounts_df.columns)
    if missing:
        raise ValueError(f"accounts_df is missing columns: {missing}")

    required_bal = {"account_id", "date", "balance"}
    if not required_bal.issubset(set(balances_df.columns)):
        raise ValueError(
            f"balances_df is missing columns: {required_bal - set(balances_df.columns)}"
        )

    if len(accounts_df) == 0 or len(banking_days) == 0:
        return pd.DataFrame(
            columns=["account_id", "borrower_id", "household_id",
                     "account_type", "date", "balance", "credit_limit"]
        )

    has_close = "close_date" in accounts_df.columns

    # ---- Build (account_id, date) grid for all active (account, banking_day) pairs ---
    grid_parts: list[pd.DataFrame] = []
    for row in accounts_df.itertuples(index=False):
        a_id: int = row.account_id
        open_date: pd.Timestamp = pd.Timestamp(row.open_date).normalize()
        active = banking_days[banking_days >= open_date]
        if has_close and row.close_date is not None and not pd.isna(row.close_date):
            close_date = pd.Timestamp(row.close_date).normalize()
            active = active[active <= close_date]
        if len(active) == 0:
            continue
        grid_parts.append(
            pd.DataFrame({"account_id": np.int64(a_id), "date": active})
        )

    if not grid_parts:
        return pd.DataFrame(
            columns=["account_id", "borrower_id", "household_id",
                     "account_type", "date", "balance", "credit_limit"]
        )

    grid = pd.concat(grid_parts, ignore_index=True)
    # merge_asof requires the 'on' key (date) to be globally sorted in both frames.
    grid = grid.sort_values("date").reset_index(drop=True)

    # ---- LOCF via merge_asof (vectorised, single pass) ---
    bal_sorted = (
        balances_df[["account_id", "date", "balance"]]
        .sort_values("date")
        .reset_index(drop=True)
    )
    # Normalise dates to midnight for safe matching
    bal_sorted["date"] = pd.to_datetime(bal_sorted["date"]).dt.normalize()
    grid["date"] = pd.to_datetime(grid["date"]).dt.normalize()

    eod = pd.merge_asof(
        grid,
        bal_sorted,
        on="date",
        by="account_id",
        direction="backward",
    )

    # ---- Join account metadata ---
    meta_cols = ["account_id", "borrower_id", "household_id", "account_type", "credit_limit"]
    eod = eod.merge(accounts_df[meta_cols], on="account_id", how="left")

    eod["balance"] = eod["balance"].astype(np.float64)
    eod["credit_limit"] = eod["credit_limit"].astype(np.float64)
    return eod[
        ["account_id", "borrower_id", "household_id", "account_type",
         "date", "balance", "credit_limit"]
    ].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Account-level derived variables
# ---------------------------------------------------------------------------

def compute_account_level_daily(eod_states_df: pd.DataFrame) -> pd.DataFrame:
    """Compute account-level derived financial variables for each banking day.

    Derived variables
    -----------------
    * ``available_funds`` = balance + credit_limit
      (credit_limit is 0 for non-credit accounts).
    * ``utilization``     = max(−balance / credit_limit, 0) for credit accounts
      (account_type == 2) with credit_limit > 0; else 0.
    * ``overdraft``       = min(available_funds, 0)  (≤ 0 by construction).

    Parameters
    ----------
    eod_states_df : pd.DataFrame
        Output of :func:`reconstruct_eod_states`.

    Returns
    -------
    pd.DataFrame
        Columns: account_id, borrower_id, household_id, date, balance,
        credit_limit, available_funds, utilization, overdraft.
    """
    if len(eod_states_df) == 0:
        return pd.DataFrame(
            columns=["account_id", "borrower_id", "household_id", "date",
                     "balance", "credit_limit", "available_funds",
                     "utilization", "overdraft"]
        )

    # Avoid .copy() – compute derived arrays then assemble once
    balance = eod_states_df["balance"].fillna(0.0).to_numpy(dtype=np.float64)
    credit_limit = eod_states_df["credit_limit"].fillna(0.0).to_numpy(dtype=np.float64)
    acct_type = eod_states_df["account_type"].to_numpy()

    available_funds = balance + credit_limit
    is_credit = (acct_type == 2) & (credit_limit > 0.0)
    utilization = np.where(
        is_credit,
        np.maximum(-balance / np.where(credit_limit > 0, credit_limit, 1e-9), 0.0),
        0.0,
    )
    overdraft = np.minimum(available_funds, 0.0)

    return pd.DataFrame({
        "account_id":      eod_states_df["account_id"].to_numpy(),
        "borrower_id":     eod_states_df["borrower_id"].to_numpy(),
        "household_id":    eod_states_df["household_id"].to_numpy(),
        "date":            eod_states_df["date"].to_numpy(),
        "balance":         balance,
        "credit_limit":    credit_limit,
        "available_funds": available_funds,
        "utilization":     utilization,
        "overdraft":       overdraft,
    })


# ---------------------------------------------------------------------------
# 3. Borrower-level aggregation
# ---------------------------------------------------------------------------

def aggregate_to_borrower_level(account_daily_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate account-level daily data to borrower level.

    Sums all per-account variables across accounts belonging to the same
    borrower on the same banking day, then derives:

    * ``gearing``            = credit_limit / max(available_funds, 1 000)
    * ``relative_overdraft`` = overdraft / credit_limit  (where credit_limit > 0)

    Parameters
    ----------
    account_daily_df : pd.DataFrame
        Output of :func:`compute_account_level_daily`.

    Returns
    -------
    pd.DataFrame
        Columns: borrower_id, date, balance, credit_limit, available_funds,
        overdraft, utilization, gearing, relative_overdraft.
    """
    if len(account_daily_df) == 0:
        return pd.DataFrame(
            columns=["borrower_id", "date", "balance", "credit_limit",
                     "available_funds", "overdraft", "utilization",
                     "gearing", "relative_overdraft"]
        )

    agg = (
        account_daily_df
        .groupby(["borrower_id", "date"], as_index=False)
        .agg(
            balance=("balance", "sum"),
            credit_limit=("credit_limit", "sum"),
            available_funds=("available_funds", "sum"),
            overdraft=("overdraft", "sum"),
            utilization=("utilization", "sum"),
        )
    )

    cl = agg["credit_limit"].astype(np.float64)
    af = agg["available_funds"].astype(np.float64)
    od = agg["overdraft"].astype(np.float64)

    agg["gearing"] = cl / np.maximum(af, 1_000.0)
    agg["relative_overdraft"] = np.where(cl > 0.0, od / cl, 0.0).astype(np.float64)

    for col in ["balance", "credit_limit", "available_funds", "overdraft", "utilization"]:
        agg[col] = agg[col].astype(np.float64)

    return agg[
        ["borrower_id", "date", "balance", "credit_limit", "available_funds",
         "overdraft", "utilization", "gearing", "relative_overdraft"]
    ].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. Household-level aggregation
# ---------------------------------------------------------------------------

def aggregate_to_household_level(
    account_daily_df: pd.DataFrame,
    accounts_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate account-level daily data to household level, then map to borrowers.

    Aggregation mirrors :func:`aggregate_to_borrower_level` but sums across
    all accounts in the same household.  The result is joined back so that
    every borrower receives the household-level variables for their household.
    All output columns are prefixed with ``'hh_'``.

    Parameters
    ----------
    account_daily_df : pd.DataFrame
        Output of :func:`compute_account_level_daily`.
    accounts_df : pd.DataFrame
        Account metadata containing ``household_id`` and ``borrower_id``.

    Returns
    -------
    pd.DataFrame
        Columns: borrower_id, date, hh_balance, hh_credit_limit,
        hh_available_funds, hh_overdraft, hh_utilization,
        hh_gearing, hh_relative_overdraft.
    """
    output_cols = [
        "borrower_id", "date",
        "hh_balance", "hh_credit_limit", "hh_available_funds",
        "hh_overdraft", "hh_utilization", "hh_gearing", "hh_relative_overdraft",
    ]
    if len(account_daily_df) == 0 or len(accounts_df) == 0:
        return pd.DataFrame(columns=output_cols)

    # ---- Household-level sums ---
    agg = (
        account_daily_df
        .groupby(["household_id", "date"], as_index=False)
        .agg(
            hh_balance=("balance", "sum"),
            hh_credit_limit=("credit_limit", "sum"),
            hh_available_funds=("available_funds", "sum"),
            hh_overdraft=("overdraft", "sum"),
            hh_utilization=("utilization", "sum"),
        )
    )

    cl = agg["hh_credit_limit"].astype(np.float64)
    af = agg["hh_available_funds"].astype(np.float64)
    od = agg["hh_overdraft"].astype(np.float64)

    agg["hh_gearing"] = cl / np.maximum(af, 1_000.0)
    agg["hh_relative_overdraft"] = np.where(cl > 0.0, od / cl, 0.0).astype(np.float64)

    for col in ["hh_balance", "hh_credit_limit", "hh_available_funds",
                "hh_overdraft", "hh_utilization"]:
        agg[col] = agg[col].astype(np.float64)

    # ---- Map household → all borrowers in that household ---
    hh_borrow_map = (
        accounts_df[["household_id", "borrower_id"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    result = hh_borrow_map.merge(agg, on="household_id", how="inner")
    result = result.drop(columns=["household_id"])

    return result[output_cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 5. Sampling date assignment
# ---------------------------------------------------------------------------

def assign_sampling_dates(
    borrower_ids: List[int],
    banking_days: pd.DatetimeIndex,
    month_end_days: pd.Series,
    min_lookback_months: int = 12,
    min_forward_months: int = 12,
    dev_start: str = "2017-08-31",
    dev_end: str = "2019-12-31",
    oot_start: str = "2021-01-31",
    oot_end: str = "2024-12-31",
    seed: int = 42,
) -> pd.DataFrame:
    """Assign one end-of-month sampling date per borrower (dev or OOT split).

    Each borrower is randomly allocated to the development (``'dev'``) or
    out-of-time (``'oot'``) sample (70 / 30 split by default).  A random
    valid month-end banking day is then drawn from the appropriate date window.

    A sampling date T is **valid** if:

    * T is a month-end banking day in the specified window.
    * At least *min_lookback_months* calendar months of banking days exist
      before T in *banking_days*.
    * At least *min_forward_months* calendar months of banking days exist
      after T in *banking_days*.

    Parameters
    ----------
    borrower_ids : list of int
        Borrower identifiers to process.
    banking_days : pd.DatetimeIndex
        Full set of banking days (from :func:`src.calendar_utils.generate_banking_days`).
    month_end_days : pd.Series
        Last banking day per month (from
        :func:`src.calendar_utils.get_month_end_banking_days`).
    min_lookback_months : int
        Minimum months of history required before T.
    min_forward_months : int
        Minimum months of future required after T.
    dev_start, dev_end : str
        ISO date strings bounding the development sampling window (inclusive).
    oot_start, oot_end : str
        ISO date strings bounding the OOT sampling window (inclusive).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: borrower_id, sampling_date, sample_type (``'dev'`` or ``'oot'``).

    Raises
    ------
    ValueError
        If no valid sampling dates exist in either the dev or OOT window.
    """
    if len(banking_days) == 0:
        raise ValueError("banking_days is empty.")
    if len(month_end_days) == 0:
        raise ValueError("month_end_days is empty.")

    rng = np.random.default_rng(seed)

    # Valid date range for T: must allow lookback and forward periods
    bd_start = pd.Timestamp(banking_days[0]).normalize()
    bd_end = pd.Timestamp(banking_days[-1]).normalize()
    lookback_cutoff = bd_start + pd.DateOffset(months=min_lookback_months)
    forward_cutoff = bd_end - pd.DateOffset(months=min_forward_months)

    # Extract month-end timestamps as a sorted list
    all_me = sorted(pd.Timestamp(d).normalize() for d in month_end_days.values)

    def _valid_dates(win_start: str, win_end: str) -> List[pd.Timestamp]:
        ws = pd.Timestamp(win_start).normalize()
        we = pd.Timestamp(win_end).normalize()
        return [
            d for d in all_me
            if ws <= d <= we and lookback_cutoff <= d <= forward_cutoff
        ]

    dev_dates = _valid_dates(dev_start, dev_end)
    oot_dates = _valid_dates(oot_start, oot_end)

    if not dev_dates:
        raise ValueError(
            f"No valid month-end banking days found in dev window "
            f"[{dev_start}, {dev_end}] after applying lookback/forward constraints."
        )
    if not oot_dates:
        raise ValueError(
            f"No valid month-end banking days found in OOT window "
            f"[{oot_start}, {oot_end}] after applying lookback/forward constraints."
        )

    dev_arr = np.array(dev_dates, dtype="datetime64[ns]")
    oot_arr = np.array(oot_dates, dtype="datetime64[ns]")

    n = len(borrower_ids)
    # 70 % dev, 30 % OOT
    shuffle_idx = rng.permutation(n)
    n_dev = int(np.round(0.70 * n))
    dev_mask = np.zeros(n, dtype=bool)
    dev_mask[shuffle_idx[:n_dev]] = True

    records: list[dict] = []
    for i, b_id in enumerate(borrower_ids):
        if dev_mask[i]:
            chosen = pd.Timestamp(rng.choice(dev_arr))
            sample_type = "dev"
        else:
            chosen = pd.Timestamp(rng.choice(oot_arr))
            sample_type = "oot"
        records.append(
            {"borrower_id": int(b_id), "sampling_date": chosen,
             "sample_type": sample_type}
        )

    df = pd.DataFrame(records)
    df["borrower_id"] = df["borrower_id"].astype(np.int64)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 6. Default label creation
# ---------------------------------------------------------------------------

def create_default_labels(
    sampling_df: pd.DataFrame,
    defaults_df: pd.DataFrame,
    horizon_months: int = 12,
) -> pd.DataFrame:
    """Attach a binary default label to each (borrower, sampling_date) row.

    A borrower is labelled Y = 1 if any entry in *defaults_df* has
    ``default_date ∈ (T, T + horizon_months]``, where T is the borrower's
    ``sampling_date``.

    Parameters
    ----------
    sampling_df : pd.DataFrame
        Output of :func:`assign_sampling_dates` with columns borrower_id and
        sampling_date.
    defaults_df : pd.DataFrame
        Default events with columns borrower_id and default_date.
    horizon_months : int
        Length of the forward observation window in calendar months.

    Returns
    -------
    pd.DataFrame
        *sampling_df* with an additional integer column ``default_label``
        (0 or 1).
    """
    if len(sampling_df) == 0:
        return sampling_df.assign(default_label=pd.Series(dtype=np.int8))

    result = sampling_df.copy()
    result["sampling_date"] = pd.to_datetime(result["sampling_date"]).dt.normalize()

    if len(defaults_df) == 0:
        result["default_label"] = np.int8(0)
        return result

    def_df = defaults_df[["borrower_id", "default_date"]].copy()
    def_df["default_date"] = pd.to_datetime(def_df["default_date"]).dt.normalize()

    # Cross-join sampling with defaults (only matching borrowers)
    merged = result[["borrower_id", "sampling_date"]].merge(
        def_df, on="borrower_id", how="left"
    )

    horizon_offsets = merged["sampling_date"].map(
        lambda t: t + pd.DateOffset(months=horizon_months)
    )
    in_window = (
        (merged["default_date"] > merged["sampling_date"])
        & (merged["default_date"] <= horizon_offsets)
    )

    # Aggregate: any default in window per (borrower_id, sampling_date)?
    merged["_in_window"] = in_window.fillna(False).astype(bool)
    label_map = (
        merged.groupby(["borrower_id", "sampling_date"])["_in_window"]
        .any()
        .astype(np.int8)
        .reset_index()
        .rename(columns={"_in_window": "default_label"})
    )

    result = result.merge(label_map, on=["borrower_id", "sampling_date"], how="left")
    result["default_label"] = result["default_label"].fillna(0).astype(np.int8)
    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 7–8. Winsorization
# ---------------------------------------------------------------------------

def winsorize_series(
    series: pd.Series,
    lower_pct: float = 0.02,
    upper_pct: float = 0.98,
) -> Tuple[float, float]:
    """Compute winsorization bounds from a training sample.

    Parameters
    ----------
    series : pd.Series
        Numeric series (training data only).
    lower_pct : float
        Lower quantile threshold (0 ≤ lower_pct < upper_pct).
    upper_pct : float
        Upper quantile threshold (lower_pct < upper_pct ≤ 1).

    Returns
    -------
    tuple of (float, float)
        ``(lower_bound, upper_bound)`` to be stored and later passed to
        :func:`apply_winsorization`.

    Raises
    ------
    ValueError
        If the quantile bounds are out of range or inverted.
    """
    if not 0.0 <= lower_pct < upper_pct <= 1.0:
        raise ValueError(
            f"Require 0 ≤ lower_pct < upper_pct ≤ 1, "
            f"got lower_pct={lower_pct}, upper_pct={upper_pct}."
        )
    clean = series.dropna().astype(np.float64)
    if len(clean) == 0:
        return (np.nan, np.nan)
    lower = float(clean.quantile(lower_pct))
    upper = float(clean.quantile(upper_pct))
    return (lower, upper)


def apply_winsorization(
    df: pd.DataFrame,
    variable_cols: List[str],
    winsor_thresholds: Dict[str, Tuple[float, float]],
) -> pd.DataFrame:
    """Apply pre-computed winsorization bounds to specified columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame whose columns are to be clipped.
    variable_cols : list of str
        Column names to winsorize (others are left unchanged).
    winsor_thresholds : dict
        Mapping ``{column_name: (lower_bound, upper_bound)}`` produced by
        :func:`winsorize_series` on training data.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with the specified columns clipped to their bounds.
        Columns in *variable_cols* that are absent from *winsor_thresholds*
        are left unchanged.
    """
    df = df.copy()
    for col in variable_cols:
        if col not in df.columns:
            continue
        if col not in winsor_thresholds:
            continue
        lo, hi = winsor_thresholds[col]
        if pd.isna(lo) or pd.isna(hi):
            continue
        df[col] = df[col].astype(np.float64).clip(lower=float(lo), upper=float(hi))
    return df

"""
Synthetic retail banking data generator for Probability of Default (PD) modelling.

Mimics the data structure of the paper comparing scalar logistic regression,
XGBoost, and functional logistic regression for retail banking PD prediction.
All underlying proprietary data is replaced by fully synthetic data generated
from parameterised stochastic processes.

Accounts
--------
Each borrower holds 2–5 accounts.  Account types follow the paper's coding:
  1 = deposit   2 = credit   3 = loan   4 = investment
Credit accounts carry a credit limit and negative balances (drawn funds).

Balances
--------
Monthly end-of-month balance snapshots are generated for every account using
AR(1) dynamics with type-specific parameters.  The preprocessing layer
(src/preprocessing.py) reconstructs end-of-banking-day states via LOCF.

Defaults
--------
Default events occur at ~3–5 % annual rate.  Each borrower's monthly default
probability is scaled by a stress factor derived from credit-card utilisation.
Once a borrower defaults, they leave the at-risk pool (no repeat defaults).
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_synthetic_banking_data(
    n_borrowers: int = 3000,
    seed: int = 42,
    start_date: str = "2016-01-01",
    end_date: str = "2025-01-01",
) -> dict:
    """Generate synthetic retail banking data for PD modelling.

    Parameters
    ----------
    n_borrowers : int
        Number of distinct borrowers to simulate.
    seed : int
        NumPy random seed for reproducibility.
    start_date : str
        Earliest calendar date in ISO format (YYYY-MM-DD).
    end_date : str
        Latest calendar date in ISO format (YYYY-MM-DD).

    Returns
    -------
    dict with keys:
        ``'accounts'``  : pd.DataFrame – account metadata
        ``'balances'``  : pd.DataFrame – monthly end-of-month balance events
        ``'defaults'``  : pd.DataFrame – observed default events
    """
    rng = np.random.default_rng(seed)
    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date)

    if end_dt <= start_dt:
        raise ValueError("end_date must be strictly after start_date.")
    if n_borrowers < 1:
        raise ValueError("n_borrowers must be at least 1.")

    # End-of-month dates spanning the full simulation horizon
    month_ends: pd.DatetimeIndex = pd.date_range(start=start_dt, end=end_dt, freq="ME")

    borrower_ids = np.arange(1, n_borrowers + 1, dtype=np.int64)
    household_ids = _assign_households(n_borrowers, rng)

    accounts_df = _create_accounts(borrower_ids, household_ids, start_dt, end_dt, rng)
    balances_df = _generate_balances(accounts_df, month_ends, rng)
    defaults_df = _generate_defaults(accounts_df, balances_df, borrower_ids, month_ends, rng)

    return {"accounts": accounts_df, "balances": balances_df, "defaults": defaults_df}


def save_synthetic_data(
    data_dict: dict,
    output_dir: str = "data/raw",
) -> None:
    """Save the three synthetic DataFrames to CSV files.

    Parameters
    ----------
    data_dict : dict
        Dictionary returned by :func:`generate_synthetic_banking_data`.
    output_dir : str
        Directory in which to write the CSV files.  Created if absent.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    data_dict["accounts"].to_csv(out / "accounts.csv", index=False)
    data_dict["balances"].to_csv(out / "balances.csv", index=False)
    data_dict["defaults"].to_csv(out / "defaults.csv", index=False)

    print(
        f"Saved {len(data_dict['accounts'])} accounts, "
        f"{len(data_dict['balances'])} balance records, "
        f"and {len(data_dict['defaults'])} default events to '{out}'."
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _assign_households(n_borrowers: int, rng: np.random.Generator) -> np.ndarray:
    """Randomly assign borrowers to households of size 1–3.

    Parameters
    ----------
    n_borrowers : int
        Total number of borrowers.
    rng : np.random.Generator
        Seeded random number generator.

    Returns
    -------
    np.ndarray of shape (n_borrowers,) with integer household IDs (1-based).
    """
    household_ids = np.empty(n_borrowers, dtype=np.int64)
    h_id = 1
    i = 0
    while i < n_borrowers:
        size = int(rng.integers(1, 4))  # 1, 2, or 3
        end = min(i + size, n_borrowers)
        household_ids[i:end] = h_id
        h_id += 1
        i = end
    return household_ids


def _credit_limit_sample(rng: np.random.Generator) -> float:
    """Draw a realistic credit limit (DKK) from a discrete set."""
    options = np.array(
        [10_000, 20_000, 30_000, 50_000, 75_000, 100_000, 150_000, 200_000],
        dtype=np.float64,
    )
    return float(rng.choice(options))


def _create_accounts(
    borrower_ids: np.ndarray,
    household_ids: np.ndarray,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Create account-level metadata for all borrowers.

    Each borrower receives 2–5 accounts.  Every borrower is guaranteed at
    least one deposit (type 1) and one credit account (type 2); remaining
    slots are filled with randomly chosen types.

    Parameters
    ----------
    borrower_ids : np.ndarray
        1-D array of borrower identifiers.
    household_ids : np.ndarray
        Household assignment for each borrower (same length as borrower_ids).
    start_dt, end_dt : pd.Timestamp
        Simulation date bounds.
    rng : np.random.Generator
        Seeded random number generator.

    Returns
    -------
    pd.DataFrame with columns: account_id, borrower_id, household_id,
    account_type, open_date, credit_limit.
    """
    n_borrowers = len(borrower_ids)
    # Maximum offset for open_date: first half of the simulation window
    max_open_offset = max(int((end_dt - start_dt).days * 0.50), 1)

    records = []
    account_id = 1

    for idx in range(n_borrowers):
        b_id = int(borrower_ids[idx])
        h_id = int(household_ids[idx])

        n_accounts = int(rng.integers(2, 6))  # 2, 3, 4, or 5
        # Guaranteed base types; extras chosen uniformly from {1,2,3,4}
        base_types = [1, 2]
        extra_types = rng.integers(1, 5, size=n_accounts - 2).tolist() if n_accounts > 2 else []
        acct_types = rng.permutation(base_types + extra_types).tolist()

        for acct_type in acct_types:
            open_offset = int(rng.integers(0, max_open_offset))
            open_date = start_dt + pd.Timedelta(days=open_offset)

            credit_limit = _credit_limit_sample(rng) if acct_type == 2 else 0.0

            records.append(
                {
                    "account_id": account_id,
                    "borrower_id": b_id,
                    "household_id": h_id,
                    "account_type": int(acct_type),
                    "open_date": open_date,
                    "credit_limit": credit_limit,
                }
            )
            account_id += 1

    df = pd.DataFrame(records)
    df["account_id"] = df["account_id"].astype(np.int64)
    df["borrower_id"] = df["borrower_id"].astype(np.int64)
    df["household_id"] = df["household_id"].astype(np.int64)
    df["account_type"] = df["account_type"].astype(np.int64)
    df["credit_limit"] = df["credit_limit"].astype(np.float64)
    return df


def _ar1_series(
    n: int,
    alpha: float,
    mu: float,
    sigma: float,
    x0: float,
    rng: np.random.Generator,
    lo: float = -np.inf,
    hi: float = np.inf,
) -> np.ndarray:
    """Simulate an AR(1) process X_t = α X_{t-1} + (1-α) μ + ε_t.

    Parameters
    ----------
    n : int
        Number of time steps.
    alpha : float
        AR(1) persistence coefficient (0 < α < 1).
    mu : float
        Long-run mean.
    sigma : float
        Noise standard deviation.
    x0 : float
        Initial value.
    rng : np.random.Generator
        Random number generator.
    lo, hi : float
        Optional clipping bounds applied at each step.

    Returns
    -------
    np.ndarray of shape (n,) in float64.
    """
    eps = rng.normal(0.0, sigma, n).astype(np.float64)
    x = np.empty(n, dtype=np.float64)
    x[0] = np.clip(x0, lo, hi)
    for t in range(1, n):
        x[t] = np.clip(alpha * x[t - 1] + (1.0 - alpha) * mu + eps[t], lo, hi)
    return x


def _loan_series(
    n: int,
    alpha: float,
    payment: float,
    sigma: float,
    x0: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate an amortising loan balance: X_t = max(α X_{t-1} - payment + ε, 0).

    Parameters
    ----------
    n : int
        Number of periods.
    alpha : float
        AR(1) coefficient (≈1, slight mean-reversion dampening).
    payment : float
        Monthly payment that reduces the principal.
    sigma : float
        Payment noise standard deviation.
    x0 : float
        Initial (opening) loan balance.
    rng : np.random.Generator

    Returns
    -------
    np.ndarray of shape (n,) in float64.
    """
    eps = rng.normal(0.0, sigma, n).astype(np.float64)
    x = np.empty(n, dtype=np.float64)
    x[0] = float(x0)
    for t in range(1, n):
        x[t] = max(alpha * x[t - 1] - payment + eps[t], 0.0)
    return x


def _generate_balances(
    accounts_df: pd.DataFrame,
    month_ends: pd.DatetimeIndex,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate monthly end-of-month balance events for every account.

    Balance dynamics are AR(1) with type-specific parameters:
      * Type 1 (deposit):    positive balance, strong mean-reversion.
      * Type 2 (credit):     negative balance clipped to [−credit_limit, 0].
      * Type 3 (loan):       amortising balance, floored at zero.
      * Type 4 (investment): positive, slow-drifting AR(1).

    Parameters
    ----------
    accounts_df : pd.DataFrame
        Account metadata returned by :func:`_create_accounts`.
    month_ends : pd.DatetimeIndex
        Sequence of end-of-month dates for the simulation period.
    rng : np.random.Generator

    Returns
    -------
    pd.DataFrame with columns: account_id, date, balance.
    """
    records = []

    for row in accounts_df.itertuples(index=False):
        a_id: int = row.account_id
        acct_type: int = row.account_type
        open_date: pd.Timestamp = row.open_date
        credit_limit: float = row.credit_limit

        # Restrict to month-ends on or after open_date
        active_ends = month_ends[month_ends >= open_date]
        n = len(active_ends)
        if n == 0:
            continue

        if acct_type == 1:  # deposit
            mu = rng.uniform(20_000.0, 120_000.0)
            sigma = rng.uniform(3_000.0, 20_000.0)
            alpha = rng.uniform(0.88, 0.97)
            x0 = mu + rng.normal(0.0, sigma)
            balances = _ar1_series(n, alpha, mu, sigma, x0, rng, lo=0.0)

        elif acct_type == 2:  # credit (drawn balance is negative)
            alpha = rng.uniform(0.85, 0.95)
            init_util = rng.uniform(0.05, 0.85)
            mu = -credit_limit * rng.uniform(0.20, 0.65)
            sigma = credit_limit * rng.uniform(0.04, 0.12)
            x0 = -credit_limit * init_util
            balances = _ar1_series(
                n, alpha, mu, sigma, x0, rng, lo=-credit_limit, hi=0.0
            )

        elif acct_type == 3:  # loan
            loan_amount = rng.uniform(80_000.0, 600_000.0)
            term_months = rng.integers(60, 361)  # 5–30 years
            payment = loan_amount / float(term_months)
            sigma = loan_amount * rng.uniform(0.005, 0.015)
            alpha = 0.999  # nearly-unit-root; amortisation drives the drift
            balances = _loan_series(n, alpha, payment, sigma, loan_amount, rng)

        else:  # investment (type 4)
            mu = rng.uniform(30_000.0, 300_000.0)
            drift = rng.uniform(100.0, 1_000.0)  # monthly appreciation
            sigma = rng.uniform(5_000.0, 40_000.0)
            alpha = rng.uniform(0.92, 0.99)
            x0 = mu * rng.uniform(0.6, 1.4)
            raw = _ar1_series(n, alpha, mu, sigma, x0, rng, lo=0.0)
            # Overlay mild upward trend
            trend = drift * np.arange(n, dtype=np.float64)
            balances = raw + trend

        for t, date in enumerate(active_ends):
            records.append(
                {"account_id": a_id, "date": date, "balance": float(balances[t])}
            )

    if not records:
        return pd.DataFrame(columns=["account_id", "date", "balance"])

    df = pd.DataFrame(records)
    df["account_id"] = df["account_id"].astype(np.int64)
    df["balance"] = df["balance"].astype(np.float64)
    return df.reset_index(drop=True)


# Base monthly default probability → ≈ 9.6 % annual for a zero-stress borrower
# (higher than paper to ensure sufficient defaults with smaller synthetic sample)
_BASE_MONTHLY_PD: float = 0.008


def _generate_defaults(
    accounts_df: pd.DataFrame,
    balances_df: pd.DataFrame,
    borrower_ids: np.ndarray,
    month_ends: pd.DatetimeIndex,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate default events at a ~3–5 % annual rate.

    The monthly default probability of each borrower is:
        p(t) = p_base × (1 + 5 × max(utilisation(t) − 0.70, 0))

    where ``utilisation`` is the credit-weighted average utilisation across
    all of the borrower's credit accounts (type 2).  Borrowers with no credit
    accounts carry only the base probability.  Once a borrower defaults, they
    are removed from the at-risk pool (no repeat defaults).

    The actual default date is drawn uniformly from the 30 calendar days
    following the month-end trigger point.

    Parameters
    ----------
    accounts_df : pd.DataFrame
        Account metadata.
    balances_df : pd.DataFrame
        Monthly balance events.
    borrower_ids : np.ndarray
        1-D array of all borrower IDs.
    month_ends : pd.DatetimeIndex
        Chronological sequence of end-of-month dates.
    rng : np.random.Generator

    Returns
    -------
    pd.DataFrame with columns: borrower_id, default_date.
    """
    # --- Build borrower-level monthly utilisation table ---
    credit_meta = accounts_df.loc[
        accounts_df["account_type"] == 2, ["account_id", "borrower_id", "credit_limit"]
    ]

    borrow_util: pd.DataFrame | None = None

    if len(credit_meta) > 0 and len(balances_df) > 0:
        cbal = balances_df.merge(credit_meta, on="account_id", how="inner")
        cbal = cbal[cbal["credit_limit"] > 0.0].copy()

        if len(cbal) > 0:
            # Account-level utilisation (clipped to [0, 1] for the stress factor)
            cbal["util"] = np.clip(
                -cbal["balance"] / cbal["credit_limit"], 0.0, 1.0
            )
            cbal["util_weighted"] = cbal["util"] * cbal["credit_limit"]

            agg = cbal.groupby(["borrower_id", "date"], as_index=False).agg(
                util_w_sum=("util_weighted", "sum"),
                cl_sum=("credit_limit", "sum"),
            )
            agg["utilization"] = agg["util_w_sum"] / agg["cl_sum"].clip(lower=1.0)
            agg["stress_mult"] = 1.0 + 5.0 * np.maximum(agg["utilization"] - 0.70, 0.0)
            agg["default_prob"] = (_BASE_MONTHLY_PD * agg["stress_mult"]).clip(upper=0.50)
            borrow_util = agg[["borrower_id", "date", "default_prob"]].copy()

    # --- Process month by month ---
    defaulted: set[int] = set()
    records: list[dict] = []

    for month_end in month_ends:
        remaining = borrower_ids[~np.isin(borrower_ids, list(defaulted))]
        if len(remaining) == 0:
            break

        # Attach per-borrower default probability for this month
        if borrow_util is not None:
            month_probs = borrow_util[borrow_util["date"] == month_end].set_index(
                "borrower_id"
            )["default_prob"]
        else:
            month_probs = pd.Series(dtype=np.float64)

        probs = np.where(
            np.isin(remaining, month_probs.index),
            month_probs.reindex(remaining).fillna(_BASE_MONTHLY_PD).values,
            _BASE_MONTHLY_PD,
        )

        draws = rng.random(len(remaining))
        new_defaulters = remaining[draws < probs]

        for b_id in new_defaulters:
            day_offset = int(rng.integers(1, 31))
            default_date = month_end + pd.Timedelta(days=day_offset)
            records.append(
                {"borrower_id": int(b_id), "default_date": default_date}
            )
            defaulted.add(int(b_id))

    if not records:
        return pd.DataFrame(columns=["borrower_id", "default_date"])

    df = pd.DataFrame(records)
    df["borrower_id"] = df["borrower_id"].astype(np.int64)
    return df.reset_index(drop=True)

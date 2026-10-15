import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from data.synthetic_generator import generate_synthetic_banking_data
from src.calendar_utils import generate_banking_days, get_month_end_banking_days
from src.preprocessing import (
    aggregate_to_borrower_level,
    aggregate_to_household_level,
    assign_sampling_dates,
    compute_account_level_daily,
    create_default_labels,
    reconstruct_eod_states,
)


@pytest.fixture(scope="session")
def synthetic_data():
    """Small synthetic dataset with 100 borrowers (seed=0)."""
    return generate_synthetic_banking_data(n_borrowers=100, seed=0)


@pytest.fixture(scope="session")
def banking_days():
    """Banking days from 2015-01-01 to 2025-12-31."""
    return generate_banking_days("2015-01-01", "2025-12-31")


@pytest.fixture(scope="session")
def month_end_days(banking_days):
    """Last banking day per month for 2015-01-01 to 2025-12-31."""
    return get_month_end_banking_days("2015-01-01", "2025-12-31")


@pytest.fixture(scope="session")
def small_sample_data(synthetic_data, banking_days, month_end_days):
    """Full preprocessing pipeline on the 100-borrower synthetic dataset.

    Returns a dict with all intermediate DataFrames:
    - accounts_df, balances_df, defaults_df  (raw synthetic tables)
    - eod_states      (reconstruct_eod_states output)
    - account_daily   (compute_account_level_daily output)
    - borrower_daily  (aggregate_to_borrower_level output)
    - hh_daily        (aggregate_to_household_level output)
    - sampling_df     (assign_sampling_dates output)
    - labeled_df      (create_default_labels output)
    """
    accounts_df = synthetic_data["accounts"]
    balances_df = synthetic_data["balances"]
    defaults_df = synthetic_data["defaults"]

    eod_states = reconstruct_eod_states(accounts_df, balances_df, banking_days)
    account_daily = compute_account_level_daily(eod_states)
    borrower_daily = aggregate_to_borrower_level(account_daily)
    hh_daily = aggregate_to_household_level(account_daily, accounts_df)

    borrower_ids = accounts_df["borrower_id"].unique().tolist()
    sampling_df = assign_sampling_dates(
        borrower_ids,
        banking_days,
        month_end_days,
        dev_start="2017-08-31",
        dev_end="2019-12-31",
        oot_start="2021-01-31",
        oot_end="2024-12-31",
        seed=0,
    )
    labeled_df = create_default_labels(sampling_df, defaults_df, horizon_months=12)

    return {
        "accounts_df": accounts_df,
        "balances_df": balances_df,
        "defaults_df": defaults_df,
        "eod_states": eod_states,
        "account_daily": account_daily,
        "borrower_daily": borrower_daily,
        "hh_daily": hh_daily,
        "sampling_df": sampling_df,
        "labeled_df": labeled_df,
    }

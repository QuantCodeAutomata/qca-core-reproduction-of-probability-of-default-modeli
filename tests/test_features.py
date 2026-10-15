"""Tests for src/feature_engineering.py."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    _make_clamped_knots,
    build_functional_predictors,
    build_scalar_features,
    compute_inner_product_matrix,
    compute_roughness_penalty_matrix,
    compute_spline_basis_matrix,
)


# ---------------------------------------------------------------------------
# Scalar features – count, naming and spot-value equality
# ---------------------------------------------------------------------------

def test_scalar_features_count(small_sample_data, banking_days, month_end_days):
    """build_scalar_features must produce exactly 50 columns (5 vars × 2 levels × 5 horizons)."""
    features = build_scalar_features(
        small_sample_data["borrower_daily"],
        small_sample_data["hh_daily"],
        small_sample_data["sampling_df"],
        banking_days,
        month_end_days,
    )
    assert features.shape[1] == 50, (
        f"Expected 50 feature columns, got {features.shape[1]}"
    )


def test_scalar_feature_naming(small_sample_data, banking_days, month_end_days):
    """Feature columns must follow the naming pattern '{level}_{variable}_M{horizon}'."""
    from src.config import CONFIG

    variables = CONFIG["scalar_variables"]
    horizons = CONFIG["scalar_horizons"]
    levels = ["bw_", "hh_"]

    expected = {
        f"{level}{var}_M{h}"
        for var in variables
        for level in levels
        for h in horizons
    }

    features = build_scalar_features(
        small_sample_data["borrower_daily"],
        small_sample_data["hh_daily"],
        small_sample_data["sampling_df"],
        banking_days,
        month_end_days,
    )
    actual = set(features.columns)
    assert actual == expected, (
        f"Feature column mismatch.\n"
        f"Missing: {expected - actual}\n"
        f"Extra:   {actual - expected}"
    )


def test_scalar_m0_equals_spot(small_sample_data, banking_days, month_end_days):
    """M=0 features must equal the spot (same-day) value in the daily DataFrame."""
    borrower_daily = small_sample_data["borrower_daily"]
    hh_daily = small_sample_data["hh_daily"]
    sampling_df = small_sample_data["sampling_df"]

    features = build_scalar_features(
        borrower_daily,
        hh_daily,
        sampling_df,
        banking_days,
        month_end_days,
        variables=["available_funds"],
        horizons=[0],
    )

    bw_idx = (
        borrower_daily.copy()
        .assign(date=lambda d: pd.to_datetime(d["date"]).dt.normalize())
        .set_index(["borrower_id", "date"])
    )

    for srow in sampling_df.itertuples(index=False):
        b_id = int(srow.borrower_id)
        T = pd.Timestamp(srow.sampling_date).normalize()

        try:
            spot = float(bw_idx.at[(b_id, T), "available_funds"])
        except KeyError:
            continue  # no data at T – skip this borrower

        if b_id not in features.index:
            continue

        feature_val = float(features.at[b_id, "bw_available_funds_M0"])
        if np.isnan(spot) or np.isnan(feature_val):
            continue

        assert abs(feature_val - spot) < 1e-9, (
            f"Borrower {b_id}: bw_available_funds_M0={feature_val} ≠ spot={spot}"
        )


def test_scalar_features_index_is_borrower_id(small_sample_data, banking_days, month_end_days):
    """The index of the returned DataFrame must be borrower_id."""
    features = build_scalar_features(
        small_sample_data["borrower_daily"],
        small_sample_data["hh_daily"],
        small_sample_data["sampling_df"],
        banking_days,
        month_end_days,
    )
    assert features.index.name == "borrower_id", (
        f"Expected index name 'borrower_id', got '{features.index.name}'"
    )


# ---------------------------------------------------------------------------
# compute_spline_basis_matrix
# ---------------------------------------------------------------------------

def test_spline_basis_shape():
    """compute_spline_basis_matrix must return an array of shape (n_pts, n_basis)."""
    n_pts, n_basis, domain_end = 50, 7, 4.0
    t = np.linspace(0.01, domain_end, n_pts)
    B = compute_spline_basis_matrix(t, domain_end, n_basis)
    assert B.shape == (n_pts, n_basis), (
        f"Expected shape ({n_pts}, {n_basis}), got {B.shape}"
    )


def test_spline_basis_partition_of_unity():
    """Each row of the spline basis matrix must sum to approximately 1 (partition of unity).

    Clamped B-splines form a partition of unity inside the domain [0, domain_end].
    """
    domain_end, n_basis = 6.0, 8
    t = np.linspace(0.01, domain_end, 80)  # strictly interior points
    B = compute_spline_basis_matrix(t, domain_end, n_basis)
    row_sums = B.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-10), (
        f"Partition of unity violated; max deviation = {np.abs(row_sums - 1.0).max()}"
    )


def test_spline_basis_nonnegative():
    """B-spline basis functions must be non-negative everywhere in the domain."""
    domain_end, n_basis = 5.0, 7
    t = np.linspace(0.01, domain_end, 100)
    B = compute_spline_basis_matrix(t, domain_end, n_basis)
    assert np.all(B >= -1e-14), "Negative B-spline values detected"


def test_spline_basis_invalid_domain_raises():
    """domain_end ≤ 0 must raise ValueError."""
    with pytest.raises(ValueError):
        compute_spline_basis_matrix(np.array([0.5, 1.0]), 0.0, 5)


# ---------------------------------------------------------------------------
# compute_roughness_penalty_matrix
# ---------------------------------------------------------------------------

def test_roughness_penalty_symmetric():
    """The roughness penalty matrix S must be symmetric (S = Sᵀ)."""
    n_basis, domain_end = 7, 6.0
    knots = _make_clamped_knots(domain_end, n_basis)
    S = compute_roughness_penalty_matrix(knots, domain_end=domain_end)
    assert S.shape == (n_basis, n_basis), f"Expected square ({n_basis}×{n_basis}), got {S.shape}"
    assert np.allclose(S, S.T, atol=1e-12), "Roughness penalty matrix is not symmetric"


def test_roughness_penalty_psd():
    """The roughness penalty matrix S must be positive semi-definite.

    All eigenvalues must be ≥ −1e-10 (allowing for floating-point noise).
    """
    n_basis, domain_end = 8, 5.0
    knots = _make_clamped_knots(domain_end, n_basis)
    S = compute_roughness_penalty_matrix(knots, domain_end=domain_end)
    eigvals = np.linalg.eigvalsh(S)
    assert np.all(eigvals >= -1e-10), (
        f"Roughness penalty matrix is not PSD; min eigenvalue = {eigvals.min()}"
    )


def test_roughness_penalty_shape():
    """Roughness penalty shape must be (n_basis, n_basis)."""
    for n_basis in [5, 7, 10]:
        domain_end = float(n_basis)
        knots = _make_clamped_knots(domain_end, n_basis)
        S = compute_roughness_penalty_matrix(knots, domain_end=domain_end)
        assert S.shape == (n_basis, n_basis), (
            f"n_basis={n_basis}: expected ({n_basis}, {n_basis}), got {S.shape}"
        )


# ---------------------------------------------------------------------------
# compute_inner_product_matrix
# ---------------------------------------------------------------------------

def test_inner_product_matrix_shape():
    """compute_inner_product_matrix must return an array of shape (L, K)."""
    L, K, domain_end = 5, 8, 6.0
    phi_knots = _make_clamped_knots(domain_end, L)
    psi_knots = _make_clamped_knots(domain_end, K)

    phi_params = {"knots": phi_knots, "degree": 3}
    psi_params = {"knots": psi_knots, "degree": 3}

    J = compute_inner_product_matrix(phi_params, psi_params, domain_end=domain_end)
    assert J.shape == (L, K), f"Expected ({L}, {K}), got {J.shape}"


def test_inner_product_matrix_symmetric_when_same_basis():
    """When φ and ψ are the same basis, J must be symmetric."""
    n_basis, domain_end = 6, 4.0
    knots = _make_clamped_knots(domain_end, n_basis)
    params = {"knots": knots, "degree": 3}
    J = compute_inner_product_matrix(params, params, domain_end=domain_end)
    assert np.allclose(J, J.T, atol=1e-10), "J should be symmetric when φ == ψ"


def test_inner_product_matrix_missing_keys_raises():
    """Missing 'knots' or 'degree' keys must raise ValueError."""
    with pytest.raises(ValueError):
        compute_inner_product_matrix({"knots": np.array([0, 0, 0, 1, 1, 1])}, {"degree": 3})


# ---------------------------------------------------------------------------
# build_functional_predictors
# ---------------------------------------------------------------------------

def test_functional_predictor_shape(small_sample_data, banking_days, month_end_days):
    """build_functional_predictors must return coefficient matrix of shape (n_borrowers, n_basis)."""
    sampling_df = small_sample_data["sampling_df"].head(20).copy()
    borrower_daily = small_sample_data["borrower_daily"]
    hh_daily = small_sample_data["hh_daily"]

    M = 3
    n_basis_expected = max(6, M + 3)

    coeff_matrix, basis_info = build_functional_predictors(
        borrower_daily,
        hh_daily,
        sampling_df,
        banking_days,
        month_end_days,
        variable="available_funds",
        level="bw",
        M=M,
        lambda_x=0.0,
        basis_type="cubic_spline",
    )

    assert coeff_matrix.shape[0] == len(sampling_df), (
        f"Row count mismatch: expected {len(sampling_df)}, got {coeff_matrix.shape[0]}"
    )
    assert coeff_matrix.shape[1] == n_basis_expected, (
        f"Column count mismatch: expected {n_basis_expected}, got {coeff_matrix.shape[1]}"
    )


def test_functional_predictor_mean_spline_shape(small_sample_data, banking_days, month_end_days):
    """mean_spline basis must return coefficient matrix with exactly M columns."""
    sampling_df = small_sample_data["sampling_df"].head(10).copy()
    M = 6

    coeff_matrix, basis_info = build_functional_predictors(
        small_sample_data["borrower_daily"],
        small_sample_data["hh_daily"],
        sampling_df,
        banking_days,
        month_end_days,
        variable="utilization",
        level="bw",
        M=M,
        basis_type="mean_spline",
    )

    assert coeff_matrix.shape == (len(sampling_df), M), (
        f"Expected shape ({len(sampling_df)}, {M}), got {coeff_matrix.shape}"
    )
    assert basis_info["n_basis"] == M, f"Expected n_basis={M} in basis_info"


def test_functional_predictor_invalid_level_raises(small_sample_data, banking_days, month_end_days):
    """Invalid level must raise ValueError."""
    with pytest.raises(ValueError):
        build_functional_predictors(
            small_sample_data["borrower_daily"],
            small_sample_data["hh_daily"],
            small_sample_data["sampling_df"].head(5),
            banking_days,
            month_end_days,
            variable="gearing",
            level="xx",
            M=3,
        )

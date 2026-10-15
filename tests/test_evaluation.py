"""Tests for src/evaluation.py."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    compute_all_metrics,
    compute_auc,
    compute_brier_score,
    compute_focal_loss,
    compute_log_loss,
    compute_weighted_log_loss,
    delong_auc_variance,
    evaluate_model_suite,
    holm_bonferroni_adjust,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _balanced_labels(n=200, seed=0):
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=np.float64)
    y[: n // 2] = 1.0
    rng.shuffle(y)
    return y


def _imbalanced_labels(n=200, default_rate=0.05, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < default_rate).astype(np.float64)
    # Ensure at least one positive
    if y.sum() == 0:
        y[0] = 1.0
    return y


# ---------------------------------------------------------------------------
# compute_auc
# ---------------------------------------------------------------------------

def test_auc_perfect():
    """Perfect predictions must yield AUC = 1.0."""
    y = _balanced_labels(n=100)
    # Perfect: predict 1.0 for positives, 0.0 for negatives
    p = y.copy()
    auc = compute_auc(y, p)
    assert abs(auc - 1.0) < 1e-10, f"Expected AUC=1.0, got {auc}"


def test_auc_random():
    """Random predictions must yield AUC ≈ 0.5 (within 0.1 tolerance)."""
    rng = np.random.default_rng(99)
    y = _balanced_labels(n=500)
    p = rng.random(len(y))
    auc = compute_auc(y, p)
    assert 0.4 <= auc <= 0.6, f"Random AUC={auc} not near 0.5"


def test_auc_in_unit_interval():
    """AUC must always lie in [0, 1]."""
    rng = np.random.default_rng(42)
    y = _balanced_labels(n=200)
    p = rng.random(len(y))
    auc = compute_auc(y, p)
    assert 0.0 <= auc <= 1.0, f"AUC={auc} outside [0, 1]"


def test_auc_inverted_predictions():
    """Perfectly wrong predictions (all negatives ranked above all positives) yield AUC = 0."""
    y = _balanced_labels(n=100)
    p = 1.0 - y  # invert: predict 0 for positives, 1 for negatives
    auc = compute_auc(y, p)
    assert abs(auc - 0.0) < 1e-10, f"Expected AUC=0.0 for inverted predictions, got {auc}"


# ---------------------------------------------------------------------------
# compute_brier_score
# ---------------------------------------------------------------------------

def test_brier_score_perfect():
    """Perfect probability forecasts must give Brier score = 0."""
    y = _balanced_labels(n=100)
    score = compute_brier_score(y, y)
    assert abs(score) < 1e-14, f"Expected Brier=0 for perfect predictions, got {score}"


def test_brier_score_range():
    """Brier score must lie in [0, 1]."""
    rng = np.random.default_rng(0)
    y = _balanced_labels(n=200)
    p = rng.random(len(y))
    score = compute_brier_score(y, p)
    assert 0.0 <= score <= 1.0, f"Brier score={score} out of [0, 1]"


def test_brier_score_worst_case():
    """Completely wrong deterministic predictions give Brier score = 1."""
    y = np.array([1.0, 0.0, 1.0, 0.0])
    p = np.array([0.0, 1.0, 0.0, 1.0])
    score = compute_brier_score(y, p)
    assert abs(score - 1.0) < 1e-12, f"Expected Brier=1.0, got {score}"


def test_brier_score_constant_half():
    """Predicting 0.5 for all observations gives Brier score = 0.25."""
    y = _balanced_labels(n=100)
    p = np.full(len(y), 0.5)
    score = compute_brier_score(y, p)
    assert abs(score - 0.25) < 1e-12, f"Expected Brier=0.25 for constant 0.5 pred, got {score}"


# ---------------------------------------------------------------------------
# compute_log_loss
# ---------------------------------------------------------------------------

def test_log_loss_clipping():
    """Log loss must remain finite even when predictions are exactly 0 or 1."""
    y = np.array([0.0, 1.0, 0.0, 1.0])
    p_extreme = np.array([0.0, 1.0, 0.0, 1.0])  # exact 0s and 1s → log(0) risk
    loss = compute_log_loss(y, p_extreme)
    assert np.isfinite(loss), "Log loss must be finite even for p=0 or p=1"
    assert loss >= 0.0, "Log loss must be non-negative"


def test_log_loss_perfect_predictions():
    """Perfect probability forecasts must give log loss near 0."""
    y = _balanced_labels(n=100)
    # Near-perfect: 0.9999 for positives, 0.0001 for negatives
    p = np.where(y == 1, 0.9999, 0.0001)
    loss = compute_log_loss(y, p)
    assert loss < 0.01, f"Expected very small log loss for near-perfect predictions, got {loss}"


def test_log_loss_nonnegative():
    """Log loss must always be ≥ 0."""
    rng = np.random.default_rng(5)
    y = _balanced_labels(n=200)
    p = rng.random(len(y))
    loss = compute_log_loss(y, p)
    assert loss >= 0.0, f"Log loss must be non-negative, got {loss}"


# ---------------------------------------------------------------------------
# compute_weighted_log_loss
# ---------------------------------------------------------------------------

def test_weighted_log_loss_higher_for_defaults():
    """Weighted log loss must be higher than standard log loss for class-imbalanced data.

    Up-weighting the minority class (defaults) increases the loss relative
    to the standard (equal-weight) log loss.
    """
    rng = np.random.default_rng(42)
    n = 500
    y = _imbalanced_labels(n=n, default_rate=0.05, seed=42)
    # Predict the majority class for all → bad for defaults
    p = np.full(n, 0.05)

    standard = compute_log_loss(y, p)
    weighted = compute_weighted_log_loss(y, p)
    assert weighted > standard, (
        f"Weighted log loss ({weighted:.4f}) must exceed standard log loss "
        f"({standard:.4f}) for imbalanced data with majority-class predictions"
    )


def test_weighted_log_loss_explicit_w1():
    """Explicit w1 parameter must be respected."""
    y = np.array([0.0, 0.0, 0.0, 1.0])
    p = np.array([0.1, 0.1, 0.1, 0.9])
    loss1 = compute_weighted_log_loss(y, p, w1=1.0)  # equal weights
    loss_std = compute_log_loss(y, p)
    assert abs(loss1 - loss_std) < 1e-10, (
        "With w1=1.0, weighted log loss should equal standard log loss"
    )


def test_weighted_log_loss_all_same_class_raises():
    """All-positive or all-negative labels must raise ValueError when w1 is None."""
    y_all_zeros = np.zeros(10)
    p = np.full(10, 0.1)
    with pytest.raises(ValueError):
        compute_weighted_log_loss(y_all_zeros, p)


# ---------------------------------------------------------------------------
# compute_focal_loss
# ---------------------------------------------------------------------------

def test_focal_loss_nonnegative():
    """Focal loss must be ≥ 0 for any combination of predictions and labels."""
    rng = np.random.default_rng(11)
    y = _balanced_labels(n=200)
    p = rng.random(len(y))
    loss = compute_focal_loss(y, p, gamma=2.0, alpha=0.25)
    assert loss >= 0.0, f"Focal loss must be non-negative, got {loss}"


def test_focal_loss_gamma_zero_equals_bce():
    """With gamma=0, focal loss degenerates to (alpha-weighted) binary cross-entropy."""
    rng = np.random.default_rng(13)
    y = _balanced_labels(n=100)
    p = np.clip(rng.random(len(y)), 1e-6, 1 - 1e-6)
    alpha = 0.5

    focal = compute_focal_loss(y, p, gamma=0.0, alpha=alpha)
    # Manual weighted BCE with same alpha
    eps = 1e-15
    p_c = np.clip(p, eps, 1 - eps)
    bce_manual = -np.mean(
        alpha * y * np.log(p_c + eps)
        + (1 - alpha) * (1 - y) * np.log(1 - p_c + eps)
    )
    assert abs(focal - bce_manual) < 1e-10, (
        f"Focal loss with gamma=0 ({focal:.6f}) should equal alpha-weighted BCE ({bce_manual:.6f})"
    )


def test_focal_loss_extreme_probabilities():
    """Focal loss must be finite even with very small or very large predictions."""
    y = np.array([1.0, 0.0, 1.0, 0.0])
    p_extreme = np.array([0.0, 1.0, 1.0, 0.0])
    loss = compute_focal_loss(y, p_extreme)
    assert np.isfinite(loss), "Focal loss must be finite for extreme probabilities"


# ---------------------------------------------------------------------------
# DeLong paired AUC test
# ---------------------------------------------------------------------------

def test_delong_identical_predictions():
    """When both models produce identical predictions, delta_AUC must be 0."""
    rng = np.random.default_rng(0)
    y = _balanced_labels(n=100)
    p = rng.random(len(y))

    auc1, auc2, delta_auc, z_stat, p_val = delong_auc_variance(y, p, p)

    assert abs(auc1 - auc2) < 1e-12, f"AUC1 ({auc1}) ≠ AUC2 ({auc2}) for identical predictions"
    assert abs(delta_auc) < 1e-12, f"delta_AUC should be 0, got {delta_auc}"


def test_delong_perfect_vs_random():
    """Perfect model must have higher AUC than random; delta_AUC must be positive."""
    rng = np.random.default_rng(7)
    y = _balanced_labels(n=200)
    pred_perfect = y.copy()                # AUC = 1
    pred_random = rng.random(len(y))       # AUC ≈ 0.5

    auc1, auc2, delta_auc, z_stat, p_val = delong_auc_variance(y, pred_perfect, pred_random)

    assert auc1 > auc2, f"Perfect model AUC ({auc1}) should exceed random AUC ({auc2})"
    assert delta_auc > 0, f"delta_AUC should be positive, got {delta_auc}"
    assert p_val < 0.05, f"p-value should be significant, got {p_val}"


def test_delong_returns_five_values():
    """delong_auc_variance must return exactly 5 values."""
    y = _balanced_labels(n=50)
    p1 = np.random.default_rng(1).random(len(y))
    p2 = np.random.default_rng(2).random(len(y))
    result = delong_auc_variance(y, p1, p2)
    assert len(result) == 5, f"Expected 5 return values, got {len(result)}"


def test_delong_auc_values_in_unit_interval():
    """AUC values returned by delong_auc_variance must be in [0, 1]."""
    rng = np.random.default_rng(99)
    y = _balanced_labels(n=100)
    p1, p2 = rng.random(len(y)), rng.random(len(y))
    auc1, auc2, *_ = delong_auc_variance(y, p1, p2)
    assert 0.0 <= auc1 <= 1.0, f"AUC1={auc1} out of [0, 1]"
    assert 0.0 <= auc2 <= 1.0, f"AUC2={auc2} out of [0, 1]"


def test_delong_too_few_positives_raises():
    """delong_auc_variance must raise ValueError when n1 < 2 or n0 < 2."""
    y = np.array([1.0, 0.0, 0.0, 0.0])  # only 1 positive
    p = np.array([0.9, 0.2, 0.3, 0.1])
    with pytest.raises(ValueError):
        delong_auc_variance(y, p, p)


# ---------------------------------------------------------------------------
# holm_bonferroni_adjust
# ---------------------------------------------------------------------------

def test_holm_bonferroni_length():
    """Output must have the same length as the input p-value list."""
    p_values = [0.01, 0.05, 0.10, 0.20, 0.50]
    adjusted = holm_bonferroni_adjust(p_values)
    assert len(adjusted) == len(p_values), (
        f"Expected length {len(p_values)}, got {len(adjusted)}"
    )


def test_holm_bonferroni_adjusted_ge_raw():
    """Adjusted p-values must be ≥ the corresponding raw p-values."""
    p_values = [0.001, 0.010, 0.030, 0.150, 0.400]
    adjusted = holm_bonferroni_adjust(p_values)
    for raw, adj in zip(p_values, adjusted):
        assert adj >= raw - 1e-12, (
            f"Adjusted p-value ({adj}) < raw p-value ({raw})"
        )


def test_holm_bonferroni_capped_at_one():
    """Adjusted p-values must never exceed 1.0."""
    p_values = [0.40, 0.45, 0.50]
    adjusted = holm_bonferroni_adjust(p_values)
    assert np.all(adjusted <= 1.0), f"Some adjusted p-value > 1.0: {adjusted}"


def test_holm_bonferroni_empty_input():
    """Empty input must return an empty array."""
    adjusted = holm_bonferroni_adjust([])
    assert len(adjusted) == 0, "Expected empty array for empty input"


def test_holm_bonferroni_single_pvalue():
    """Single p-value must be capped at 1.0 (multiplier = 1)."""
    adjusted = holm_bonferroni_adjust([0.03])
    assert len(adjusted) == 1
    assert adjusted[0] == 0.03, f"Expected 0.03, got {adjusted[0]}"


def test_holm_bonferroni_monotone_after_sorting():
    """After Holm-Bonferroni, sorted adjusted p-values must be non-decreasing."""
    p_values = [0.001, 0.005, 0.02, 0.08, 0.3]
    adjusted = holm_bonferroni_adjust(p_values)
    sorted_adj = np.sort(adjusted)
    assert np.all(np.diff(sorted_adj) >= -1e-14), (
        "Sorted adjusted p-values must be non-decreasing"
    )


# ---------------------------------------------------------------------------
# evaluate_model_suite
# ---------------------------------------------------------------------------

def test_evaluate_model_suite_columns():
    """evaluate_model_suite must return a DataFrame with all expected metric columns."""
    y = _balanced_labels(n=200)
    rng = np.random.default_rng(0)
    preds = {
        "model_A": rng.random(len(y)),
        "model_B": rng.random(len(y)),
    }
    result = evaluate_model_suite(y, preds)
    expected_cols = {"auc", "brier", "log_loss", "weighted_log_loss", "focal_loss"}
    assert expected_cols.issubset(set(result.columns)), (
        f"Missing metric columns: {expected_cols - set(result.columns)}"
    )


def test_evaluate_model_suite_index_is_model_name():
    """The index of evaluate_model_suite output must contain model names."""
    y = _balanced_labels(n=100)
    rng = np.random.default_rng(1)
    preds = {"logistic": rng.random(len(y)), "xgboost": rng.random(len(y))}
    result = evaluate_model_suite(y, preds)
    assert set(result.index) == {"logistic", "xgboost"}, (
        f"Unexpected index: {list(result.index)}"
    )


def test_evaluate_model_suite_auc_in_range():
    """AUC values in the evaluate_model_suite output must be in [0, 1]."""
    y = _balanced_labels(n=200)
    rng = np.random.default_rng(2)
    preds = {"m1": rng.random(len(y)), "m2": rng.random(len(y))}
    result = evaluate_model_suite(y, preds)
    assert (result["auc"] >= 0.0).all() and (result["auc"] <= 1.0).all(), (
        f"AUC values out of [0, 1]: {result['auc'].tolist()}"
    )


# ---------------------------------------------------------------------------
# compute_all_metrics
# ---------------------------------------------------------------------------

def test_compute_all_metrics_keys():
    """compute_all_metrics must return a dict with all five expected metric keys."""
    y = _balanced_labels(n=100)
    rng = np.random.default_rng(3)
    p = rng.random(len(y))
    metrics = compute_all_metrics(y, p)

    expected_keys = {"auc", "brier", "log_loss", "weighted_log_loss", "focal_loss"}
    assert expected_keys == set(metrics.keys()), (
        f"Metric keys mismatch.\n"
        f"Missing: {expected_keys - set(metrics.keys())}\n"
        f"Extra:   {set(metrics.keys()) - expected_keys}"
    )


def test_compute_all_metrics_values_are_finite():
    """All values returned by compute_all_metrics must be finite floats."""
    y = _balanced_labels(n=100)
    rng = np.random.default_rng(4)
    p = rng.random(len(y))
    metrics = compute_all_metrics(y, p)
    for key, val in metrics.items():
        assert np.isfinite(val), f"Metric '{key}'={val} is not finite"


def test_compute_all_metrics_perfect_auc():
    """compute_all_metrics must report AUC=1.0 for perfect predictions."""
    y = _balanced_labels(n=100)
    p = y.copy()
    metrics = compute_all_metrics(y, p)
    assert abs(metrics["auc"] - 1.0) < 1e-10, (
        f"Expected AUC=1.0 for perfect predictions, got {metrics['auc']}"
    )

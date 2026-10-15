"""Evaluation metrics and statistical tests for probability-of-default models.

Implements standard binary classification metrics (AUC, Brier score, log loss,
weighted log loss, focal loss) together with the DeLong paired AUC test and
Holm-Bonferroni multiple-testing correction.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.stats
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------------
# Scalar metrics
# ---------------------------------------------------------------------------


def compute_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the ROC area under the curve.

    Parameters
    ----------
    y_true : np.ndarray of shape (n,)
        Binary ground-truth labels (0 or 1).
    y_pred : np.ndarray of shape (n,)
        Predicted default probabilities.

    Returns
    -------
    float
        ROC AUC in [0, 1].
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(roc_auc_score(y_true, y_pred))


def compute_brier_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the Brier score (mean squared error of probability forecasts).

    Parameters
    ----------
    y_true : np.ndarray of shape (n,)
        Binary ground-truth labels.
    y_pred : np.ndarray of shape (n,)
        Predicted default probabilities.

    Returns
    -------
    float
        Brier score: mean((y_pred - y_true)^2).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean((y_pred - y_true) ** 2))


def compute_log_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1e-15,
) -> float:
    """Compute binary cross-entropy (log loss) with probability clipping.

    Parameters
    ----------
    y_true : np.ndarray of shape (n,)
        Binary ground-truth labels.
    y_pred : np.ndarray of shape (n,)
        Predicted default probabilities, clipped to [eps, 1 - eps].
    eps : float, default 1e-15
        Clipping bound to avoid log(0).

    Returns
    -------
    float
        Mean log loss (non-negative).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    p = np.clip(np.asarray(y_pred, dtype=np.float64), eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))


def compute_weighted_log_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    w1: Optional[float] = None,
    eps: float = 1e-15,
) -> float:
    """Compute class-weighted binary log loss.

    Default observations (y=1) receive weight ``w1``; non-default observations
    (y=0) receive weight 1.  This up-weights the minority class when defaults
    are rare.

    Parameters
    ----------
    y_true : np.ndarray of shape (n,)
        Binary ground-truth labels.
    y_pred : np.ndarray of shape (n,)
        Predicted default probabilities.
    w1 : float or None, default None
        Weight for default observations.  If *None*, set to
        ``1 / mean(y_true)`` (inverse of the empirical default rate).
    eps : float, default 1e-15
        Clipping bound for predicted probabilities.

    Returns
    -------
    float
        Weighted mean log loss.

    Raises
    ------
    ValueError
        If all labels are identical (default rate is 0 or 1) when ``w1`` is
        *None*.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    p = np.clip(np.asarray(y_pred, dtype=np.float64), eps, 1.0 - eps)

    default_rate = float(np.mean(y_true))
    if w1 is None:
        if default_rate <= 0.0 or default_rate >= 1.0:
            raise ValueError(
                "Cannot compute default w1: all labels are identical "
                f"(default_rate={default_rate})."
            )
        w1 = 1.0 / default_rate

    weights = np.where(y_true == 1, w1, 1.0)
    per_sample = -(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))
    return float(np.sum(weights * per_sample) / np.sum(weights))


def compute_focal_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    gamma: float = 2.0,
    alpha: float = 0.25,
    eps: float = 1e-15,
) -> float:
    """Compute focal loss (Lin et al., 2017).

    FL = -mean(
            alpha * y * (1-p)^gamma * log(p + eps)
          + (1-alpha) * (1-y) * p^gamma * log(1-p + eps)
         )

    Predicted probabilities are clipped to [eps, 1 - eps] before evaluation.

    Parameters
    ----------
    y_true : np.ndarray of shape (n,)
        Binary ground-truth labels.
    y_pred : np.ndarray of shape (n,)
        Predicted default probabilities.
    gamma : float, default 2.0
        Focusing parameter; larger values down-weight easy examples more.
    alpha : float, default 0.25
        Class-balancing weight for the positive class.
    eps : float, default 1e-15
        Small constant added inside the logarithm.

    Returns
    -------
    float
        Mean focal loss (non-negative).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    p = np.clip(np.asarray(y_pred, dtype=np.float64), eps, 1.0 - eps)

    pos_term = alpha * y_true * (1.0 - p) ** gamma * np.log(p + eps)
    neg_term = (1.0 - alpha) * (1.0 - y_true) * p ** gamma * np.log(1.0 - p + eps)
    return float(-np.mean(pos_term + neg_term))


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    w1: Optional[float] = None,
) -> dict:
    """Compute the full suite of PD evaluation metrics.

    Parameters
    ----------
    y_true : np.ndarray of shape (n,)
        Binary ground-truth labels.
    y_pred : np.ndarray of shape (n,)
        Predicted default probabilities.
    w1 : float or None, default None
        Weight for defaults in the weighted log loss.  Passed directly to
        :func:`compute_weighted_log_loss`.

    Returns
    -------
    dict
        Keys: ``'auc'``, ``'brier'``, ``'log_loss'``, ``'weighted_log_loss'``,
        ``'focal_loss'``.
    """
    return {
        "auc": compute_auc(y_true, y_pred),
        "brier": compute_brier_score(y_true, y_pred),
        "log_loss": compute_log_loss(y_true, y_pred),
        "weighted_log_loss": compute_weighted_log_loss(y_true, y_pred, w1=w1),
        "focal_loss": compute_focal_loss(y_true, y_pred),
    }


# ---------------------------------------------------------------------------
# DeLong paired AUC test
# ---------------------------------------------------------------------------


def _structural_components(
    y_true: np.ndarray, pred: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute DeLong structural components V10 and V01.

    Parameters
    ----------
    y_true : np.ndarray of shape (n,)
        Binary labels.
    pred : np.ndarray of shape (n,)
        Predicted probabilities.

    Returns
    -------
    V10 : np.ndarray of shape (n1,)
        For each positive observation, the fraction of negatives it outranks
        (ties split as 0.5).
    V01 : np.ndarray of shape (n0,)
        For each negative observation, the fraction of positives that outrank
        it (ties split as 0.5).
    """
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(pred, dtype=np.float64)

    pred_pos = p[y == 1]  # (n1,)
    pred_neg = p[y == 0]  # (n0,)

    # V10: (n1, n0) comparison matrix -> mean over negatives
    cmp_10 = (pred_pos[:, np.newaxis] > pred_neg[np.newaxis, :]).astype(np.float64)
    tie_10 = (pred_pos[:, np.newaxis] == pred_neg[np.newaxis, :]).astype(np.float64)
    V10 = (cmp_10 + 0.5 * tie_10).mean(axis=1)

    # V01: (n0, n1) comparison matrix -> mean over positives
    cmp_01 = (pred_pos[np.newaxis, :] > pred_neg[:, np.newaxis]).astype(np.float64)
    tie_01 = (pred_pos[np.newaxis, :] == pred_neg[:, np.newaxis]).astype(np.float64)
    V01 = (cmp_01 + 0.5 * tie_01).mean(axis=1)

    return V10, V01


def delong_auc_variance(
    y_true: np.ndarray,
    pred1: np.ndarray,
    pred2: np.ndarray,
) -> Tuple[float, float, float, float, float]:
    """Paired DeLong test for comparing two correlated AUC estimates.

    Implements the structural-components method of DeLong, DeLong & Clarke-
    Pearson (1988) to estimate the variance of the difference between two
    correlated AUC values and compute a two-sided Z-test.

    Parameters
    ----------
    y_true : np.ndarray of shape (n,)
        Binary ground-truth labels.
    pred1 : np.ndarray of shape (n,)
        Predicted probabilities from model 1.
    pred2 : np.ndarray of shape (n,)
        Predicted probabilities from model 2.

    Returns
    -------
    auc1 : float
        ROC AUC of model 1.
    auc2 : float
        ROC AUC of model 2.
    delta_auc : float
        ``auc1 - auc2``.
    z_stat : float
        Z-statistic under the null hypothesis delta_AUC = 0.
    p_value_two_sided : float
        Two-sided p-value from the standard normal distribution.

    Notes
    -----
    The variance of delta AUC is:

        Var(AUC1 - AUC2) = Var(AUC1) + Var(AUC2) - 2 * Cov(AUC1, AUC2)

    where each term is estimated via the sample covariance of the structural
    components:

        Var(AUC_k)     = S10_k / n1 + S01_k / n0
        Cov(AUC1,AUC2) = S10_12 / n1 + S01_12 / n0

    and S10_k, S01_k are sample variances and S10_12, S01_12 are sample
    covariances of the V10 and V01 structural components.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    pred1 = np.asarray(pred1, dtype=np.float64)
    pred2 = np.asarray(pred2, dtype=np.float64)

    n1 = int(np.sum(y_true == 1))
    n0 = int(np.sum(y_true == 0))
    if n1 < 2 or n0 < 2:
        raise ValueError(
            f"DeLong test requires at least 2 positives and 2 negatives "
            f"(got n1={n1}, n0={n0})."
        )

    V10_1, V01_1 = _structural_components(y_true, pred1)
    V10_2, V01_2 = _structural_components(y_true, pred2)

    auc1 = float(np.mean(V10_1))
    auc2 = float(np.mean(V10_2))
    delta_auc = auc1 - auc2

    # Within-model variances
    s10_1 = float(np.var(V10_1, ddof=1))
    s01_1 = float(np.var(V01_1, ddof=1))
    s10_2 = float(np.var(V10_2, ddof=1))
    s01_2 = float(np.var(V01_2, ddof=1))

    var_auc1 = s10_1 / n1 + s01_1 / n0
    var_auc2 = s10_2 / n1 + s01_2 / n0

    # Cross-model covariances
    cov_10 = float(np.cov(V10_1, V10_2, ddof=1)[0, 1])
    cov_01 = float(np.cov(V01_1, V01_2, ddof=1)[0, 1])
    cov_aucs = cov_10 / n1 + cov_01 / n0

    var_delta = var_auc1 + var_auc2 - 2.0 * cov_aucs

    if var_delta <= 0.0:
        return auc1, auc2, delta_auc, 0.0, 1.0

    z_stat = delta_auc / float(np.sqrt(var_delta))
    p_value = float(2.0 * scipy.stats.norm.sf(abs(z_stat)))

    return auc1, auc2, delta_auc, z_stat, p_value


# ---------------------------------------------------------------------------
# Multiple-testing correction
# ---------------------------------------------------------------------------


def holm_bonferroni_adjust(p_values: List[float]) -> np.ndarray:
    """Apply Holm-Bonferroni step-down correction to a set of p-values.

    The adjustment controls the family-wise error rate (FWER) at the same
    alpha level as the Bonferroni correction, but with strictly greater power.

    Parameters
    ----------
    p_values : list[float]
        Raw (unadjusted) p-values.

    Returns
    -------
    np.ndarray of shape (k,)
        Adjusted p-values, capped at 1, in the same order as the input.
    """
    p = np.asarray(p_values, dtype=np.float64)
    k = len(p)
    if k == 0:
        return np.array([], dtype=np.float64)

    order = np.argsort(p)
    sorted_p = p[order]

    # Step-down multipliers: (k, k-1, ..., 1)
    multipliers = k - np.arange(k, dtype=np.float64)
    adjusted_sorted = np.minimum(1.0, multipliers * sorted_p)

    # Enforce monotonicity (cumulative maximum)
    for i in range(1, k):
        adjusted_sorted[i] = max(adjusted_sorted[i], adjusted_sorted[i - 1])

    result = np.empty(k, dtype=np.float64)
    result[order] = adjusted_sorted
    return result


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------


def run_delong_comparisons(
    y_true: np.ndarray,
    pred_functional: np.ndarray,
    pred_logistic: np.ndarray,
    pred_xgboost: np.ndarray,
    label: str = "test",
) -> pd.DataFrame:
    """Run paired DeLong tests comparing the functional model against baselines.

    Performs two comparisons (Functional vs Logistic, Functional vs XGBoost)
    and applies Holm-Bonferroni correction across both raw p-values.

    Parameters
    ----------
    y_true : np.ndarray of shape (n,)
        Binary ground-truth labels.
    pred_functional : np.ndarray of shape (n,)
        Predicted probabilities from the functional logistic model.
    pred_logistic : np.ndarray of shape (n,)
        Predicted probabilities from the scalar logistic model.
    pred_xgboost : np.ndarray of shape (n,)
        Predicted probabilities from the XGBoost model.
    label : str, default ``'test'``
        Dataset label prepended to the comparison identifiers.

    Returns
    -------
    pd.DataFrame
        Columns: ``comparison``, ``auc1``, ``auc2``, ``delta_auc``,
        ``z_stat``, ``raw_pvalue``, ``adjusted_pvalue``.
    """
    auc1_l, auc2_l, d_l, z_l, p_l = delong_auc_variance(
        y_true, pred_functional, pred_logistic
    )
    auc1_x, auc2_x, d_x, z_x, p_x = delong_auc_variance(
        y_true, pred_functional, pred_xgboost
    )

    raw_pvalues = [p_l, p_x]
    adj_pvalues = holm_bonferroni_adjust(raw_pvalues)

    records = [
        {
            "comparison": f"{label}_functional_vs_logistic",
            "auc1": auc1_l,
            "auc2": auc2_l,
            "delta_auc": d_l,
            "z_stat": z_l,
            "raw_pvalue": p_l,
            "adjusted_pvalue": float(adj_pvalues[0]),
        },
        {
            "comparison": f"{label}_functional_vs_xgboost",
            "auc1": auc1_x,
            "auc2": auc2_x,
            "delta_auc": d_x,
            "z_stat": z_x,
            "raw_pvalue": p_x,
            "adjusted_pvalue": float(adj_pvalues[1]),
        },
    ]
    return pd.DataFrame(records)


def evaluate_model_suite(
    y_true: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    w1: Optional[float] = None,
) -> pd.DataFrame:
    """Compute the full metric suite for multiple models.

    Parameters
    ----------
    y_true : np.ndarray of shape (n,)
        Binary ground-truth labels.
    predictions_dict : dict[str, np.ndarray]
        Maps model name to its predicted default probabilities.
    w1 : float or None, default None
        Weight for default observations in the weighted log loss.
        If *None*, computed as ``1 / mean(y_true)`` for each model.

    Returns
    -------
    pd.DataFrame
        Rows are model names; columns are metric names:
        ``auc``, ``brier``, ``log_loss``, ``weighted_log_loss``,
        ``focal_loss``.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    rows: List[dict] = []
    for model_name, y_pred in predictions_dict.items():
        metrics = compute_all_metrics(y_true, np.asarray(y_pred, dtype=np.float64), w1=w1)
        metrics["model"] = model_name
        rows.append(metrics)

    df = pd.DataFrame(rows).set_index("model")
    return df[["auc", "brier", "log_loss", "weighted_log_loss", "focal_loss"]]

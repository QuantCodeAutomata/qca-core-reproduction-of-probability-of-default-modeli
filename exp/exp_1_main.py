"""Experiment 1 – Main PD modelling comparison.

Compares three model families on synthetic retail banking data:
  - Scalar logistic regression (greedy forward selection)
  - XGBoost with Optuna hyperparameter tuning
  - Functional logistic regression (three pre-specified terms)

Results are saved to results/exp1_metrics.csv and results/exp1_*.png.

Run from the project root:
    python exp/exp_1_main.py
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

# Ensure project root is on the Python path so src/* imports work.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.interpolate import BSpline
except ImportError as e:
    print(f"[ERROR] scipy not available: {e}")
    sys.exit(1)

try:
    from src.config import CONFIG
    from src.calendar_utils import generate_banking_days, get_month_end_banking_days
    from src.preprocessing import (
        reconstruct_eod_states,
        compute_account_level_daily,
        aggregate_to_borrower_level,
        aggregate_to_household_level,
        assign_sampling_dates,
        create_default_labels,
        winsorize_series,
        apply_winsorization,
    )
    from src.feature_engineering import (
        build_scalar_features,
        build_functional_predictors,
        compute_roughness_penalty_matrix,
        compute_inner_product_matrix,
    )
    from src.models.logistic_model import LogisticPDModel
    from src.models.xgboost_model import XGBoostPDModel
    from src.models.functional_logistic import FunctionalLogisticModel
    from src.evaluation import (
        compute_all_metrics,
        run_delong_comparisons,
        evaluate_model_suite,
    )
    from data.synthetic_generator import generate_synthetic_banking_data
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    print("  Make sure you are running from the project root and all dependencies are installed.")
    sys.exit(1)

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Split proportions derived from CONFIG paper targets ────────────────────────
_TOTAL_DEV = CONFIG["train_size"] + CONFIG["val_size"] + CONFIG["test_size"]
TRAIN_FRAC = CONFIG["train_size"] / _TOTAL_DEV   # ≈ 0.564
VAL_FRAC   = CONFIG["val_size"]   / _TOTAL_DEV   # ≈ 0.242
# test gets the remainder

# ── Degree of B-spline coefficient function basis ─────────────────────────────
_DEG = 3
_L   = CONFIG["L_coeff_basis"]  # 5 coefficient basis functions


# ---------------------------------------------------------------------------
# Helper: build J and S matrices for a functional term
# ---------------------------------------------------------------------------

def _make_coeff_knots(domain_end: float, L: int, deg: int = _DEG) -> np.ndarray:
    """Build clamped cubic B-spline knot vector for the coefficient basis."""
    n_int = L - deg - 1
    interior = np.linspace(0.0, domain_end, n_int + 2)[1:-1] if n_int > 0 else np.array([])
    return np.concatenate([np.zeros(deg + 1), interior, np.full(deg + 1, domain_end)])


def _build_term_matrices(
    basis_info: dict,
    L: int = _L,
) -> tuple:
    """Compute the inner-product matrix J and roughness-penalty matrices.

    Parameters
    ----------
    basis_info : dict
        Returned by :func:`build_functional_predictors`.
    L : int
        Number of coefficient basis functions (β(s) expansion).

    Returns
    -------
    J : np.ndarray (L, K)
    S_coeff : np.ndarray (L, L)   – penalty for β(s)
    S_pred  : np.ndarray (K, K)   – penalty for x̃(s)  (reference only)
    coeff_knots : np.ndarray
    """
    domain_end = float(basis_info["domain"][1])
    coeff_knots = _make_coeff_knots(domain_end, L)
    S_coeff = compute_roughness_penalty_matrix(
        coeff_knots, degree=_DEG, domain_start=0.0, domain_end=domain_end
    )

    if basis_info["type"] == "cubic_spline":
        phi_params = {"knots": coeff_knots, "degree": _DEG}
        psi_params = {"knots": basis_info["knots"], "degree": basis_info["degree"]}
        J = compute_inner_product_matrix(
            phi_params, psi_params, domain_start=0.0, domain_end=domain_end
        )
        S_pred = compute_roughness_penalty_matrix(
            basis_info["knots"],
            degree=basis_info["degree"],
            domain_end=domain_end,
        )

    elif basis_info["type"] == "mean_spline":
        # Predictor basis: piecewise-constant indicators over [k−1, k]
        K = basis_info["n_basis"]
        xi, wi = np.polynomial.legendre.leggauss(50)
        J = np.zeros((L, K), dtype=np.float64)
        for k in range(K):
            lo, hi = float(k), float(k + 1)
            scale = 0.5 * (hi - lo)
            shift = 0.5 * (hi + lo)
            s_q = scale * xi + shift
            w_q = scale * wi
            for ll in range(L):
                c = np.zeros(L, dtype=np.float64)
                c[ll] = 1.0
                spl = BSpline(coeff_knots, c, _DEG, extrapolate=False)
                vals = spl(s_q)
                J[ll, k] = np.dot(w_q, np.where(np.isnan(vals), 0.0, vals))
        S_pred = np.eye(K, dtype=np.float64)

    else:
        raise ValueError(f"Unknown basis type: {basis_info['type']}")

    return J, S_coeff, S_pred, coeff_knots


def _impute_nan(arr: np.ndarray, col_means: np.ndarray | None = None) -> np.ndarray:
    """Replace NaN entries with column means (computed from arr if not supplied)."""
    arr = arr.copy()
    if col_means is None:
        col_means = np.nanmean(arr, axis=0)
    for j in range(arr.shape[1]):
        mask = np.isnan(arr[:, j])
        if mask.any():
            fill = col_means[j] if np.isfinite(col_means[j]) else 0.0
            arr[mask, j] = fill
    return arr


def _build_functional_term(
    bw_daily: pd.DataFrame,
    hh_daily: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    oot_df: pd.DataFrame,
    banking_days: pd.DatetimeIndex,
    month_end_days: pd.Series,
    term_name: str,
    variable: str,
    level: str,
    M: int,
    basis_type: str,
    lambda_x: float,
) -> dict:
    """Build one functional term dict ready for FunctionalLogisticModel.

    Returns a dict with keys:
        name, C_train, C_val, C_test, C_oot, J, S_pred, S_coeff, domain_end,
        coeff_col_means (for NaN imputation of future splits).
    """
    def _fp(sdf):
        return build_functional_predictors(
            bw_daily, hh_daily, sdf[["borrower_id", "sampling_date"]],
            banking_days, month_end_days,
            variable=variable, level=level, M=M,
            lambda_x=lambda_x, basis_type=basis_type,
        )

    C_train, basis_info = _fp(train_df)
    C_val,   _          = _fp(val_df)
    C_test,  _          = _fp(test_df)
    C_oot,   _          = _fp(oot_df)

    # Impute NaN rows using training column means
    col_means = np.nanmean(C_train, axis=0)
    C_train = _impute_nan(C_train, col_means)
    C_val   = _impute_nan(C_val,   col_means)
    C_test  = _impute_nan(C_test,  col_means)
    C_oot   = _impute_nan(C_oot,   col_means)

    J, S_coeff, S_pred, coeff_knots = _build_term_matrices(basis_info, L=_L)
    domain_end = float(basis_info["domain"][1])

    return {
        "name":           term_name,
        "C_train":        C_train,
        "C_val":          C_val,
        "C_test":         C_test,
        "C_oot":          C_oot,
        "J":              J,
        "S_pred":         S_pred,
        "S_coeff":        S_coeff,
        "domain_end":     domain_end,
        "coeff_col_means": col_means,
        "basis_info":     basis_info,
        "coeff_knots":    coeff_knots,
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_exp1() -> None:
    print("=" * 70)
    print("Experiment 1 – Main PD Modelling Comparison")
    print("=" * 70)

    # ── Step 1: Generate synthetic banking data ────────────────────────────
    print("\n[Step 1] Generating synthetic banking data …")
    raw = generate_synthetic_banking_data(
        n_borrowers=CONFIG["n_synthetic_borrowers"],
        seed=CONFIG["random_seed"],
    )
    accounts_df  = raw["accounts"]
    balances_df  = raw["balances"]
    defaults_df  = raw["defaults"]
    print(f"  Accounts: {len(accounts_df):,}  |  "
          f"Balance records: {len(balances_df):,}  |  "
          f"Defaults: {len(defaults_df):,}")

    # ── Step 2: Build banking-day calendar ────────────────────────────────
    print("\n[Step 2] Building banking-day calendar …")
    banking_days   = generate_banking_days("2016-01-01", "2025-01-01")
    month_end_days = get_month_end_banking_days("2016-01-01", "2025-01-01")
    print(f"  Banking days: {len(banking_days):,}  |  "
          f"Month-ends: {len(month_end_days):,}")

    # ── Step 3: Reconstruct end-of-day states ─────────────────────────────
    print("\n[Step 3] Reconstructing end-of-day states (LOCF) …")
    eod_df = reconstruct_eod_states(accounts_df, balances_df, banking_days)
    print(f"  EOD records: {len(eod_df):,}")
    if len(eod_df) == 0:
        print("[ERROR] EOD state DataFrame is empty – aborting.")
        return

    # ── Step 4: Compute account-level derived variables ───────────────────
    print("\n[Step 4] Computing account-level daily variables …")
    acct_daily = compute_account_level_daily(eod_df)
    del eod_df  # free memory: EOD states no longer needed
    print(f"  Account-daily records: {len(acct_daily):,}")

    # ── Step 5: Aggregate to borrower and household level ─────────────────
    print("\n[Step 5] Aggregating to borrower and household level …")
    bw_daily = aggregate_to_borrower_level(acct_daily)
    hh_daily = aggregate_to_household_level(acct_daily, accounts_df)
    del acct_daily  # free memory: account-daily no longer needed
    print(f"  Borrower-daily: {len(bw_daily):,}  |  "
          f"Household-daily: {len(hh_daily):,}")

    # ── Step 6: Assign sampling dates (dev + OOT) ─────────────────────────
    print("\n[Step 6] Assigning sampling dates …")
    borrower_ids = sorted(accounts_df["borrower_id"].unique().tolist())
    sampling_df = assign_sampling_dates(
        borrower_ids=borrower_ids,
        banking_days=banking_days,
        month_end_days=month_end_days,
        min_lookback_months=CONFIG["min_lookback_months"],
        min_forward_months=CONFIG["default_horizon_months"],
        dev_start=CONFIG["dev_start"],
        dev_end=CONFIG["dev_end"],
        oot_start=CONFIG["oot_start"],
        oot_end=CONFIG["oot_end"],
        seed=CONFIG["random_seed"],
    )
    n_dev = (sampling_df["sample_type"] == "dev").sum()
    n_oot = (sampling_df["sample_type"] == "oot").sum()
    print(f"  Dev: {n_dev:,}  |  OOT: {n_oot:,}")

    # ── Step 7: Create default labels ─────────────────────────────────────
    print("\n[Step 7] Creating default labels (12-month horizon) …")
    labeled_df = create_default_labels(
        sampling_df, defaults_df, horizon_months=CONFIG["default_horizon_months"]
    )
    dr_dev = labeled_df.loc[labeled_df["sample_type"] == "dev", "default_label"].mean()
    dr_oot = labeled_df.loc[labeled_df["sample_type"] == "oot", "default_label"].mean()
    print(f"  Default rate – dev: {dr_dev:.3f}  |  OOT: {dr_oot:.3f}")

    # ── Step 8: Split dev into train / val / test ─────────────────────────
    print("\n[Step 8] Splitting dev sample into train / val / test …")
    dev_df = labeled_df[labeled_df["sample_type"] == "dev"].reset_index(drop=True)
    oot_df = labeled_df[labeled_df["sample_type"] == "oot"].reset_index(drop=True)

    n_total = len(dev_df)
    n_train = int(np.floor(n_total * TRAIN_FRAC))
    n_val   = int(np.floor(n_total * VAL_FRAC))
    n_test  = n_total - n_train - n_val

    rng = np.random.default_rng(CONFIG["random_seed"])
    idx = rng.permutation(n_total)
    train_df = dev_df.iloc[idx[:n_train]].reset_index(drop=True)
    val_df   = dev_df.iloc[idx[n_train : n_train + n_val]].reset_index(drop=True)
    test_df  = dev_df.iloc[idx[n_train + n_val :]].reset_index(drop=True)
    print(f"  Train: {len(train_df):,}  |  Val: {len(val_df):,}  |  "
          f"Test: {len(test_df):,}  |  OOT: {len(oot_df):,}")

    # ── Step 9: Compute winsorization thresholds on training data ─────────
    print("\n[Step 9] Computing winsorization thresholds from training data …")
    # We build scalar features first to get the feature-level bounds.
    # Only train set is used to compute thresholds.
    X_train_raw = build_scalar_features(
        bw_daily, hh_daily, train_df[["borrower_id", "sampling_date"]],
        banking_days, month_end_days,
    )
    feature_cols = list(X_train_raw.columns)

    winsor_thresholds: dict = {}
    for col in feature_cols:
        lo, hi = winsorize_series(
            X_train_raw[col],
            lower_pct=CONFIG["winsorization_lower"],
            upper_pct=CONFIG["winsorization_upper"],
        )
        winsor_thresholds[col] = (lo, hi)
    print(f"  Computed thresholds for {len(winsor_thresholds)} features.")

    # ── Step 10: Build and winsorize scalar features for all splits ────────
    print("\n[Step 10] Building scalar features for all splits …")
    X_val_raw  = build_scalar_features(
        bw_daily, hh_daily, val_df[["borrower_id", "sampling_date"]],
        banking_days, month_end_days,
    )
    X_test_raw = build_scalar_features(
        bw_daily, hh_daily, test_df[["borrower_id", "sampling_date"]],
        banking_days, month_end_days,
    )
    X_oot_raw  = build_scalar_features(
        bw_daily, hh_daily, oot_df[["borrower_id", "sampling_date"]],
        banking_days, month_end_days,
    )

    # Apply winsorization (thresholds from training only)
    X_train_w = apply_winsorization(X_train_raw.reset_index(), feature_cols, winsor_thresholds)
    X_val_w   = apply_winsorization(X_val_raw.reset_index(),   feature_cols, winsor_thresholds)
    X_test_w  = apply_winsorization(X_test_raw.reset_index(),  feature_cols, winsor_thresholds)
    X_oot_w   = apply_winsorization(X_oot_raw.reset_index(),   feature_cols, winsor_thresholds)

    # Align with label DataFrames (merge on borrower_id)
    def _align(feat_df: pd.DataFrame, label_df: pd.DataFrame) -> tuple:
        merged = label_df[["borrower_id", "default_label"]].merge(
            feat_df, on="borrower_id", how="left"
        )
        X = merged[feature_cols].fillna(0.0)
        y = merged["default_label"].to_numpy(dtype=np.float64)
        return X, y

    X_train, y_train = _align(X_train_w, train_df)
    X_val,   y_val   = _align(X_val_w,   val_df)
    X_test,  y_test  = _align(X_test_w,  test_df)
    X_oot,   y_oot   = _align(X_oot_w,   oot_df)
    print(f"  Feature matrix shapes – train {X_train.shape}, "
          f"val {X_val.shape}, test {X_test.shape}, oot {X_oot.shape}")

    # ── Step 11: Fit LogisticPDModel ──────────────────────────────────────
    print("\n[Step 11] Fitting LogisticPDModel (greedy forward selection) …")
    logistic_model = LogisticPDModel()
    logistic_model.fit(
        X_train, y_train, X_val, y_val,
        auc_threshold=CONFIG["auc_improvement_threshold"],
        max_features=15,
        random_state=CONFIG["random_seed"],
    )
    sel_feats = logistic_model.get_selected_features()
    print(f"  Selected {len(sel_feats)} features: {sel_feats[:5]} …")

    # ── Step 12: Fit XGBoostPDModel ───────────────────────────────────────
    print("\n[Step 12] Fitting XGBoostPDModel (Optuna HPO) …")
    xgb_model = XGBoostPDModel()
    xgb_model.fit(
        X_train.to_numpy(dtype=np.float64), y_train,
        X_val.to_numpy(dtype=np.float64),   y_val,
        random_state=CONFIG["random_seed"],
        n_trials=CONFIG["xgboost_n_trials"],
    )
    print(f"  Best XGB params: {xgb_model.best_params_}")

    # ── Step 13: Build functional predictors ──────────────────────────────
    print("\n[Step 13] Building functional predictors for 3 selected terms …")
    # Paper-reported terms: (variable, level, M, basis_type, lambda_x)
    term_specs = [
        ("hh_gearing_M9",              "gearing",           "hh", 9,  "cubic_spline", 1e-4),
        ("bw_relative_overdraft_M12",  "relative_overdraft", "bw", 12, "mean_spline",  0.0),
        ("hh_available_funds_M3",      "available_funds",   "hh", 3,  "cubic_spline", 1.0),
    ]

    func_terms = []
    for t_name, var, lvl, M, btype, lx in term_specs:
        print(f"  Building term '{t_name}' ({btype}, M={M}, λ_x={lx}) …")
        td = _build_functional_term(
            bw_daily, hh_daily,
            train_df, val_df, test_df, oot_df,
            banking_days, month_end_days,
            term_name=t_name, variable=var, level=lvl,
            M=M, basis_type=btype, lambda_x=lx,
        )
        func_terms.append(td)
        K = td["C_train"].shape[1]
        print(f"    C_train shape: ({len(train_df)}, {K})  "
              f"J shape: {td['J'].shape}")

    # ── Step 14: Fit FunctionalLogisticModel ──────────────────────────────
    print("\n[Step 14] Fitting FunctionalLogisticModel (greedy forward selection) …")
    # Build terms_list expected by FunctionalLogisticModel.fit
    terms_list = [
        {
            "name":       td["name"],
            "C_train":    td["C_train"],
            "C_val":      td["C_val"],
            "J":          td["J"],
            "S_pred":     td["S_pred"],
            "S_coeff":    td["S_coeff"],
            "domain_end": td["domain_end"],
        }
        for td in func_terms
    ]

    func_model = FunctionalLogisticModel()
    try:
        func_model.fit(
            terms_list=terms_list,
            y_train=y_train,
            y_val=y_val,
            auc_threshold=CONFIG["auc_improvement_threshold"],
            L=_L,
        )
        print(f"  Selected terms: {func_model.selected_terms_}")
        print(f"  Estimated λ_β: {func_model.lambda_beta_:.4f}")
    except Exception as exc:
        print(f"  [WARN] FunctionalLogisticModel fitting failed: {exc}")
        print("  Continuing without functional model results.")
        func_model = None

    # ── Step 15: Score all models on test and OOT ────────────────────────
    print("\n[Step 15] Scoring models on test and OOT sets …")

    pred_test_logistic = logistic_model.predict_proba(X_test)
    pred_test_xgb      = xgb_model.predict_proba(X_test.to_numpy(dtype=np.float64))
    pred_oot_logistic  = logistic_model.predict_proba(X_oot)
    pred_oot_xgb       = xgb_model.predict_proba(X_oot.to_numpy(dtype=np.float64))

    pred_test_func = None
    pred_oot_func  = None
    if func_model is not None and func_model.selected_terms_:
        # Build prediction term lists for test and OOT
        def _pred_terms(split_key: str):
            return [
                {"name": td["name"], "C": td[split_key], "J": td["J"]}
                for td in func_terms
            ]
        try:
            pred_test_func = func_model.predict_proba(_pred_terms("C_test"))
            pred_oot_func  = func_model.predict_proba(_pred_terms("C_oot"))
        except Exception as exc:
            print(f"  [WARN] Functional model prediction failed: {exc}")

    # ── Step 16: Compute all metrics ──────────────────────────────────────
    print("\n[Step 16] Computing evaluation metrics …")
    metrics_records = []
    for split, y_true, preds in [
        ("test", y_test, [
            ("logistic",   pred_test_logistic),
            ("xgboost",    pred_test_xgb),
            ("functional", pred_test_func),
        ]),
        ("oot", y_oot, [
            ("logistic",   pred_oot_logistic),
            ("xgboost",    pred_oot_xgb),
            ("functional", pred_oot_func),
        ]),
    ]:
        for model_name, pred in preds:
            if pred is None:
                continue
            m = compute_all_metrics(y_true, pred)
            m["model"] = model_name
            m["split"] = split
            metrics_records.append(m)
            print(f"  [{split}] {model_name:12s}  AUC={m['auc']:.4f}  "
                  f"Brier={m['brier']:.4f}  LogLoss={m['log_loss']:.4f}")

    metrics_df = pd.DataFrame(metrics_records)
    metrics_path = RESULTS_DIR / "exp1_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n  Metrics saved to {metrics_path}")

    # Compare to paper targets
    targets = CONFIG["paper_target_aucs"]
    print("\n  Paper target AUCs (for reference):")
    for split in ("test", "oot"):
        for m in ("functional", "logistic", "xgboost"):
            tgt = targets[split][m]
            row = metrics_df[
                (metrics_df["split"] == split) & (metrics_df["model"] == m)
            ]
            obs = row["auc"].iloc[0] if len(row) else float("nan")
            print(f"    [{split}] {m:12s}  paper={tgt:.4f}  synthetic={obs:.4f}")

    # ── Step 17: Run DeLong comparisons on test ────────────────────────────
    if pred_test_func is not None:
        print("\n[Step 17] Running DeLong AUC comparisons on test set …")
        try:
            delong_df = run_delong_comparisons(
                y_test,
                pred_functional=pred_test_func,
                pred_logistic=pred_test_logistic,
                pred_xgboost=pred_test_xgb,
                label="test",
            )
            print(delong_df.to_string(index=False))
            delong_path = RESULTS_DIR / "exp1_delong.csv"
            delong_df.to_csv(delong_path, index=False)
            print(f"  DeLong results saved to {delong_path}")
        except Exception as exc:
            print(f"  [WARN] DeLong test failed: {exc}")
    else:
        print("\n[Step 17] Skipped – functional model not available.")

    # ── Step 18: Plot coefficient functions ───────────────────────────────
    if func_model is not None and func_model.selected_terms_:
        print("\n[Step 18] Plotting coefficient functions …")
        try:
            coeff_funcs = func_model.get_coefficient_functions(
                grid_points_per_term=CONFIG["bootstrap_dense_grid_points"]
            )
            n_terms = len(coeff_funcs)
            fig, axes = plt.subplots(1, max(n_terms, 1), figsize=(5 * max(n_terms, 1), 4))
            if n_terms == 1:
                axes = [axes]
            for ax, (tname, cf) in zip(axes, coeff_funcs.items()):
                ax.plot(cf["t_grid"], cf["beta"], color="steelblue", linewidth=2)
                ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
                ax.set_xlabel("Relative time τ")
                ax.set_ylabel("β(τ)")
                ax.set_title(tname.replace("_", " "))
                ax.grid(True, alpha=0.3)
            fig.suptitle("Functional Logistic Regression – Coefficient Functions", fontsize=13)
            plt.tight_layout()
            plot_path = RESULTS_DIR / "exp1_coeff_functions.png"
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            print(f"  Coefficient function plot saved to {plot_path}")
        except Exception as exc:
            print(f"  [WARN] Coefficient function plot failed: {exc}")

    # ── Step 19: Save AUC comparison bar chart ────────────────────────────
    print("\n[Step 19] Saving AUC comparison plot …")
    try:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
        for ax, split in zip(axes, ("test", "oot")):
            sub = metrics_df[metrics_df["split"] == split]
            ax.bar(sub["model"], sub["auc"], color=["steelblue", "darkorange", "seagreen"][:len(sub)])
            ax.set_ylim(max(0.5, sub["auc"].min() - 0.02), min(1.0, sub["auc"].max() + 0.02))
            ax.set_title(f"AUC – {split.upper()}")
            ax.set_ylabel("ROC AUC")
            ax.set_xlabel("Model")
            for bar, val in zip(ax.patches, sub["auc"]):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                        f"{val:.4f}", ha="center", va="bottom", fontsize=9)
        fig.suptitle("Exp 1 – Model AUC Comparison (Synthetic Data)", fontsize=13)
        plt.tight_layout()
        bar_path = RESULTS_DIR / "exp1_auc_comparison.png"
        fig.savefig(bar_path, dpi=150)
        plt.close(fig)
        print(f"  AUC comparison plot saved to {bar_path}")
    except Exception as exc:
        print(f"  [WARN] AUC plot failed: {exc}")

    print("\n" + "=" * 70)
    print("Experiment 1 complete.")
    print(f"  Results directory: {RESULTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    run_exp1()

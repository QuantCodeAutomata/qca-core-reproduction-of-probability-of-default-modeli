"""Experiment 2 – Month-end-only functional logistic regression.

Uses only the last banking day of each calendar month (rather than all daily
observations) to build functional predictors.  This tests whether the full
daily path information is necessary or whether coarser monthly snapshots
achieve comparable discrimination.

Paper reference values (Table 4):
    Month-end functional AUC = 0.8920
    ΔAUC vs logistic = 0.0019, Z = 0.832, p = 0.4052

Run from the project root:
    python exp/exp_2_month_end.py
"""

from __future__ import annotations

import sys
from pathlib import Path

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
    from src.calendar_utils import (
        generate_banking_days,
        get_month_end_banking_days,
        get_banking_days_in_month,
    )
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
        compute_roughness_penalty_matrix,
        compute_inner_product_matrix,
    )
    from src.models.logistic_model import LogisticPDModel
    from src.models.functional_logistic import FunctionalLogisticModel
    from src.evaluation import compute_all_metrics, delong_auc_variance
    from data.synthetic_generator import generate_synthetic_banking_data
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    sys.exit(1)

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Split proportions (same as exp_1) ─────────────────────────────────────────
_TOTAL_DEV = CONFIG["train_size"] + CONFIG["val_size"] + CONFIG["test_size"]
TRAIN_FRAC = CONFIG["train_size"] / _TOTAL_DEV
VAL_FRAC   = CONFIG["val_size"]   / _TOTAL_DEV

_DEG = 3   # cubic B-spline degree
_L   = CONFIG["L_coeff_basis"]  # 5 coefficient basis functions

# Variables and levels for which month-end functional predictors are built
_FUNC_SPEC = [
    # (term_name, variable, level)
    ("hh_gearing",             "gearing",             "hh"),
    ("bw_relative_overdraft",  "relative_overdraft",  "bw"),
    ("hh_available_funds",     "available_funds",     "hh"),
    ("bw_utilization",         "utilization",         "bw"),
    ("hh_overdraft",           "overdraft",           "hh"),
]


# ---------------------------------------------------------------------------
# Month-end functional predictor builder
# ---------------------------------------------------------------------------

def _month_col(variable: str, level: str) -> str:
    """Map (variable, level) to the column name in the daily DataFrame."""
    if level == "bw":
        return variable
    col_map = {
        "gearing":            "hh_gearing",
        "relative_overdraft": "hh_relative_overdraft",
        "available_funds":    "hh_available_funds",
        "overdraft":          "hh_overdraft",
        "utilization":        "hh_utilization",
    }
    return col_map[variable]


def _make_me_knots(M: int, deg: int = _DEG) -> np.ndarray:
    """Clamped B-spline knot vector on [1, M] with n_basis = max(deg+1, M)."""
    domain_lo, domain_hi = 1.0, float(M)
    n_basis = max(deg + 1, M)
    n_int = n_basis - deg - 1
    interior = np.linspace(domain_lo, domain_hi, n_int + 2)[1:-1] if n_int > 0 else np.array([])
    return np.concatenate([
        np.full(deg + 1, domain_lo), interior, np.full(deg + 1, domain_hi)
    ]), n_basis


def _build_month_end_coeff_matrix(
    bw_daily: pd.DataFrame,
    hh_daily: pd.DataFrame,
    sampling_df: pd.DataFrame,
    banking_days: pd.DatetimeIndex,
    month_end_days: pd.Series,
    variable: str,
    level: str,
    M: int,
) -> tuple:
    """Build a K-dimensional B-spline coefficient vector per borrower from
    M month-end observations only.

    Parameters
    ----------
    M : int
        Number of calendar months to look back.

    Returns
    -------
    C : np.ndarray (n_obs, n_basis)
        Coefficient matrix (NaN rows for borrowers with missing data).
    knots : np.ndarray
        Knot vector used.
    n_basis : int
    """
    col = _month_col(variable, level)
    if level == "bw":
        daily_df = bw_daily
    else:
        daily_df = hh_daily

    daily_idx = (
        daily_df.copy()
        .assign(date=lambda d: pd.to_datetime(d["date"]).dt.normalize())
        .set_index(["borrower_id", "date"])
    )

    # Build month-end lookup: (year, month) → Timestamp
    me_lookup: dict = {}
    for period, ts in month_end_days.items():
        p = pd.Period(period, freq="M")
        me_lookup[(int(p.year), int(p.month))] = pd.Timestamp(ts).normalize()

    knots, n_basis = _make_me_knots(M)
    # Fixed evaluation points: t = 1, 2, ..., M
    t_grid = np.arange(1, M + 1, dtype=np.float64)
    # Basis matrix at the M integer points (same for every borrower)
    B_fixed = np.zeros((M, n_basis), dtype=np.float64)
    for k in range(n_basis):
        c = np.zeros(n_basis, dtype=np.float64)
        c[k] = 1.0
        spl = BSpline(knots, c, _DEG, extrapolate=True)
        B_fixed[:, k] = spl(t_grid)

    coeff_rows = []
    for srow in sampling_df.itertuples(index=False):
        b_id = int(srow.borrower_id)
        T = pd.Timestamp(srow.sampling_date).normalize()

        vals = []
        for m in range(1, M + 1):
            # m=1: M months before T (oldest); m=M: 1 month before T (newest)
            months_before = M - m + 1
            raw_month = T.month - months_before
            t_year  = T.year + (raw_month - 1) // 12
            t_month = (raw_month - 1) % 12 + 1
            me_date = me_lookup.get((t_year, t_month))
            if me_date is None:
                vals.append(np.nan)
                continue
            try:
                v = float(daily_idx.at[(b_id, me_date), col])
                vals.append(v)
            except KeyError:
                vals.append(np.nan)

        vals_arr = np.array(vals, dtype=np.float64)
        if np.all(np.isnan(vals_arr)) or M < 2:
            coeff_rows.append(np.full(n_basis, np.nan, dtype=np.float64))
            continue

        # Fill NaN with column mean (within this borrower's window)
        nan_mask = np.isnan(vals_arr)
        if nan_mask.any():
            mean_val = float(np.nanmean(vals_arr))
            vals_arr[nan_mask] = mean_val

        # Solve for spline coefficients: B_fixed @ c = vals_arr
        c_vec, _, _, _ = np.linalg.lstsq(B_fixed, vals_arr, rcond=None)
        coeff_rows.append(c_vec.astype(np.float64))

    C = np.vstack(coeff_rows) if coeff_rows else np.empty((0, n_basis), dtype=np.float64)
    return C, knots, n_basis


def _make_coeff_basis_knots(domain_lo: float, domain_hi: float, L: int, deg: int = _DEG) -> np.ndarray:
    """Coefficient basis knot vector on [domain_lo, domain_hi]."""
    n_int = L - deg - 1
    interior = np.linspace(domain_lo, domain_hi, n_int + 2)[1:-1] if n_int > 0 else np.array([])
    return np.concatenate([np.full(deg + 1, domain_lo), interior, np.full(deg + 1, domain_hi)])


def _build_me_term_matrices(pred_knots: np.ndarray, n_basis_K: int, M: int, L: int = _L) -> tuple:
    """Build J and S matrices for a month-end functional term.

    The predictor basis is a cubic B-spline on [1, M].
    The coefficient function basis is also a cubic B-spline on [1, M] with L functions.

    Returns
    -------
    J       : (L, K) inner-product matrix
    S_coeff : (L, L) roughness penalty for β(s)
    S_pred  : (K, K) roughness penalty for x̃(s)
    coeff_knots : np.ndarray
    """
    domain_lo, domain_hi = 1.0, float(M)
    coeff_knots = _make_coeff_basis_knots(domain_lo, domain_hi, L)
    S_coeff = compute_roughness_penalty_matrix(
        coeff_knots, degree=_DEG, domain_start=domain_lo, domain_end=domain_hi
    )
    S_pred = compute_roughness_penalty_matrix(
        pred_knots, degree=_DEG, domain_start=domain_lo, domain_end=domain_hi
    )
    phi_params = {"knots": coeff_knots, "degree": _DEG}
    psi_params = {"knots": pred_knots,  "degree": _DEG}
    J = compute_inner_product_matrix(
        phi_params, psi_params, domain_start=domain_lo, domain_end=domain_hi
    )
    return J, S_coeff, S_pred, coeff_knots


def _impute_nan(arr: np.ndarray, col_means: np.ndarray | None = None) -> np.ndarray:
    arr = arr.copy()
    if col_means is None:
        col_means = np.nanmean(arr, axis=0)
    for j in range(arr.shape[1]):
        mask = np.isnan(arr[:, j])
        if mask.any():
            fill = col_means[j] if np.isfinite(col_means[j]) else 0.0
            arr[mask, j] = fill
    return arr


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_exp2() -> None:
    print("=" * 70)
    print("Experiment 2 – Month-End-Only Functional Logistic Regression")
    print("=" * 70)

    # ── Data pipeline (same as exp_1) ─────────────────────────────────────
    print("\n[Step 1] Generating synthetic data …")
    raw = generate_synthetic_banking_data(
        n_borrowers=CONFIG["n_synthetic_borrowers"],
        seed=CONFIG["random_seed"],
    )
    accounts_df = raw["accounts"]
    balances_df = raw["balances"]
    defaults_df = raw["defaults"]

    print("[Step 2] Building calendar …")
    banking_days   = generate_banking_days("2016-01-01", "2025-01-01")
    month_end_days = get_month_end_banking_days("2016-01-01", "2025-01-01")

    print("[Step 3] Reconstructing EOD states …")
    eod_df = reconstruct_eod_states(accounts_df, balances_df, banking_days)
    if len(eod_df) == 0:
        print("[ERROR] EOD DataFrame is empty – aborting.")
        return

    print("[Step 4] Account-level daily variables …")
    acct_daily = compute_account_level_daily(eod_df)
    del eod_df  # free memory

    print("[Step 5] Aggregating to borrower / household level …")
    bw_daily = aggregate_to_borrower_level(acct_daily)
    hh_daily = aggregate_to_household_level(acct_daily, accounts_df)
    del acct_daily  # free memory

    print("[Step 6] Assigning sampling dates …")
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

    print("[Step 7] Creating default labels …")
    labeled_df = create_default_labels(
        sampling_df, defaults_df,
        horizon_months=CONFIG["default_horizon_months"],
    )

    print("[Step 8] Splitting dev into train / val / test …")
    dev_df = labeled_df[labeled_df["sample_type"] == "dev"].reset_index(drop=True)
    oot_df = labeled_df[labeled_df["sample_type"] == "oot"].reset_index(drop=True)

    n_total = len(dev_df)
    n_train = int(np.floor(n_total * TRAIN_FRAC))
    n_val   = int(np.floor(n_total * VAL_FRAC))

    rng = np.random.default_rng(CONFIG["random_seed"])
    idx = rng.permutation(n_total)
    train_df = dev_df.iloc[idx[:n_train]].reset_index(drop=True)
    val_df   = dev_df.iloc[idx[n_train : n_train + n_val]].reset_index(drop=True)
    test_df  = dev_df.iloc[idx[n_train + n_val :]].reset_index(drop=True)
    print(f"  Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}  OOT: {len(oot_df)}")

    # ── Scalar logistic baseline ───────────────────────────────────────────
    print("\n[Step 9] Building scalar features for logistic baseline …")
    X_train_raw = build_scalar_features(
        bw_daily, hh_daily, train_df[["borrower_id", "sampling_date"]],
        banking_days, month_end_days,
    )
    X_val_raw  = build_scalar_features(
        bw_daily, hh_daily, val_df[["borrower_id", "sampling_date"]],
        banking_days, month_end_days,
    )
    X_test_raw = build_scalar_features(
        bw_daily, hh_daily, test_df[["borrower_id", "sampling_date"]],
        banking_days, month_end_days,
    )

    feature_cols = list(X_train_raw.columns)

    # Winsorize using training thresholds
    winsor_thresholds: dict = {}
    for col in feature_cols:
        lo, hi = winsorize_series(
            X_train_raw[col],
            lower_pct=CONFIG["winsorization_lower"],
            upper_pct=CONFIG["winsorization_upper"],
        )
        winsor_thresholds[col] = (lo, hi)

    X_train_w = apply_winsorization(X_train_raw.reset_index(), feature_cols, winsor_thresholds)
    X_val_w   = apply_winsorization(X_val_raw.reset_index(),   feature_cols, winsor_thresholds)
    X_test_w  = apply_winsorization(X_test_raw.reset_index(),  feature_cols, winsor_thresholds)

    def _align(feat_df, label_df):
        merged = label_df[["borrower_id", "default_label"]].merge(
            feat_df, on="borrower_id", how="left"
        )
        return merged[feature_cols].fillna(0.0), merged["default_label"].to_numpy(dtype=np.float64)

    X_train, y_train = _align(X_train_w, train_df)
    X_val,   y_val   = _align(X_val_w,   val_df)
    X_test,  y_test  = _align(X_test_w,  test_df)

    print("[Step 10] Fitting scalar LogisticPDModel …")
    logistic_model = LogisticPDModel()
    logistic_model.fit(
        X_train, y_train, X_val, y_val,
        auc_threshold=CONFIG["auc_improvement_threshold"],
        max_features=15,
        random_state=CONFIG["random_seed"],
    )
    pred_test_logistic = logistic_model.predict_proba(X_test)
    auc_logistic = compute_all_metrics(y_test, pred_test_logistic)["auc"]
    print(f"  Scalar logistic test AUC: {auc_logistic:.4f}")

    # ── Month-end functional predictors ───────────────────────────────────
    print("\n[Step 11] Building month-end functional predictors …")
    me_terms_all_M = []
    for M in CONFIG["functional_horizons"]:         # [3, 6, 9, 12]
        for t_name, variable, level in _FUNC_SPEC:
            name = f"{t_name}_me_M{M}"
            print(f"  Building '{name}' …")

            def _fp(sdf, _M=M, _var=variable, _lvl=level):
                return _build_month_end_coeff_matrix(
                    bw_daily, hh_daily,
                    sdf[["borrower_id", "sampling_date"]],
                    banking_days, month_end_days,
                    variable=_var, level=_lvl, M=_M,
                )

            C_train, knots, K = _fp(train_df)
            C_val,   _,     _ = _fp(val_df)
            C_test,  _,     _ = _fp(test_df)

            col_means = np.nanmean(C_train, axis=0)
            C_train = _impute_nan(C_train, col_means)
            C_val   = _impute_nan(C_val,   col_means)
            C_test  = _impute_nan(C_test,  col_means)

            J, S_coeff, S_pred, coeff_knots = _build_me_term_matrices(knots, K, M)
            me_terms_all_M.append({
                "name":       name,
                "C_train":    C_train,
                "C_val":      C_val,
                "C_test":     C_test,
                "J":          J,
                "S_pred":     S_pred,
                "S_coeff":    S_coeff,
                "domain_end": float(M),
                "M":          M,
            })

    print(f"  Built {len(me_terms_all_M)} month-end functional terms.")

    # ── Fit month-end FunctionalLogisticModel ─────────────────────────────
    print("\n[Step 12] Fitting month-end FunctionalLogisticModel …")
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
        for td in me_terms_all_M
    ]

    func_model = FunctionalLogisticModel()
    pred_test_func = None
    try:
        func_model.fit(
            terms_list=terms_list,
            y_train=y_train,
            y_val=y_val,
            auc_threshold=CONFIG["auc_improvement_threshold"],
            L=_L,
        )
        sel = func_model.selected_terms_
        print(f"  Selected terms: {sel}")

        # Score on test
        pred_terms_test = [
            {"name": td["name"], "C": td["C_test"], "J": td["J"]}
            for td in me_terms_all_M
        ]
        pred_test_func = func_model.predict_proba(pred_terms_test)
        auc_func = compute_all_metrics(y_test, pred_test_func)["auc"]
        print(f"  Month-end functional test AUC: {auc_func:.4f}")
    except Exception as exc:
        print(f"  [WARN] FunctionalLogisticModel failed: {exc}")
        pred_test_func = None
        auc_func = float("nan")

    # ── DeLong test ───────────────────────────────────────────────────────
    delta_auc = z_stat = p_value = float("nan")
    if pred_test_func is not None:
        print("\n[Step 13] DeLong test: month-end functional vs logistic …")
        try:
            auc1, auc2, delta_auc, z_stat, p_value = delong_auc_variance(
                y_test, pred_test_func, pred_test_logistic
            )
            print(f"  AUC functional = {auc1:.4f}  AUC logistic = {auc2:.4f}")
            print(f"  ΔAUC = {delta_auc:+.4f}  Z = {z_stat:.3f}  p = {p_value:.4f}")
            # Compare to paper reference
            print("\n  Paper reference: ΔAUC=0.0019, Z=0.832, p=0.4052")
        except Exception as exc:
            print(f"  [WARN] DeLong test failed: {exc}")

    # ── Save results ──────────────────────────────────────────────────────
    print("\n[Step 14] Saving results …")
    metrics_rows = [
        {"model": "logistic",   "split": "test", "auc": auc_logistic},
        {"model": "me_functional", "split": "test", "auc": auc_func},
    ]
    delong_rows = [{
        "comparison":    "test_me_functional_vs_logistic",
        "auc_functional": auc_func,
        "auc_logistic":   auc_logistic,
        "delta_auc":      delta_auc,
        "z_stat":         z_stat,
        "p_value":        p_value,
        "paper_delta_auc": 0.0019,
        "paper_z":        0.832,
        "paper_p":        0.4052,
    }]

    pd.DataFrame(metrics_rows).to_csv(RESULTS_DIR / "exp2_metrics.csv", index=False)
    pd.DataFrame(delong_rows).to_csv(RESULTS_DIR / "exp2_delong.csv", index=False)

    # ── AUC comparison plot ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    models  = ["Scalar Logistic", "Month-End Functional"]
    aucs    = [auc_logistic, auc_func]
    colors  = ["steelblue", "seagreen"]
    bars    = ax.bar(models, aucs, color=colors)
    ax.set_ylim(max(0.5, min(aucs) - 0.03), min(1.0, max(aucs) + 0.03))
    ax.set_ylabel("ROC AUC")
    ax.set_title("Exp 2 – Month-End Functional vs Logistic (Test Set)")
    for bar, val in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10)
    ax.axhline(0.8920, color="red", linestyle="--", linewidth=1.2,
               label="Paper target (0.8920)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "exp2_auc_comparison.png", dpi=150)
    plt.close(fig)

    # ── Coefficient function plots ─────────────────────────────────────────
    if func_model is not None and func_model.selected_terms_:
        coeff_funcs = {}
        try:
            coeff_funcs = func_model.get_coefficient_functions(
                grid_points_per_term=CONFIG["bootstrap_dense_grid_points"]
            )
        except Exception as exc:
            print(f"  [WARN] Could not retrieve coefficient functions: {exc}")

        if coeff_funcs:
            n_plots = len(coeff_funcs)
            fig, axes = plt.subplots(1, max(n_plots, 1), figsize=(5 * max(n_plots, 1), 4))
            if n_plots == 1:
                axes = [axes]
            for ax, (tname, cf) in zip(axes, coeff_funcs.items()):
                ax.plot(cf["t_grid"], cf["beta"], color="seagreen", linewidth=2)
                ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
                ax.set_xlabel("Relative month τ")
                ax.set_ylabel("β(τ)")
                ax.set_title(tname.replace("_", " "))
                ax.grid(True, alpha=0.3)
            fig.suptitle("Exp 2 – Month-End Functional Coefficient Functions", fontsize=12)
            plt.tight_layout()
            fig.savefig(RESULTS_DIR / "exp2_coeff_functions.png", dpi=150)
            plt.close(fig)
            print("  Coefficient functions saved to exp2_coeff_functions.png")

    print(f"\n  Results saved to {RESULTS_DIR}")
    print("\n" + "=" * 70)
    print("Experiment 2 complete.")
    print("=" * 70)


if __name__ == "__main__":
    run_exp2()

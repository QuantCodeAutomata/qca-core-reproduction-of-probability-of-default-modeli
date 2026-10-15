"""Experiment 3 – Bootstrap confidence bands for functional coefficient functions.

Procedure
---------
1. Build full pipeline identical to exp_1 through to training data with
   functional predictors.
2. Fit the full functional logistic model on training data; estimate λ_β by REML.
3. Fix λ_β at the REML estimate.
4. Draw B bootstrap resamples from the training data.
5. For each resample:
   a. Re-build functional predictors (same terms, same λ_x, same basis).
   b. Refit the model with λ_β FIXED (no REML re-estimation).
   c. Evaluate β̂(s) on a dense grid of 200 points per term.
   d. Store the coefficient values.
6. Compute pointwise 5th/95th (90 % CI) and 2.5th/97.5th (95 % CI) percentiles.
7. Plot coefficient functions with confidence bands.

Use --fast to run B = 50 resamples instead of CONFIG['n_bootstrap'] = 1500.

Run from the project root:
    python exp/exp_3_bootstrap.py [--fast]
"""

from __future__ import annotations

import sys
import argparse
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
    from joblib import Parallel, delayed
    _HAS_JOBLIB = True
except ImportError:
    _HAS_JOBLIB = False

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
    from src.models.functional_logistic import FunctionalLogisticModel
    from data.synthetic_generator import generate_synthetic_banking_data
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    sys.exit(1)

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

_TOTAL_DEV = CONFIG["train_size"] + CONFIG["val_size"] + CONFIG["test_size"]
TRAIN_FRAC = CONFIG["train_size"] / _TOTAL_DEV
VAL_FRAC   = CONFIG["val_size"]   / _TOTAL_DEV

_DEG = 3
_L   = CONFIG["L_coeff_basis"]  # 5

# Same 3 terms as exp_1
TERM_SPECS = [
    ("hh_gearing_M9",             "gearing",             "hh", 9,  "cubic_spline", 1e-4),
    ("bw_relative_overdraft_M12", "relative_overdraft",  "bw", 12, "mean_spline",  0.0),
    ("hh_available_funds_M3",     "available_funds",     "hh", 3,  "cubic_spline", 1.0),
]

N_GRID = CONFIG["bootstrap_dense_grid_points"]  # 200


# ---------------------------------------------------------------------------
# Helpers (shared with exp_1)
# ---------------------------------------------------------------------------

def _make_coeff_knots(domain_end: float, L: int, deg: int = _DEG) -> np.ndarray:
    n_int = L - deg - 1
    interior = np.linspace(0.0, domain_end, n_int + 2)[1:-1] if n_int > 0 else np.array([])
    return np.concatenate([np.zeros(deg + 1), interior, np.full(deg + 1, domain_end)])


def _build_term_matrices(basis_info: dict, L: int = _L) -> tuple:
    """Build J, S_coeff, S_pred, coeff_knots for a functional term."""
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
            basis_info["knots"], degree=basis_info["degree"], domain_end=domain_end
        )
    elif basis_info["type"] == "mean_spline":
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
    arr = arr.copy()
    if col_means is None:
        col_means = np.nanmean(arr, axis=0)
    for j in range(arr.shape[1]):
        mask = np.isnan(arr[:, j])
        if mask.any():
            fill = col_means[j] if np.isfinite(col_means[j]) else 0.0
            arr[mask, j] = fill
    return arr


def _fp(bw_daily, hh_daily, sdf, banking_days, month_end_days,
        variable, level, M, lambda_x, basis_type):
    return build_functional_predictors(
        bw_daily, hh_daily, sdf[["borrower_id", "sampling_date"]],
        banking_days, month_end_days,
        variable=variable, level=level, M=M,
        lambda_x=lambda_x, basis_type=basis_type,
    )


# ---------------------------------------------------------------------------
# Single bootstrap iteration
# ---------------------------------------------------------------------------

def _run_one_bootstrap(
    b: int,
    n_train: int,
    C_train_list: list,      # list of (n_train, K) arrays, one per term
    y_train: np.ndarray,
    J_list: list,            # list of (L, K) inner-product matrices
    S_coeff_list: list,      # list of (L, L) penalty matrices
    selected_indices: list,  # indices into term specs for selected terms
    S_block_full: np.ndarray,
    lambda_beta_fixed: float,
    coeff_knots_list: list,  # one per selected term
    domain_ends: list,
    seed_offset: int,
) -> dict | None:
    """Fit the model on one bootstrap resample and return per-term beta grids."""
    rng = np.random.default_rng(CONFIG["bootstrap_seed"] + seed_offset + b)
    boot_idx = rng.integers(0, n_train, size=n_train)

    y_boot = y_train[boot_idx]
    # Check that both classes are present
    if y_boot.sum() == 0 or y_boot.sum() == n_train:
        return None

    # Build combined design matrix over selected terms only
    tilde_X_blocks = []
    for i, tidx in enumerate(selected_indices):
        C_boot = C_train_list[tidx][boot_idx]   # (n_train, K)
        J = J_list[tidx]                          # (L, K)
        tilde_X_blocks.append(C_boot @ J.T)      # (n_train, L)

    tilde_X_boot = np.hstack(tilde_X_blocks)     # (n_train, total_L)

    # Refit with fixed lambda_beta (no REML)
    try:
        model_tmp = FunctionalLogisticModel()
        _, b_vec, _ = model_tmp._fit_penalized_logistic_irls(
            tilde_X_boot, y_boot, S_block_full, lambda_beta_fixed
        )
    except Exception:
        return None

    # Evaluate coefficient functions on dense grid
    result = {}
    cursor = 0
    for i, tidx in enumerate(selected_indices):
        b_term = b_vec[cursor : cursor + _L]
        cursor += _L

        domain_end = domain_ends[i]
        coeff_knots = coeff_knots_list[i]
        t_grid = np.linspace(0.0, domain_end, N_GRID)

        # Evaluate coefficient function on grid
        Phi = np.zeros((N_GRID, _L), dtype=np.float64)
        for ll in range(_L):
            c = np.zeros(_L, dtype=np.float64)
            c[ll] = 1.0
            spl = BSpline(coeff_knots, c, _DEG, extrapolate=True)
            Phi[:, ll] = spl(t_grid)
        beta_grid = Phi @ b_term

        term_name = TERM_SPECS[tidx][0]
        result[term_name] = {"t_grid": t_grid, "beta": beta_grid}

    return result


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_exp3(fast: bool = False) -> None:
    B = 50 if fast else CONFIG["n_bootstrap"]
    print("=" * 70)
    print(f"Experiment 3 – Bootstrap Coefficient Confidence Bands  (B={B})")
    if fast:
        print("  [FAST MODE] Using B=50 resamples.")
    print("=" * 70)

    # ── Pipeline (same as exp_1) ───────────────────────────────────────────
    print("\n[Pipeline] Generating synthetic data and building training set …")
    raw = generate_synthetic_banking_data(
        n_borrowers=CONFIG["n_synthetic_borrowers"],
        seed=CONFIG["random_seed"],
    )
    accounts_df = raw["accounts"]
    balances_df = raw["balances"]
    defaults_df = raw["defaults"]

    banking_days   = generate_banking_days("2016-01-01", "2025-01-01")
    month_end_days = get_month_end_banking_days("2016-01-01", "2025-01-01")

    eod_df = reconstruct_eod_states(accounts_df, balances_df, banking_days)
    if len(eod_df) == 0:
        print("[ERROR] EOD DataFrame empty – aborting.")
        return
    acct_daily = compute_account_level_daily(eod_df)
    del eod_df  # free memory
    bw_daily   = aggregate_to_borrower_level(acct_daily)
    hh_daily   = aggregate_to_household_level(acct_daily, accounts_df)
    del acct_daily  # free memory

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
    labeled_df = create_default_labels(
        sampling_df, defaults_df, horizon_months=CONFIG["default_horizon_months"]
    )

    dev_df = labeled_df[labeled_df["sample_type"] == "dev"].reset_index(drop=True)
    n_total = len(dev_df)
    n_train = int(np.floor(n_total * TRAIN_FRAC))
    n_val   = int(np.floor(n_total * VAL_FRAC))

    rng_split = np.random.default_rng(CONFIG["random_seed"])
    idx = rng_split.permutation(n_total)
    train_df = dev_df.iloc[idx[:n_train]].reset_index(drop=True)
    val_df   = dev_df.iloc[idx[n_train : n_train + n_val]].reset_index(drop=True)
    print(f"  Train: {len(train_df)}  Val: {len(val_df)}")

    # ── Build functional predictors for all 3 terms ───────────────────────
    print("\n[Step 1] Building functional predictors for all 3 terms …")
    C_train_list = []
    C_val_list   = []
    J_list       = []
    S_coeff_list = []
    S_pred_list  = []
    coeff_knots_list = []
    domain_ends  = []
    col_means_list   = []
    basis_infos      = []

    for t_name, var, lvl, M, btype, lx in TERM_SPECS:
        print(f"  '{t_name}' …")
        C_tr, bi = _fp(bw_daily, hh_daily, train_df, banking_days, month_end_days,
                       var, lvl, M, lx, btype)
        C_vl, _  = _fp(bw_daily, hh_daily, val_df, banking_days, month_end_days,
                       var, lvl, M, lx, btype)

        col_means = np.nanmean(C_tr, axis=0)
        C_tr = _impute_nan(C_tr, col_means)
        C_vl = _impute_nan(C_vl, col_means)

        J, S_coeff, S_pred, c_knots = _build_term_matrices(bi, L=_L)
        C_train_list.append(C_tr)
        C_val_list.append(C_vl)
        J_list.append(J)
        S_coeff_list.append(S_coeff)
        S_pred_list.append(S_pred)
        coeff_knots_list.append(c_knots)
        domain_ends.append(float(bi["domain"][1]))
        col_means_list.append(col_means)
        basis_infos.append(bi)

    y_train = train_df["default_label"].to_numpy(dtype=np.float64)
    y_val   = val_df["default_label"].to_numpy(dtype=np.float64)

    # ── Step 2: Fit full model, estimate lambda_beta by REML ─────────────
    print("\n[Step 2] Fitting full functional logistic model (REML) …")
    terms_list = [
        {
            "name":       TERM_SPECS[i][0],
            "C_train":    C_train_list[i],
            "C_val":      C_val_list[i],
            "J":          J_list[i],
            "S_pred":     S_pred_list[i],
            "S_coeff":    S_coeff_list[i],
            "domain_end": domain_ends[i],
        }
        for i in range(len(TERM_SPECS))
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
    except Exception as exc:
        print(f"  [ERROR] Full model fit failed: {exc}")
        return

    lambda_beta_fixed = func_model.lambda_beta_
    selected_terms    = func_model.selected_terms_
    print(f"  Selected terms: {selected_terms}")
    print(f"  λ_β (fixed for bootstrap): {lambda_beta_fixed:.6f}")

    if not selected_terms:
        print("[WARN] No terms selected – cannot run bootstrap.")
        return

    # Indices into TERM_SPECS for selected terms
    term_names = [spec[0] for spec in TERM_SPECS]
    selected_indices = [term_names.index(n) for n in selected_terms if n in term_names]
    n_sel = len(selected_indices)

    # Block-diagonal penalty for selected terms
    total_L = n_sel * _L
    S_block_full = np.zeros((total_L, total_L), dtype=np.float64)
    for k, tidx in enumerate(selected_indices):
        s, e = k * _L, (k + 1) * _L
        S_block_full[s:e, s:e] = S_coeff_list[tidx]

    sel_coeff_knots = [coeff_knots_list[i] for i in selected_indices]
    sel_domain_ends = [domain_ends[i] for i in selected_indices]

    # ── Step 3: Run bootstrap ─────────────────────────────────────────────
    print(f"\n[Step 3] Running {B} bootstrap resamples …")

    n_train_sz = len(y_train)

    def _run_b(b_idx):
        return _run_one_bootstrap(
            b=b_idx,
            n_train=n_train_sz,
            C_train_list=C_train_list,
            y_train=y_train,
            J_list=J_list,
            S_coeff_list=S_coeff_list,
            selected_indices=selected_indices,
            S_block_full=S_block_full,
            lambda_beta_fixed=lambda_beta_fixed,
            coeff_knots_list=sel_coeff_knots,
            domain_ends=sel_domain_ends,
            seed_offset=0,
        )

    if _HAS_JOBLIB and not fast:
        results = Parallel(n_jobs=-1, backend="loky", verbose=5)(
            delayed(_run_b)(b) for b in range(B)
        )
    else:
        results = []
        for b in range(B):
            if (b + 1) % max(1, B // 10) == 0:
                print(f"  Bootstrap {b + 1}/{B} …")
            results.append(_run_b(b))

    valid_results = [r for r in results if r is not None]
    print(f"  Completed {len(valid_results)}/{B} valid resamples.")

    if len(valid_results) < 10:
        print("[ERROR] Too few valid bootstrap resamples – aborting.")
        return

    # ── Step 4: Aggregate bootstrap distributions ─────────────────────────
    print("\n[Step 4] Computing pointwise confidence bands …")
    # Organise into (B_valid, N_GRID) arrays per selected term
    boot_betas: dict[str, np.ndarray] = {n: [] for n in selected_terms}
    boot_grids: dict[str, np.ndarray] = {}

    for res in valid_results:
        for name in selected_terms:
            if name in res:
                boot_betas[name].append(res[name]["beta"])
                boot_grids[name] = res[name]["t_grid"]

    for name in selected_terms:
        boot_betas[name] = np.vstack(boot_betas[name])  # (B_valid, N_GRID)

    # ── Step 5: Plot with confidence bands ───────────────────────────────
    print("[Step 5] Plotting coefficient functions with confidence bands …")
    # Get point-estimate coefficient functions from the fitted model
    point_est = func_model.get_coefficient_functions(N_GRID)

    n_terms = len(selected_terms)
    fig, axes = plt.subplots(1, max(n_terms, 1), figsize=(6 * max(n_terms, 1), 5))
    if n_terms == 1:
        axes = [axes]

    summary_rows = []
    for ax, name in zip(axes, selected_terms):
        t_grid = boot_grids.get(name, np.linspace(0.0, 1.0, N_GRID))
        betas  = boot_betas[name]         # (B_valid, N_GRID)
        p_lo95, p_hi95 = np.percentile(betas, [2.5,  97.5], axis=0)
        p_lo90, p_hi90 = np.percentile(betas, [5.0,  95.0], axis=0)
        beta_hat = point_est.get(name, {}).get("beta", np.nanmean(betas, axis=0))

        ax.fill_between(t_grid, p_lo95, p_hi95, alpha=0.18, color="steelblue", label="95 % CI")
        ax.fill_between(t_grid, p_lo90, p_hi90, alpha=0.32, color="steelblue", label="90 % CI")
        ax.plot(t_grid, beta_hat, color="steelblue", linewidth=2, label="Point estimate")
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Relative time τ")
        ax.set_ylabel("β(τ)")
        ax.set_title(name.replace("_", " "))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Summary stats
        for t_idx in np.linspace(0, N_GRID - 1, 10, dtype=int):
            summary_rows.append({
                "term":    name,
                "tau":     float(t_grid[t_idx]),
                "beta_hat": float(beta_hat[t_idx]),
                "ci90_lo": float(p_lo90[t_idx]),
                "ci90_hi": float(p_hi90[t_idx]),
                "ci95_lo": float(p_lo95[t_idx]),
                "ci95_hi": float(p_hi95[t_idx]),
            })

    fig.suptitle(
        f"Exp 3 – Bootstrap CI for Functional Coefficient Functions (B={len(valid_results)})",
        fontsize=12,
    )
    plt.tight_layout()
    plot_path = RESULTS_DIR / "exp3_bootstrap_coeff_functions.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved to {plot_path}")

    # ── Save summary CSV ──────────────────────────────────────────────────
    summary_df = pd.DataFrame(summary_rows)
    csv_path = RESULTS_DIR / "exp3_bootstrap_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"  Summary CSV saved to {csv_path}")

    # Also save full bootstrap arrays (optional – large)
    # np.save(RESULTS_DIR / "exp3_boot_betas.npy", boot_betas)

    print("\n" + "=" * 70)
    print(f"Experiment 3 complete.  B={len(valid_results)} valid resamples.")
    print(f"  Results directory: {RESULTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 3 – Bootstrap CI")
    parser.add_argument("--fast", action="store_true",
                        help="Use B=50 resamples instead of CONFIG['n_bootstrap']=1500")
    args = parser.parse_args()
    run_exp3(fast=args.fast)

"""Experiment 4 – Alternative functional basis robustness study.

Compares four functional logistic regression models that differ only in the
choice of basis used to represent the predictor paths and/or coefficient
functions:

  Benchmark   : cubic B-spline predictor  ×  cubic B-spline coefficient
  Fourier     : Fourier predictor          ×  cubic B-spline coefficient
  Polynomial  : polynomial predictor       ×  cubic B-spline coefficient
  Mixed       : cubic B-spline predictor  ×  Fourier coefficient

All four models use the same three functional terms selected in exp_1:
  1. hh_gearing_M9
  2. bw_relative_overdraft_M12
  3. hh_available_funds_M3

Paper reference AUCs (test set):
  Benchmark   0.8996
  Fourier     0.8989
  Polynomial  0.8984
  Mixed       0.8989

Run from the project root:
    python exp/exp_4_basis_robustness.py
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
    from src.calendar_utils import generate_banking_days, get_month_end_banking_days, get_banking_days_in_month
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
    from src.models.functional_logistic import FunctionalLogisticModel
    from src.evaluation import compute_all_metrics
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

# Terms from exp_1
TERM_SPECS = [
    ("hh_gearing_M9",             "gearing",            "hh", 9,  "cubic_spline", 1e-4),
    ("bw_relative_overdraft_M12", "relative_overdraft", "bw", 12, "mean_spline",  0.0),
    ("hh_available_funds_M3",     "available_funds",    "hh", 3,  "cubic_spline", 1.0),
]

# Column mapping for level
_HH_COL = {
    "gearing":            "hh_gearing",
    "relative_overdraft": "hh_relative_overdraft",
    "available_funds":    "hh_available_funds",
    "overdraft":          "hh_overdraft",
    "utilization":        "hh_utilization",
}


# ---------------------------------------------------------------------------
# Quadrature helper
# ---------------------------------------------------------------------------

def _gauss_quad(lo: float, hi: float, n: int = 100) -> tuple:
    xi, wi = np.polynomial.legendre.leggauss(n)
    s   = 0.5 * (hi - lo) * xi + 0.5 * (hi + lo)
    w   = 0.5 * (hi - lo) * wi
    return s, w


# ---------------------------------------------------------------------------
# Cubic B-spline coefficient knots (shared across models)
# ---------------------------------------------------------------------------

def _make_clamped_knots(domain_end: float, n_basis: int, deg: int = _DEG) -> np.ndarray:
    n_int = n_basis - deg - 1
    interior = np.linspace(0.0, domain_end, n_int + 2)[1:-1] if n_int > 0 else np.array([])
    return np.concatenate([np.zeros(deg + 1), interior, np.full(deg + 1, domain_end)])


def _eval_bspline(t: np.ndarray, knots: np.ndarray, n_basis: int, deg: int = _DEG) -> np.ndarray:
    B = np.zeros((len(t), n_basis), dtype=np.float64)
    for k in range(n_basis):
        c = np.zeros(n_basis, dtype=np.float64)
        c[k] = 1.0
        spl = BSpline(knots, c, deg, extrapolate=False)
        vals = spl(t)
        B[:, k] = np.where(np.isnan(vals), 0.0, vals)
    return B


# ---------------------------------------------------------------------------
# Fourier basis
# ---------------------------------------------------------------------------

def _fourier_basis_matrix(t: np.ndarray, M: float, n_terms: int) -> np.ndarray:
    """Evaluate Fourier basis at points t ∈ [0, M].

    Basis: {1, sin(2π·k·t/M), cos(2π·k·t/M), k=1,...,K_f}
    n_terms = 2·K_f + 1 (must be odd; rounded down if even).

    Returns matrix of shape (len(t), n_terms).
    """
    K_f = (n_terms - 1) // 2
    n_f = 2 * K_f + 1
    B = np.zeros((len(t), n_f), dtype=np.float64)
    B[:, 0] = 1.0
    for k in range(1, K_f + 1):
        B[:, 2 * k - 1] = np.sin(2 * np.pi * k * t / M)
        B[:, 2 * k]     = np.cos(2 * np.pi * k * t / M)
    return B


def _fourier_roughness_penalty(M: float, n_terms: int) -> np.ndarray:
    """Analytical integrated squared-second-derivative penalty for Fourier basis.

    For constant term: second derivative = 0.
    For sin/cos at frequency k: second derivative = -(2π·k/M)^2 · sin/cos.
    Diagonal penalty: S_{ll} = (2π·k/M)^4 · M/2 for sin/cos terms.
    """
    K_f = (n_terms - 1) // 2
    n_f = 2 * K_f + 1
    S = np.zeros((n_f, n_f), dtype=np.float64)
    for k in range(1, K_f + 1):
        freq4 = (2 * np.pi * k / M) ** 4
        S[2 * k - 1, 2 * k - 1] = freq4 * M / 2  # sin
        S[2 * k,     2 * k]     = freq4 * M / 2  # cos
    return S


# ---------------------------------------------------------------------------
# Polynomial basis
# ---------------------------------------------------------------------------

def _poly_basis_matrix(t: np.ndarray, degree: int) -> np.ndarray:
    """Evaluate polynomial basis [1, t, t^2, ..., t^d] at points t."""
    n_pts = len(t)
    n_b   = degree + 1
    B = np.zeros((n_pts, n_b), dtype=np.float64)
    for k in range(n_b):
        B[:, k] = t ** k
    return B


def _poly_roughness_penalty(M: float, degree: int, n_quad: int = 200) -> np.ndarray:
    """Numerically compute roughness penalty for polynomial basis on [0, M]."""
    n_b = degree + 1
    s, w = _gauss_quad(0.0, M, n_quad)
    B_dd = np.zeros((len(s), n_b), dtype=np.float64)
    for k in range(n_b):
        if k >= 2:
            B_dd[:, k] = k * (k - 1) * s ** (k - 2)
        # else: second derivative = 0
    S = (B_dd.T * w) @ B_dd
    return S.astype(np.float64)


# ---------------------------------------------------------------------------
# Path data extraction (replicates relative-time logic from feature_engineering)
# ---------------------------------------------------------------------------

def _extract_paths(
    bw_daily: pd.DataFrame,
    hh_daily: pd.DataFrame,
    sampling_df: pd.DataFrame,
    banking_days,
    variable: str,
    level: str,
    M: int,
) -> list:
    """Return list of (tau_arr, y_arr) per borrower in sampling_df row order."""
    col = variable if level == "bw" else _HH_COL[variable]
    daily_df = bw_daily if level == "bw" else hh_daily
    daily_idx = (
        daily_df.copy()
        .assign(date=lambda d: pd.to_datetime(d["date"]).dt.normalize())
        .set_index(["borrower_id", "date"])
    )
    paths = []
    for srow in sampling_df.itertuples(index=False):
        b_id = int(srow.borrower_id)
        T    = pd.Timestamp(srow.sampling_date).normalize()
        tau_list, y_list = [], []
        for m in range(1, M + 1):
            months_before = M - m + 1
            raw_month = T.month - months_before
            t_year  = T.year + (raw_month - 1) // 12
            t_month = (raw_month - 1) % 12 + 1
            m_bdays = get_banking_days_in_month(t_year, t_month, banking_days)
            k_m = len(m_bdays)
            if k_m == 0:
                continue
            for j, d in enumerate(m_bdays, start=1):
                tau = j / k_m + (m - 1)
                try:
                    yv = float(daily_idx.at[(b_id, d.normalize()), col])
                    tau_list.append(tau)
                    y_list.append(yv)
                except KeyError:
                    pass
        paths.append((np.array(tau_list, dtype=np.float64),
                      np.array(y_list,   dtype=np.float64)))
    return paths


def _fit_custom_basis(
    paths: list,
    basis_fn,          # callable(tau_arr) -> B  (n_pts, n_basis)
    n_basis: int,
    min_obs: int = 2,
) -> np.ndarray:
    """Fit a custom predictor basis to each borrower's path.

    Parameters
    ----------
    paths    : list of (tau_arr, y_arr) per borrower
    basis_fn : callable; takes tau_arr and returns design matrix B
    n_basis  : number of basis functions

    Returns
    -------
    C : np.ndarray (n_obs, n_basis) – coefficient matrix (NaN rows for missing)
    """
    coeff_rows = []
    for tau_arr, y_arr in paths:
        if len(tau_arr) < min_obs:
            coeff_rows.append(np.full(n_basis, np.nan, dtype=np.float64))
            continue
        B = basis_fn(tau_arr)
        c, _, _, _ = np.linalg.lstsq(B, y_arr, rcond=None)
        coeff_rows.append(c.astype(np.float64))
    return np.vstack(coeff_rows) if coeff_rows else np.empty((0, n_basis))


# ---------------------------------------------------------------------------
# J matrix computation (numerical, for arbitrary basis pairs)
# ---------------------------------------------------------------------------

def _compute_J_numerical(
    coeff_basis_fn,   # callable(s) -> (n_quad, L)
    pred_basis_fn,    # callable(s) -> (n_quad, K)
    domain_end: float,
    n_quad: int = 200,
) -> np.ndarray:
    """Compute J_{l,k} = integral_0^{domain_end} phi_l(s) psi_k(s) ds."""
    s, w = _gauss_quad(0.0, domain_end, n_quad)
    Phi = coeff_basis_fn(s)   # (n_quad, L)
    Psi = pred_basis_fn(s)    # (n_quad, K)
    J = (Phi.T * w) @ Psi     # (L, K)
    return J.astype(np.float64)


# ---------------------------------------------------------------------------
# Mean-spline J (for benchmark term 2)
# ---------------------------------------------------------------------------

def _compute_mean_spline_J(coeff_knots, L, K, domain_end) -> np.ndarray:
    """J_{l,k} = integral_{k-1}^{k} phi_l^cubic(s) ds, k=0..K-1."""
    J = np.zeros((L, K), dtype=np.float64)
    xi, wi = np.polynomial.legendre.leggauss(50)
    for k in range(K):
        lo, hi = float(k), float(k + 1)
        scale  = 0.5 * (hi - lo)
        shift  = 0.5 * (hi + lo)
        s_q    = scale * xi + shift
        w_q    = scale * wi
        for ll in range(L):
            c = np.zeros(L, dtype=np.float64)
            c[ll] = 1.0
            spl  = BSpline(coeff_knots, c, _DEG, extrapolate=False)
            vals = spl(s_q)
            J[ll, k] = np.dot(w_q, np.where(np.isnan(vals), 0.0, vals))
    return J


# ---------------------------------------------------------------------------
# Imputation
# ---------------------------------------------------------------------------

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
# Build one term for benchmark model (same as exp_1)
# ---------------------------------------------------------------------------

def _build_benchmark_term(
    bw_daily, hh_daily, train_df, val_df, test_df,
    banking_days, month_end_days,
    t_name, variable, level, M, basis_type, lambda_x,
) -> dict:
    """Build one term for the benchmark model (cubic spline, same as exp_1)."""
    def _fp(sdf):
        return build_functional_predictors(
            bw_daily, hh_daily, sdf[["borrower_id", "sampling_date"]],
            banking_days, month_end_days,
            variable=variable, level=level, M=M,
            lambda_x=lambda_x, basis_type=basis_type,
        )

    C_train, bi = _fp(train_df)
    C_val,   _  = _fp(val_df)
    C_test,  _  = _fp(test_df)

    col_means = np.nanmean(C_train, axis=0)
    C_train = _impute_nan(C_train, col_means)
    C_val   = _impute_nan(C_val,   col_means)
    C_test  = _impute_nan(C_test,  col_means)

    domain_end = float(bi["domain"][1])
    coeff_knots = _make_clamped_knots(domain_end, _L)
    S_coeff = compute_roughness_penalty_matrix(
        coeff_knots, degree=_DEG, domain_start=0.0, domain_end=domain_end
    )

    if bi["type"] == "cubic_spline":
        phi_p = {"knots": coeff_knots, "degree": _DEG}
        psi_p = {"knots": bi["knots"], "degree": bi["degree"]}
        J = compute_inner_product_matrix(phi_p, psi_p, domain_start=0.0, domain_end=domain_end)
        S_pred = compute_roughness_penalty_matrix(
            bi["knots"], degree=bi["degree"], domain_end=domain_end
        )
    else:  # mean_spline
        K = bi["n_basis"]
        J = _compute_mean_spline_J(coeff_knots, _L, K, domain_end)
        S_pred = np.eye(K, dtype=np.float64)

    return {"name": t_name, "C_train": C_train, "C_val": C_val, "C_test": C_test,
            "J": J, "S_pred": S_pred, "S_coeff": S_coeff, "domain_end": domain_end}


# ---------------------------------------------------------------------------
# Build one term with a custom predictor basis (Fourier or Polynomial)
# ---------------------------------------------------------------------------

def _build_custom_pred_term(
    bw_daily, hh_daily, train_df, val_df, test_df,
    banking_days,
    t_name, variable, level, M,
    pred_basis_fn,   # callable(tau_arr) -> (n_pts, n_pred_basis)
    n_pred_basis: int,
    domain_end: float,
    pred_J_fn,       # callable(s) -> (n_quad, n_pred_basis)  for J computation
) -> dict:
    """Build one term using a custom predictor basis.

    The coefficient function basis is always cubic B-spline with L=5.
    """
    def _paths(sdf):
        return _extract_paths(bw_daily, hh_daily, sdf, banking_days,
                              variable, level, M)

    paths_tr = _paths(train_df)
    paths_vl = _paths(val_df)
    paths_te = _paths(test_df)

    def _fit(paths):
        return _fit_custom_basis(paths, pred_basis_fn, n_pred_basis)

    C_train = _fit(paths_tr)
    C_val   = _fit(paths_vl)
    C_test  = _fit(paths_te)

    col_means = np.nanmean(C_train, axis=0)
    C_train = _impute_nan(C_train, col_means)
    C_val   = _impute_nan(C_val,   col_means)
    C_test  = _impute_nan(C_test,  col_means)

    # Coefficient basis (cubic B-spline, same for all non-mixed models)
    coeff_knots = _make_clamped_knots(domain_end, _L)
    S_coeff = compute_roughness_penalty_matrix(
        coeff_knots, degree=_DEG, domain_start=0.0, domain_end=domain_end
    )

    # Inner-product matrix J
    def _coeff_basis_fn(s):
        return _eval_bspline(s, coeff_knots, _L)

    J = _compute_J_numerical(_coeff_basis_fn, pred_J_fn, domain_end)

    S_pred = np.eye(n_pred_basis, dtype=np.float64)

    return {"name": t_name, "C_train": C_train, "C_val": C_val, "C_test": C_test,
            "J": J, "S_pred": S_pred, "S_coeff": S_coeff, "domain_end": domain_end}


# ---------------------------------------------------------------------------
# Build one term for the MIXED model (cubic-spline predictor, Fourier coeff)
# ---------------------------------------------------------------------------

def _build_mixed_term(
    bw_daily, hh_daily, train_df, val_df, test_df,
    banking_days, month_end_days,
    t_name, variable, level, M, basis_type, lambda_x,
    L_fourier: int = _L,
) -> dict:
    """Mixed basis: cubic-spline predictor, Fourier coefficient function."""
    # C matrices: same as benchmark (cubic spline predictor)
    def _fp(sdf):
        return build_functional_predictors(
            bw_daily, hh_daily, sdf[["borrower_id", "sampling_date"]],
            banking_days, month_end_days,
            variable=variable, level=level, M=M,
            lambda_x=lambda_x, basis_type=basis_type,
        )

    C_train, bi = _fp(train_df)
    C_val,   _  = _fp(val_df)
    C_test,  _  = _fp(test_df)

    col_means = np.nanmean(C_train, axis=0)
    C_train = _impute_nan(C_train, col_means)
    C_val   = _impute_nan(C_val,   col_means)
    C_test  = _impute_nan(C_test,  col_means)

    domain_end = float(bi["domain"][1])
    n_fourier  = L_fourier  # number of Fourier coefficient basis functions

    # Fourier coefficient basis
    S_coeff = _fourier_roughness_penalty(domain_end, n_fourier)

    # J: Fourier coefficient × cubic-spline predictor
    if bi["type"] == "cubic_spline":
        pred_knots = bi["knots"]
        K = bi["n_basis"]
        def _fourier_coeff_fn(s): return _fourier_basis_matrix(s, domain_end, n_fourier)
        def _cubic_pred_fn(s):   return _eval_bspline(s, pred_knots, K)
        J = _compute_J_numerical(_fourier_coeff_fn, _cubic_pred_fn, domain_end)
    else:  # mean_spline predictor – Fourier coeff basis × piecewise-constant predictor
        K = bi["n_basis"]
        xi, wi = np.polynomial.legendre.leggauss(50)
        J = np.zeros((n_fourier, K), dtype=np.float64)
        for k in range(K):
            lo, hi = float(k), float(k + 1)
            scale = 0.5 * (hi - lo); shift = 0.5 * (hi + lo)
            s_q = scale * xi + shift; w_q = scale * wi
            Phi_f = _fourier_basis_matrix(s_q, domain_end, n_fourier)  # (n_quad, n_fourier)
            J[:, k] = Phi_f.T @ w_q

    S_pred = np.eye(K, dtype=np.float64)

    return {"name": t_name, "C_train": C_train, "C_val": C_val, "C_test": C_test,
            "J": J, "S_pred": S_pred, "S_coeff": S_coeff, "domain_end": domain_end}


# ---------------------------------------------------------------------------
# Fit and score a functional model
# ---------------------------------------------------------------------------

def _fit_and_score(
    terms: list,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    model_label: str,
) -> dict:
    """Fit FunctionalLogisticModel and return test AUC and selected terms."""
    terms_list = [
        {"name": t["name"], "C_train": t["C_train"], "C_val": t["C_val"],
         "J": t["J"], "S_pred": t["S_pred"], "S_coeff": t["S_coeff"],
         "domain_end": t["domain_end"]}
        for t in terms
    ]
    model = FunctionalLogisticModel()
    try:
        model.fit(
            terms_list=terms_list,
            y_train=y_train,
            y_val=y_val,
            auc_threshold=CONFIG["auc_improvement_threshold"],
            L=_L,
        )
        pred_test = model.predict_proba([
            {"name": t["name"], "C": t["C_test"], "J": t["J"]}
            for t in terms
        ])
        auc = compute_all_metrics(y_test, pred_test)["auc"]
        sel  = model.selected_terms_
        lam  = model.lambda_beta_
        print(f"    [{model_label}] AUC={auc:.4f}  λ_β={lam:.4g}  selected={sel}")
        return {"model": model_label, "auc": auc, "lambda_beta": lam,
                "selected_terms": str(sel), "status": "ok"}
    except Exception as exc:
        print(f"    [{model_label}] FAILED: {exc}")
        return {"model": model_label, "auc": float("nan"), "lambda_beta": float("nan"),
                "selected_terms": "[]", "status": f"error: {exc}"}


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_exp4() -> None:
    print("=" * 70)
    print("Experiment 4 – Alternative Functional Basis Robustness")
    print("=" * 70)

    # ── Full pipeline ──────────────────────────────────────────────────────
    print("\n[Pipeline] Generating synthetic data …")
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
    sampling_df  = assign_sampling_dates(
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

    rng = np.random.default_rng(CONFIG["random_seed"])
    idx = rng.permutation(n_total)
    train_df = dev_df.iloc[idx[:n_train]].reset_index(drop=True)
    val_df   = dev_df.iloc[idx[n_train : n_train + n_val]].reset_index(drop=True)
    test_df  = dev_df.iloc[idx[n_train + n_val :]].reset_index(drop=True)
    print(f"  Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

    y_train = train_df["default_label"].to_numpy(dtype=np.float64)
    y_val   = val_df["default_label"].to_numpy(dtype=np.float64)
    y_test  = test_df["default_label"].to_numpy(dtype=np.float64)

    # ── Model 1: Benchmark (cubic spline, same as exp_1) ─────────────────
    print("\n[Model 1] Benchmark – Cubic B-spline predictor + coefficient …")
    bench_terms = []
    for t_name, variable, level, M, basis_type, lambda_x in TERM_SPECS:
        print(f"  Building '{t_name}' …")
        td = _build_benchmark_term(
            bw_daily, hh_daily, train_df, val_df, test_df,
            banking_days, month_end_days,
            t_name, variable, level, M, basis_type, lambda_x,
        )
        bench_terms.append(td)
        print(f"    C_train={td['C_train'].shape}  J={td['J'].shape}")

    res_bench = _fit_and_score(bench_terms, y_train, y_val, y_test, "benchmark")

    # ── Model 2: Fourier predictor basis ──────────────────────────────────
    print("\n[Model 2] Fourier predictor basis …")
    fourier_terms = []
    for t_name, variable, level, M, _btype, _lx in TERM_SPECS:
        print(f"  Building '{t_name}' (Fourier) …")
        domain_end = float(M)
        # n_fourier: match approximately n_basis = max(6, M+3) of cubic spline
        n_cubic    = max(6, M + 3)
        K_f        = max(2, (n_cubic - 1) // 2)
        n_fourier  = 2 * K_f + 1

        def _pred_fourier(tau, _M=M, _nf=n_fourier):
            return _fourier_basis_matrix(tau, _M, _nf)

        def _pred_fourier_quad(s, _M=M, _nf=n_fourier):
            return _fourier_basis_matrix(s, _M, _nf)

        td = _build_custom_pred_term(
            bw_daily, hh_daily, train_df, val_df, test_df,
            banking_days,
            t_name, variable, level, M,
            pred_basis_fn=_pred_fourier,
            n_pred_basis=n_fourier,
            domain_end=domain_end,
            pred_J_fn=_pred_fourier_quad,
        )
        fourier_terms.append(td)
        print(f"    C_train={td['C_train'].shape}  J={td['J'].shape}  n_fourier={n_fourier}")

    res_fourier = _fit_and_score(fourier_terms, y_train, y_val, y_test, "fourier")

    # ── Model 3: Local polynomial predictor basis ─────────────────────────
    print("\n[Model 3] Local polynomial predictor basis (degree 3) …")
    poly_terms  = []
    POLY_DEGREE = 3
    n_poly      = POLY_DEGREE + 1  # 4 basis functions: 1, t, t^2, t^3

    for t_name, variable, level, M, _btype, _lx in TERM_SPECS:
        print(f"  Building '{t_name}' (polynomial d={POLY_DEGREE}) …")
        domain_end = float(M)

        def _pred_poly(tau, _d=POLY_DEGREE):
            return _poly_basis_matrix(tau, _d)

        def _pred_poly_quad(s, _d=POLY_DEGREE):
            return _poly_basis_matrix(s, _d)

        td = _build_custom_pred_term(
            bw_daily, hh_daily, train_df, val_df, test_df,
            banking_days,
            t_name, variable, level, M,
            pred_basis_fn=_pred_poly,
            n_pred_basis=n_poly,
            domain_end=domain_end,
            pred_J_fn=_pred_poly_quad,
        )
        poly_terms.append(td)
        print(f"    C_train={td['C_train'].shape}  J={td['J'].shape}")

    res_poly = _fit_and_score(poly_terms, y_train, y_val, y_test, "local_polynomial")

    # ── Model 4: Mixed basis (cubic spline predictor, Fourier coefficient) ─
    print("\n[Model 4] Mixed basis – cubic spline predictor × Fourier coefficient …")
    mixed_terms = []
    for t_name, variable, level, M, basis_type, lambda_x in TERM_SPECS:
        print(f"  Building '{t_name}' (mixed) …")
        td = _build_mixed_term(
            bw_daily, hh_daily, train_df, val_df, test_df,
            banking_days, month_end_days,
            t_name, variable, level, M, basis_type, lambda_x,
            L_fourier=_L,
        )
        mixed_terms.append(td)
        print(f"    C_train={td['C_train'].shape}  J={td['J'].shape}")

    res_mixed = _fit_and_score(mixed_terms, y_train, y_val, y_test, "mixed")

    # ── Collate results ───────────────────────────────────────────────────
    print("\n[Results] AUC comparison …")
    paper_refs = {
        "benchmark":       0.8996,
        "fourier":         0.8989,
        "local_polynomial": 0.8984,
        "mixed":           0.8989,
    }
    all_results = [res_bench, res_fourier, res_poly, res_mixed]
    metrics_rows = []
    for res in all_results:
        ref = paper_refs.get(res["model"], float("nan"))
        metrics_rows.append({**res, "paper_auc": ref})
        print(f"  {res['model']:20s}  synthetic={res['auc']:.4f}  paper_ref={ref:.4f}")

    metrics_df = pd.DataFrame(metrics_rows)
    csv_path = RESULTS_DIR / "exp4_metrics.csv"
    metrics_df.to_csv(csv_path, index=False)
    print(f"\n  Saved to {csv_path}")

    # ── Comparison bar chart ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    model_labels = [r["model"] for r in all_results]
    auc_vals     = [r["auc"]   for r in all_results]
    ref_vals     = [paper_refs.get(m, float("nan")) for m in model_labels]
    x = np.arange(len(model_labels))
    width = 0.35

    b1 = ax.bar(x - width / 2, auc_vals, width, label="Synthetic data", color="steelblue")
    b2 = ax.bar(x + width / 2, ref_vals, width, label="Paper reference", color="darkorange",
                alpha=0.7, edgecolor="black", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in model_labels], fontsize=10)
    ax.set_ylabel("Test ROC AUC")
    ax.set_title("Exp 4 – Alternative Basis Robustness (Test AUC Comparison)")
    lo = min(v for v in auc_vals + ref_vals if np.isfinite(v)) - 0.02
    ax.set_ylim(max(0.5, lo), 1.0)
    ax.legend(fontsize=10)

    for bar, val in zip(b1.patches, auc_vals):
        if np.isfinite(val):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plot_path = RESULTS_DIR / "exp4_comparison.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved to {plot_path}")

    print("\n" + "=" * 70)
    print("Experiment 4 complete.")
    print(f"  Results directory: {RESULTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    run_exp4()

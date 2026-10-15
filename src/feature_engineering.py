"""
Scalar and functional feature construction for retail banking PD modelling.

Two feature families are built:

Scalar features  (50 per borrower)
    5 variables × 2 aggregation levels × 5 lookback horizons.
    Each feature summarises the variable at or in the recent months before the
    sampling date T.

Functional predictors  (1 curve per borrower per (variable, level, M) triple)
    The full time series over the M-month lookback window is projected onto a
    finite-dimensional spline basis.  The resulting coefficient vectors form
    the functional design matrix used by functional logistic regression.

Variable catalogue
------------------
gearing, relative_overdraft, available_funds, overdraft, utilization

Levels
------
bw_ : borrower level  (from :func:`src.preprocessing.aggregate_to_borrower_level`)
hh_ : household level (from :func:`src.preprocessing.aggregate_to_household_level`)

Horizons M ∈ {0, 3, 6, 9, 12}
    M=0 : spot value at T
    M>0 : mean of month-end values over the M calendar months before T
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline

from src.calendar_utils import get_banking_days_in_month

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_DEFAULT_VARIABLES: List[str] = [
    "gearing",
    "relative_overdraft",
    "available_funds",
    "overdraft",
    "utilization",
]

_DEFAULT_HORIZONS: List[int] = [0, 3, 6, 9, 12]

# Map plain variable names → household-level column names
_HH_COL: Dict[str, str] = {
    "gearing": "hh_gearing",
    "relative_overdraft": "hh_relative_overdraft",
    "available_funds": "hh_available_funds",
    "overdraft": "hh_overdraft",
    "utilization": "hh_utilization",
}

_DEGREE = 3  # cubic B-splines throughout


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _month_offset(ref_year: int, ref_month: int, months_back: int) -> Tuple[int, int]:
    """Return (year, month) that is *months_back* calendar months before *ref*.

    Parameters
    ----------
    ref_year, ref_month : int
        Reference calendar year and month (1–12).
    months_back : int
        Number of calendar months to step back (≥ 0).

    Returns
    -------
    (int, int)
        (target_year, target_month) in the Gregorian calendar.
    """
    raw = ref_month - months_back
    # Adjust for crossing year boundaries
    target_year = ref_year + (raw - 1) // 12
    target_month = (raw - 1) % 12 + 1
    return int(target_year), int(target_month)


def _build_me_lookup(month_end_days: pd.Series) -> Dict[Tuple[int, int], pd.Timestamp]:
    """Convert month_end_days Series to a fast (year, month) → Timestamp dict."""
    lookup: Dict[Tuple[int, int], pd.Timestamp] = {}
    for period, ts in month_end_days.items():
        p = pd.Period(period, freq="M")
        lookup[(int(p.year), int(p.month))] = pd.Timestamp(ts).normalize()
    return lookup


def _make_clamped_knots(domain_end: float, n_basis: int, degree: int = _DEGREE) -> np.ndarray:
    """Build a clamped B-spline knot vector on [0, domain_end].

    The knot vector has ``degree+1`` repeated knots at each boundary and
    ``n_basis - degree - 1`` equally spaced interior knots.

    Parameters
    ----------
    domain_end : float
        Right boundary of the spline domain (left boundary is 0).
    n_basis : int
        Number of B-spline basis functions.
    degree : int
        Polynomial degree (default 3 for cubic).

    Returns
    -------
    np.ndarray
        Knot vector of length ``n_basis + degree + 1``.

    Raises
    ------
    ValueError
        If ``n_basis < degree + 1``.
    """
    if n_basis < degree + 1:
        raise ValueError(
            f"n_basis must be ≥ degree+1 = {degree+1}, got {n_basis}."
        )
    n_interior = n_basis - degree - 1
    if n_interior > 0:
        interior = np.linspace(0.0, domain_end, n_interior + 2)[1:-1]
    else:
        interior = np.empty(0, dtype=np.float64)

    knots = np.concatenate(
        [np.zeros(degree + 1), interior, np.full(degree + 1, domain_end)]
    )
    return knots.astype(np.float64)


# ---------------------------------------------------------------------------
# 1. Scalar features
# ---------------------------------------------------------------------------

def build_scalar_features(
    borrower_daily_df: pd.DataFrame,
    hh_daily_df: pd.DataFrame,
    sampling_df: pd.DataFrame,
    banking_days: pd.DatetimeIndex,
    month_end_days: pd.Series,
    variables: Optional[List[str]] = None,
    horizons: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Build 50 scalar features for each borrower (5 vars × 2 levels × 5 horizons).

    Feature construction rules
    --------------------------
    * ``M=0`` : value at the sampling date T (spot).
    * ``M>0`` : arithmetic mean of the variable at the **last banking day** of
      each of the M calendar months immediately before T.

    Feature naming convention: ``'{level}_{variable}_M{horizon}'``, e.g.
    ``'bw_gearing_M0'``, ``'hh_utilization_M12'``.

    Parameters
    ----------
    borrower_daily_df : pd.DataFrame
        Borrower-level daily data from
        :func:`src.preprocessing.aggregate_to_borrower_level`.
        Must contain columns: borrower_id, date, gearing, relative_overdraft,
        available_funds, overdraft, utilization.
    hh_daily_df : pd.DataFrame
        Household-level daily data (mapped to borrowers) from
        :func:`src.preprocessing.aggregate_to_household_level`.
        Must contain columns: borrower_id, date, hh_gearing,
        hh_relative_overdraft, hh_available_funds, hh_overdraft,
        hh_utilization.
    sampling_df : pd.DataFrame
        Output of :func:`src.preprocessing.assign_sampling_dates` with columns
        borrower_id and sampling_date.
    banking_days : pd.DatetimeIndex
        Full set of valid banking days.
    month_end_days : pd.Series
        Last banking day per month (PeriodIndex → Timestamp).
    variables : list of str, optional
        Variables to include; defaults to all five.
    horizons : list of int, optional
        Lookback horizons M; defaults to [0, 3, 6, 9, 12].

    Returns
    -------
    pd.DataFrame
        Shape ``(n_borrowers, 50)``, indexed by ``borrower_id``.
        Column order: outer loop over variables, then levels, then horizons.
    """
    if variables is None:
        variables = list(_DEFAULT_VARIABLES)
    if horizons is None:
        horizons = list(_DEFAULT_HORIZONS)

    # Index daily DataFrames for O(1) look-ups
    bw_idx = (
        borrower_daily_df.copy()
        .assign(date=lambda d: pd.to_datetime(d["date"]).dt.normalize())
        .set_index(["borrower_id", "date"])
    )
    hh_idx = (
        hh_daily_df.copy()
        .assign(date=lambda d: pd.to_datetime(d["date"]).dt.normalize())
        .set_index(["borrower_id", "date"])
    )

    me_lookup = _build_me_lookup(month_end_days)

    # Pre-build ordered list of feature column names
    feature_names: List[str] = []
    for var in variables:
        for level_prefix in ("bw_", "hh_"):
            for M in horizons:
                feature_names.append(f"{level_prefix}{var}_M{M}")

    rows: List[dict] = []

    for srow in sampling_df.itertuples(index=False):
        b_id: int = int(srow.borrower_id)
        T: pd.Timestamp = pd.Timestamp(srow.sampling_date).normalize()

        feat: Dict[str, float] = {"borrower_id": b_id}

        for var in variables:
            bw_col = var
            hh_col = _HH_COL[var]

            for level_prefix, df_idx, col in (
                ("bw_", bw_idx, bw_col),
                ("hh_", hh_idx, hh_col),
            ):
                for M in horizons:
                    fname = f"{level_prefix}{var}_M{M}"

                    if M == 0:
                        # Spot value at T
                        try:
                            val: float = float(df_idx.at[(b_id, T), col])
                        except KeyError:
                            val = np.nan
                    else:
                        # Mean over last banking day of each of M months before T
                        vals: List[float] = []
                        for m_back in range(1, M + 1):
                            t_year, t_month = _month_offset(T.year, T.month, m_back)
                            me_ts = me_lookup.get((t_year, t_month))
                            if me_ts is None:
                                continue
                            try:
                                v: float = float(df_idx.at[(b_id, me_ts), col])
                                vals.append(v)
                            except KeyError:
                                pass
                        val = float(np.nanmean(vals)) if vals else np.nan

                    feat[fname] = val

        rows.append(feat)

    if not rows:
        return pd.DataFrame(columns=["borrower_id"] + feature_names).set_index(
            "borrower_id"
        )

    result = pd.DataFrame(rows).set_index("borrower_id")
    # Ensure consistent column order and float64 dtype
    for col in feature_names:
        if col not in result.columns:
            result[col] = np.nan
        result[col] = result[col].astype(np.float64)

    return result[feature_names]


# ---------------------------------------------------------------------------
# 2. Functional predictors
# ---------------------------------------------------------------------------

def build_functional_predictors(
    borrower_daily_df: pd.DataFrame,
    hh_daily_df: pd.DataFrame,
    sampling_df: pd.DataFrame,
    banking_days: pd.DatetimeIndex,
    month_end_days: pd.Series,
    variable: str,
    level: str,
    M: int,
    lambda_x: float = 0.0,
    basis_type: str = "cubic_spline",
    n_basis: Optional[int] = None,
) -> Tuple[np.ndarray, dict]:
    """Build a smoothed functional predictor for one (variable, level, M) triple.

    For each borrower, the daily time series of *variable* over the M-month
    lookback window is mapped to relative time τ ∈ (0, M] and projected onto
    a finite-dimensional basis, yielding a coefficient vector.

    Basis types
    -----------
    ``'cubic_spline'``
        Cubic B-spline with ``n_basis`` basis functions and equally spaced
        interior knots.  If ``lambda_x=0`` the system is solved as ordinary
        least squares (no roughness penalty).  If ``lambda_x>0`` the
        penalised normal equations are solved:
        ``c = (BᵀB + λ S)⁻¹ Bᵀ y``
        where S is the integrated squared-second-derivative penalty matrix.
        Default ``n_basis = max(6, M + 3)``.

    ``'mean_spline'``
        Piecewise constant (step function): one coefficient per calendar month,
        equal to the mean of the variable over all banking days in that month.
        Coefficient vector has dimension M.  ``lambda_x`` is ignored.

    Relative-time mapping
    ---------------------
    Calendar month m (m=1 oldest, m=M most recent) contributes banking days
    indexed j = 1, …, k_m where k_m is the count of banking days in month m.
    The relative time is τ_j^(m) = j / k_m + (m − 1).

    Parameters
    ----------
    borrower_daily_df : pd.DataFrame
        Borrower-level daily data.
    hh_daily_df : pd.DataFrame
        Household-level daily data mapped to borrowers.
    sampling_df : pd.DataFrame
        Columns borrower_id and sampling_date.
    banking_days : pd.DatetimeIndex
        Full set of valid banking days.
    month_end_days : pd.Series
        Last banking day per month (unused in this function but kept for API
        consistency).
    variable : str
        One of: gearing, relative_overdraft, available_funds, overdraft,
        utilization.
    level : str
        ``'bw'`` for borrower level or ``'hh'`` for household level.
    M : int
        Lookback depth in calendar months (≥ 1).
    lambda_x : float
        Roughness penalty weight (≥ 0).  Ignored for ``'mean_spline'``.
    basis_type : str
        ``'cubic_spline'`` or ``'mean_spline'``.
    n_basis : int, optional
        Number of B-spline basis functions (``'cubic_spline'`` only).
        Defaults to ``max(6, M + 3)``.

    Returns
    -------
    coeff_matrix : np.ndarray of shape (n_borrowers, n_basis_funcs)
        Each row is the coefficient vector for one borrower.
        Rows for borrowers with insufficient data contain NaN.
    basis_info : dict
        Basis metadata including ``'type'``, ``'n_basis'``, ``'domain'``,
        and (for cubic_spline) ``'knots'``, ``'degree'``, ``'lambda_x'``.

    Raises
    ------
    ValueError
        If *level* or *basis_type* is unrecognised, or if M < 1.
    """
    if level not in ("bw", "hh"):
        raise ValueError(f"level must be 'bw' or 'hh', got '{level}'.")
    if basis_type not in ("cubic_spline", "mean_spline"):
        raise ValueError(
            f"basis_type must be 'cubic_spline' or 'mean_spline', got '{basis_type}'."
        )
    if M < 1:
        raise ValueError(f"M must be ≥ 1, got {M}.")
    if variable not in _DEFAULT_VARIABLES:
        raise ValueError(f"Unknown variable '{variable}'. Choose from {_DEFAULT_VARIABLES}.")

    # Select column and daily DataFrame
    if level == "bw":
        col = variable
        df_daily = borrower_daily_df
    else:
        col = _HH_COL[variable]
        df_daily = hh_daily_df

    daily_idx = (
        df_daily.copy()
        .assign(date=lambda d: pd.to_datetime(d["date"]).dt.normalize())
        .set_index(["borrower_id", "date"])
    )

    # ------------------------------------------------------------------ #
    #  mean_spline: piecewise-constant, one mean per calendar month       #
    # ------------------------------------------------------------------ #
    if basis_type == "mean_spline":
        coeff_rows: List[np.ndarray] = []
        for srow in sampling_df.itertuples(index=False):
            b_id = int(srow.borrower_id)
            T = pd.Timestamp(srow.sampling_date).normalize()
            month_means: List[float] = []

            for m in range(1, M + 1):
                months_before = M - m + 1
                t_year, t_month = _month_offset(T.year, T.month, months_before)
                m_bdays = get_banking_days_in_month(t_year, t_month, banking_days)
                if not m_bdays:
                    month_means.append(np.nan)
                    continue
                vals: List[float] = []
                for d in m_bdays:
                    try:
                        vals.append(float(daily_idx.at[(b_id, d.normalize()), col]))
                    except KeyError:
                        pass
                month_means.append(float(np.nanmean(vals)) if vals else np.nan)

            coeff_rows.append(np.array(month_means, dtype=np.float64))

        coeff_matrix = np.vstack(coeff_rows) if coeff_rows else np.empty((0, M))
        basis_info = {
            "type": "mean_spline",
            "n_basis": M,
            "domain": (0.0, float(M)),
            "M": M,
        }
        return coeff_matrix, basis_info

    # ------------------------------------------------------------------ #
    #  cubic_spline: penalised B-spline                                   #
    # ------------------------------------------------------------------ #
    if n_basis is None:
        n_basis = max(6, M + 3)

    domain_end = float(M)
    knots = _make_clamped_knots(domain_end, n_basis, _DEGREE)

    # Roughness penalty matrix (computed once for all borrowers)
    S: Optional[np.ndarray] = None
    if lambda_x > 0.0:
        S = compute_roughness_penalty_matrix(
            knots, degree=_DEGREE, domain_start=0.0, domain_end=domain_end
        )

    coeff_rows_cs: List[np.ndarray] = []
    for srow in sampling_df.itertuples(index=False):
        b_id = int(srow.borrower_id)
        T = pd.Timestamp(srow.sampling_date).normalize()

        tau_list: List[float] = []
        y_list: List[float] = []

        for m in range(1, M + 1):
            months_before = M - m + 1
            t_year, t_month = _month_offset(T.year, T.month, months_before)
            m_bdays = get_banking_days_in_month(t_year, t_month, banking_days)
            k_m = len(m_bdays)
            if k_m == 0:
                continue
            for j, d in enumerate(m_bdays, start=1):
                tau = j / k_m + (m - 1)
                try:
                    y_val = float(daily_idx.at[(b_id, d.normalize()), col])
                    tau_list.append(tau)
                    y_list.append(y_val)
                except KeyError:
                    pass

        if len(tau_list) < 2:
            coeff_rows_cs.append(np.full(n_basis, np.nan, dtype=np.float64))
            continue

        tau_arr = np.asarray(tau_list, dtype=np.float64)
        y_arr = np.asarray(y_list, dtype=np.float64)

        B = compute_spline_basis_matrix(tau_arr, domain_end, n_basis, degree=_DEGREE)

        if lambda_x == 0.0:
            c, _, _, _ = np.linalg.lstsq(B, y_arr, rcond=None)
        else:
            A = B.T @ B + lambda_x * S  # type: ignore[operator]
            rhs = B.T @ y_arr
            try:
                c = np.linalg.solve(A, rhs)
            except np.linalg.LinAlgError:
                c = np.linalg.lstsq(A, rhs, rcond=None)[0]

        coeff_rows_cs.append(c.astype(np.float64))

    if coeff_rows_cs:
        coeff_matrix = np.vstack(coeff_rows_cs)
    else:
        coeff_matrix = np.empty((0, n_basis), dtype=np.float64)

    basis_info = {
        "type": "cubic_spline",
        "knots": knots,
        "degree": _DEGREE,
        "n_basis": n_basis,
        "domain": (0.0, domain_end),
        "lambda_x": float(lambda_x),
    }
    return coeff_matrix, basis_info


# ---------------------------------------------------------------------------
# 3. B-spline design matrix
# ---------------------------------------------------------------------------

def compute_spline_basis_matrix(
    time_points: np.ndarray,
    domain_end: float,
    n_basis: int,
    degree: int = _DEGREE,
) -> np.ndarray:
    """Evaluate the B-spline design matrix at the given time points.

    Sets up a clamped cubic (or arbitrary-degree) B-spline basis on
    ``[0, domain_end]`` with ``n_basis - degree - 1`` equally spaced interior
    knots and evaluates each basis function at every element of *time_points*.

    Parameters
    ----------
    time_points : array-like of float
        Evaluation points in the domain (0, *domain_end*].
    domain_end : float
        Right boundary of the B-spline domain.
    n_basis : int
        Number of B-spline basis functions (``≥ degree + 1``).
    degree : int
        Polynomial degree (default 3 for cubic).

    Returns
    -------
    np.ndarray of shape ``(len(time_points), n_basis)`` in float64.

    Raises
    ------
    ValueError
        If ``n_basis < degree + 1`` or ``domain_end ≤ 0``.
    """
    if domain_end <= 0.0:
        raise ValueError(f"domain_end must be positive, got {domain_end}.")

    knots = _make_clamped_knots(float(domain_end), n_basis, degree)
    t = np.asarray(time_points, dtype=np.float64).ravel()

    B = np.zeros((len(t), n_basis), dtype=np.float64)
    for k in range(n_basis):
        c = np.zeros(n_basis, dtype=np.float64)
        c[k] = 1.0
        spl = BSpline(knots, c, degree, extrapolate=False)
        vals = spl(t)
        # Replace NaN (extrapolation outside knot range) with 0
        B[:, k] = np.where(np.isnan(vals), 0.0, vals)

    return B


# ---------------------------------------------------------------------------
# 4. Roughness penalty matrix
# ---------------------------------------------------------------------------

def compute_roughness_penalty_matrix(
    knots: np.ndarray,
    degree: int = _DEGREE,
    domain_start: float = 0.0,
    domain_end: Optional[float] = None,
    n_quad: int = 200,
) -> np.ndarray:
    """Compute the integrated squared-second-derivative roughness penalty matrix.

    Returns the symmetric matrix S where

        S_{kl} = ∫ ψ_k''(s) ψ_l''(s) ds

    integrated over ``[domain_start, domain_end]``, using Gauss–Legendre
    quadrature with *n_quad* nodes.

    Parameters
    ----------
    knots : array-like
        B-spline knot vector (e.g. from :func:`compute_spline_basis_matrix`
        or :func:`_make_clamped_knots`).
    degree : int
        Polynomial degree matching the knots (default 3).
    domain_start : float
        Left integration limit.
    domain_end : float, optional
        Right integration limit.  Defaults to the last knot value.
    n_quad : int
        Number of Gauss–Legendre quadrature nodes.

    Returns
    -------
    np.ndarray of shape ``(n_basis, n_basis)`` in float64, where
    ``n_basis = len(knots) - degree - 1``.
    """
    knots = np.asarray(knots, dtype=np.float64)
    if domain_end is None:
        domain_end = float(knots[-1])

    n_basis = len(knots) - degree - 1
    if n_basis <= 0:
        raise ValueError(
            f"Knot vector implies non-positive n_basis ({n_basis})."
            " Check knots and degree."
        )

    # Gauss–Legendre nodes and weights on [-1, 1], mapped to [domain_start, domain_end]
    xi, wi = np.polynomial.legendre.leggauss(n_quad)
    scale = 0.5 * (domain_end - domain_start)
    shift = 0.5 * (domain_end + domain_start)
    s = scale * xi + shift  # shape (n_quad,)
    w = scale * wi          # quadrature weights

    # Second-derivative values of each basis function at quadrature points
    B_dd = np.zeros((n_quad, n_basis), dtype=np.float64)
    for k in range(n_basis):
        c = np.zeros(n_basis, dtype=np.float64)
        c[k] = 1.0
        spl_dd = BSpline(knots, c, degree).derivative(2)
        B_dd[:, k] = spl_dd(s)

    # S = (B_dd.T diag(w)) @ B_dd  →  S_{kl} = Σ_q w_q B_dd_{qk} B_dd_{ql}
    S = (B_dd.T * w) @ B_dd
    return S.astype(np.float64)


# ---------------------------------------------------------------------------
# 5. Inner-product matrix between two bases
# ---------------------------------------------------------------------------

def compute_inner_product_matrix(
    phi_basis_params: dict,
    psi_basis_params: dict,
    domain_start: float = 0.0,
    domain_end: Optional[float] = None,
    n_quad: int = 200,
) -> np.ndarray:
    """Compute the inner-product matrix between two B-spline bases.

    Returns the matrix J of shape (L, K) where

        J_{lk} = ∫ φ_l(s) ψ_k(s) ds

    integrated over ``[domain_start, domain_end]``.

    This matrix is used in functional logistic regression to project the
    functional coefficient α(s) (expanded in the φ basis) against the
    functional predictor path x(s) (expanded in the ψ basis).

    Parameters
    ----------
    phi_basis_params : dict
        Basis parameters for the coefficient function φ.  Required keys:
        ``'knots'`` (array-like) and ``'degree'`` (int).
        L = len(knots) - degree - 1.
    psi_basis_params : dict
        Basis parameters for the predictor path ψ.  Same structure.
        K = len(knots) - degree - 1.
    domain_start : float
        Left integration limit (default 0).
    domain_end : float, optional
        Right integration limit.  Defaults to the maximum of the rightmost
        knots of φ and ψ.
    n_quad : int
        Number of Gauss–Legendre quadrature nodes.

    Returns
    -------
    np.ndarray of shape ``(L, K)`` in float64.

    Raises
    ------
    ValueError
        If required keys are missing from the parameter dictionaries.
    """
    for name, params in (("phi_basis_params", phi_basis_params),
                          ("psi_basis_params", psi_basis_params)):
        if "knots" not in params or "degree" not in params:
            raise ValueError(
                f"'{name}' must contain keys 'knots' and 'degree'."
            )

    phi_knots = np.asarray(phi_basis_params["knots"], dtype=np.float64)
    phi_deg: int = int(phi_basis_params["degree"])
    psi_knots = np.asarray(psi_basis_params["knots"], dtype=np.float64)
    psi_deg: int = int(psi_basis_params["degree"])

    L = len(phi_knots) - phi_deg - 1
    K = len(psi_knots) - psi_deg - 1

    if L <= 0 or K <= 0:
        raise ValueError(
            f"Invalid basis dimensions: L={L}, K={K}.  "
            "Check knot vectors and degrees."
        )

    if domain_end is None:
        domain_end = float(max(phi_knots[-1], psi_knots[-1]))

    # Gauss–Legendre quadrature on [domain_start, domain_end]
    xi, wi = np.polynomial.legendre.leggauss(n_quad)
    scale = 0.5 * (domain_end - domain_start)
    shift = 0.5 * (domain_end + domain_start)
    s = scale * xi + shift  # (n_quad,)
    w = scale * wi

    # Evaluate φ basis at quadrature points → shape (n_quad, L)
    Phi = np.zeros((n_quad, L), dtype=np.float64)
    for ll in range(L):
        c = np.zeros(L, dtype=np.float64)
        c[ll] = 1.0
        spl = BSpline(phi_knots, c, phi_deg, extrapolate=False)
        vals = spl(s)
        Phi[:, ll] = np.where(np.isnan(vals), 0.0, vals)

    # Evaluate ψ basis at quadrature points → shape (n_quad, K)
    Psi = np.zeros((n_quad, K), dtype=np.float64)
    for k in range(K):
        c = np.zeros(K, dtype=np.float64)
        c[k] = 1.0
        spl = BSpline(psi_knots, c, psi_deg, extrapolate=False)
        vals = spl(s)
        Psi[:, k] = np.where(np.isnan(vals), 0.0, vals)

    # J_{lk} = Σ_q w_q Phi_{ql} Psi_{qk}  →  J = Phiᵀ diag(w) Psi
    J = (Phi.T * w) @ Psi  # shape (L, K)
    return J.astype(np.float64)

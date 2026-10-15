"""Functional logistic regression PD model with REML smoothing and greedy forward selection.

Each functional predictor X_i(s) is represented in a spline basis with coefficient
vector c_i of length K.  The coefficient function beta(s) is expanded in L=5 cubic
B-spline basis functions with coefficient vector b.  The linear predictor for
observation i is:

    eta_i = alpha + sum_{terms} (J c_i)^T b_term

where J (L x K) is the inner-product matrix between the coefficient basis and the
predictor basis.  Smoothing parameters are estimated via the REML (Laplace
approximation) criterion from Wood (2011).
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.linalg
import scipy.optimize
from scipy.interpolate import BSpline
from scipy.special import expit
from sklearn.metrics import roc_auc_score


class FunctionalLogisticModel:
    """Functional logistic regression with REML smoothing and greedy forward selection.

    The model selects functional predictors greedily by maximising validation
    ROC-AUC, then estimates a single global smoothing parameter lambda via the
    REML (Laplace-approximation marginal-likelihood) criterion.

    Attributes
    ----------
    alpha_ : float
        Fitted intercept of the final model.
    term_coefficients_ : dict[str, np.ndarray]
        Maps each selected term name to its L-dimensional coefficient vector b.
    selected_terms_ : list[str]
        Ordered list of term names chosen during greedy forward selection.
    lambda_beta_ : float
        Global smoothing parameter estimated by REML.
    lambda_betas_per_term_ : dict[str, float]
        Per-term smoothing parameters (all equal to ``lambda_beta_`` for a
        single-lambda model).
    """

    # B-spline degree used for the coefficient function basis
    _BSPLINE_DEGREE: int = 3

    def __init__(self) -> None:
        self.alpha_: float = 0.0
        self.term_coefficients_: Dict[str, np.ndarray] = {}
        self.selected_terms_: List[str] = []
        self.lambda_beta_: float = 0.01
        self.lambda_betas_per_term_: Dict[str, float] = {}
        self._selection_history: List[dict] = []
        # Stored during fit for coefficient function reconstruction
        self._term_basis_info_: Dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Private utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _make_bspline_knots(domain_end: float, L: int, degree: int = 3) -> np.ndarray:
        """Build a clamped B-spline knot vector for L basis functions.

        Parameters
        ----------
        domain_end : float
            Upper bound of the domain [0, domain_end].
        L : int
            Number of B-spline basis functions.
        degree : int, default 3
            Polynomial degree (3 = cubic).

        Returns
        -------
        np.ndarray
            Knot vector of length L + degree + 1.
        """
        n_internal = L - degree - 1
        if n_internal < 0:
            raise ValueError(
                f"Cannot create {L} B-spline basis functions of degree {degree}: "
                "need L >= degree + 1."
            )
        internal = np.linspace(0.0, domain_end, n_internal + 2)[1:-1]
        return np.concatenate(
            [
                np.repeat(0.0, degree + 1),
                internal,
                np.repeat(domain_end, degree + 1),
            ]
        )

    @staticmethod
    def _eval_bspline_basis(
        t: np.ndarray, knots: np.ndarray, L: int, degree: int = 3
    ) -> np.ndarray:
        """Evaluate all L B-spline basis functions at positions t.

        Parameters
        ----------
        t : np.ndarray of shape (n_points,)
            Evaluation points.
        knots : np.ndarray
            Knot vector of length L + degree + 1.
        L : int
            Number of basis functions.
        degree : int, default 3
            Polynomial degree.

        Returns
        -------
        np.ndarray of shape (n_points, L)
            Basis matrix Phi with Phi[i, j] = phi_j(t[i]).
        """
        n_pts = len(t)
        Phi = np.zeros((n_pts, L), dtype=np.float64)
        for j in range(L):
            c = np.zeros(L, dtype=np.float64)
            c[j] = 1.0
            spl = BSpline(knots, c, degree, extrapolate=True)
            Phi[:, j] = spl(t)
        return Phi

    # ------------------------------------------------------------------
    # Core building blocks
    # ------------------------------------------------------------------

    def _build_design_matrix(
        self,
        functional_data_list: List[Dict],
    ) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
        """Build the combined design matrix and block-diagonal penalty.

        Parameters
        ----------
        functional_data_list : list[dict]
            Each dict must contain:
            - ``'J'`` : np.ndarray of shape (L, K) — inner-product matrix.
            - ``'C'`` : np.ndarray of shape (n_obs, K) — predictor basis coeff.
            - ``'S'`` : np.ndarray of shape (L, L) — roughness penalty matrix.

        Returns
        -------
        tilde_X : np.ndarray of shape (n_obs, total_params)
            Combined design matrix.  The column block for term ``i`` is
            ``C_i @ J_i.T``.
        S_block : np.ndarray of shape (total_params, total_params)
            Block-diagonal penalty matrix.
        term_slices : list[tuple[int, int]]
            ``(start, end)`` column indices in ``tilde_X`` for each term.
        """
        if not functional_data_list:
            raise ValueError("functional_data_list must not be empty.")

        tilde_X_blocks: List[np.ndarray] = []
        S_diag_blocks: List[np.ndarray] = []
        term_slices: List[Tuple[int, int]] = []
        cursor = 0

        for term in functional_data_list:
            J: np.ndarray = np.asarray(term["J"], dtype=np.float64)  # (L, K)
            C: np.ndarray = np.asarray(term["C"], dtype=np.float64)  # (n_obs, K)
            S: np.ndarray = np.asarray(term["S"], dtype=np.float64)  # (L, L)

            L = J.shape[0]
            tilde_block = C @ J.T  # (n_obs, L)
            tilde_X_blocks.append(tilde_block)
            S_diag_blocks.append(S)
            term_slices.append((cursor, cursor + L))
            cursor += L

        tilde_X = np.hstack(tilde_X_blocks).astype(np.float64)
        total = cursor
        S_block = np.zeros((total, total), dtype=np.float64)
        for blk, (s, e) in zip(S_diag_blocks, term_slices):
            S_block[s:e, s:e] = blk

        return tilde_X, S_block, term_slices

    def _fit_penalized_logistic_irls(
        self,
        tilde_X: np.ndarray,
        y: np.ndarray,
        S_block: np.ndarray,
        lambda_beta: float,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> Tuple[float, np.ndarray, bool]:
        """Fit penalised logistic regression via IRLS (Fisher scoring).

        Solves the penalised maximum-likelihood problem:

            argmax_{alpha, b} ell(alpha, b) - 0.5 * lambda * b^T S b

        using iteratively reweighted least squares.  The intercept alpha is
        not penalised.

        Parameters
        ----------
        tilde_X : np.ndarray of shape (n_obs, n_params)
            Design matrix (without intercept column).
        y : np.ndarray of shape (n_obs,)
            Binary response (0/1).
        S_block : np.ndarray of shape (n_params, n_params)
            Roughness penalty matrix for the beta coefficients.
        lambda_beta : float
            Smoothing parameter (>= 0).
        max_iter : int, default 100
            Maximum IRLS iterations.
        tol : float, default 1e-6
            Relative convergence tolerance on the parameter vector.

        Returns
        -------
        alpha : float
            Fitted intercept.
        b_vec : np.ndarray of shape (n_params,)
            Fitted beta-basis coefficient vector.
        converged : bool
            Whether IRLS converged within ``max_iter`` iterations.
        """
        n_obs, n_params = tilde_X.shape
        _EPS = 1e-10

        # Augmented design matrix: [1 | tilde_X]
        X_aug = np.column_stack([np.ones(n_obs, dtype=np.float64), tilde_X])
        n_aug = X_aug.shape[1]

        # Augmented penalty: zero for intercept, lambda*S for beta params
        S_aug = np.zeros((n_aug, n_aug), dtype=np.float64)
        S_aug[1:, 1:] = lambda_beta * S_block

        # Initialise at zeros (log-odds = 0 => pi = 0.5)
        theta_aug = np.zeros(n_aug, dtype=np.float64)

        converged = False
        for _it in range(max_iter):
            eta = X_aug @ theta_aug
            pi = np.clip(expit(eta), _EPS, 1.0 - _EPS)
            w = pi * (1.0 - pi)

            z = eta + (y - pi) / w  # adjusted dependent variable

            # Build weighted normal equations
            XtW = X_aug.T * w[np.newaxis, :]  # (n_aug, n_obs)
            A = XtW @ X_aug + S_aug
            rhs = XtW @ z

            try:
                c, low = scipy.linalg.cho_factor(A, check_finite=False)
                theta_new = scipy.linalg.cho_solve((c, low), rhs, check_finite=False)
            except (scipy.linalg.LinAlgError, np.linalg.LinAlgError):
                try:
                    theta_new = np.linalg.solve(A, rhs)
                except np.linalg.LinAlgError:
                    warnings.warn(
                        "IRLS: singular system encountered; returning current estimate.",
                        RuntimeWarning,
                        stacklevel=3,
                    )
                    return float(theta_aug[0]), theta_aug[1:].astype(np.float64), False

            delta = np.linalg.norm(theta_new - theta_aug) / (
                1.0 + np.linalg.norm(theta_aug)
            )
            theta_aug = theta_new
            if delta < tol:
                converged = True
                break

        if not converged:
            warnings.warn(
                f"IRLS did not converge after {max_iter} iterations "
                f"(lambda={lambda_beta:.4g}).",
                RuntimeWarning,
                stacklevel=3,
            )

        return float(theta_aug[0]), theta_aug[1:].astype(np.float64), converged

    def _reml_criterion(
        self,
        log_lambda: float,
        tilde_X: np.ndarray,
        y: np.ndarray,
        S_block: np.ndarray,
    ) -> float:
        """Laplace-approximation REML criterion for smoothing parameter selection.

        Computes the REML objective (to be minimised):

            REML = -ell_pen(theta_hat) + 0.5 * log|H|
                   - 0.5 * p * log(lambda) - 0.5 * log|S_+|

        where H = X_aug^T W X_aug + lambda * S_aug is the Hessian of the
        negative penalised log-likelihood at the MAP estimate, p is the rank
        of ``S_block``, and S_+ denotes the restriction of S to its nonzero
        eigenvalue subspace.

        Parameters
        ----------
        log_lambda : float
            Natural logarithm of the smoothing parameter to evaluate.
        tilde_X : np.ndarray of shape (n_obs, n_params)
            Design matrix (without intercept).
        y : np.ndarray of shape (n_obs,)
            Binary response.
        S_block : np.ndarray of shape (n_params, n_params)
            Roughness penalty matrix.

        Returns
        -------
        float
            REML criterion value (lower is better).  Returns ``np.inf`` for
            degenerate configurations.
        """
        _EPS_CLIP = 1e-15
        _EPS_EIG = 1e-10

        lambda_val = np.exp(log_lambda)
        n_obs, n_params = tilde_X.shape

        alpha, b_vec, converged = self._fit_penalized_logistic_irls(
            tilde_X, y, S_block, lambda_val
        )
        if not converged:
            return np.inf

        # Augmented system
        X_aug = np.column_stack([np.ones(n_obs, dtype=np.float64), tilde_X])
        theta_aug = np.concatenate([[alpha], b_vec])

        # Penalised log-likelihood
        eta = X_aug @ theta_aug
        pi = np.clip(expit(eta), _EPS_CLIP, 1.0 - _EPS_CLIP)
        log_lik = float(
            np.sum(y * np.log(pi) + (1.0 - y) * np.log(1.0 - pi))
        )
        penalty = 0.5 * lambda_val * float(b_vec @ S_block @ b_vec)
        ell_pen = log_lik - penalty

        # Hessian H = X_aug^T W X_aug + lambda * S_aug
        w = np.clip(pi * (1.0 - pi), _EPS_CLIP, None)
        S_aug = np.zeros((n_params + 1, n_params + 1), dtype=np.float64)
        S_aug[1:, 1:] = lambda_val * S_block
        H = (X_aug.T * w[np.newaxis, :]) @ X_aug + S_aug

        sign, logdet_H = np.linalg.slogdet(H)
        if sign <= 0:
            return np.inf

        # Rank and log-determinant of S_block (nonzero eigenvalue subspace)
        eigs = np.linalg.eigvalsh(S_block)
        max_eig = float(np.max(np.abs(eigs))) if eigs.size > 0 else 0.0
        if max_eig < _EPS_CLIP:
            return np.inf
        nz_eigs = eigs[eigs > _EPS_EIG * max_eig]
        p = len(nz_eigs)
        if p == 0:
            return np.inf
        logdet_S_nz = float(np.sum(np.log(np.maximum(nz_eigs, 1e-300))))

        reml = (
            -ell_pen
            + 0.5 * float(logdet_H)
            - 0.5 * p * np.log(lambda_val)
            - 0.5 * logdet_S_nz
        )
        return float(reml)

    def _estimate_lambda_reml(
        self,
        tilde_X: np.ndarray,
        y: np.ndarray,
        S_block: np.ndarray,
        lambda_init: float = 0.01,
        n_grid: int = 15,
    ) -> float:
        """Estimate the global smoothing parameter by minimising the REML criterion.

        Performs a coarse log-spaced grid search over [1e-6, 1e3] followed by
        scalar refinement with ``scipy.optimize.minimize_scalar``.

        Parameters
        ----------
        tilde_X : np.ndarray of shape (n_obs, n_params)
            Design matrix without intercept.
        y : np.ndarray of shape (n_obs,)
            Binary response.
        S_block : np.ndarray of shape (n_params, n_params)
            Roughness penalty matrix.
        lambda_init : float, default 0.01
            Initial guess (not used directly; grid always searched first).
        n_grid : int, default 15
            Number of grid points in the log-spaced search.

        Returns
        -------
        float
            Optimal smoothing parameter lambda.
        """
        log_lam_grid = np.linspace(np.log(1e-6), np.log(1e3), n_grid)
        reml_vals = np.array(
            [self._reml_criterion(ll, tilde_X, y, S_block) for ll in log_lam_grid],
            dtype=np.float64,
        )

        finite_mask = np.isfinite(reml_vals)
        if not np.any(finite_mask):
            warnings.warn(
                "REML criterion was non-finite for all grid points; "
                "returning lambda_init.",
                RuntimeWarning,
                stacklevel=2,
            )
            return float(lambda_init)

        best_log_lam_grid = float(log_lam_grid[np.argmin(reml_vals)])

        try:
            result = scipy.optimize.minimize_scalar(
                lambda ll: self._reml_criterion(ll, tilde_X, y, S_block),
                bounds=(np.log(1e-6), np.log(1e3)),
                method="bounded",
                options={"xatol": 1e-4, "maxiter": 100},
            )
            if result.success and np.isfinite(result.fun):
                return float(np.exp(result.x))
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"REML minimisation failed ({exc}); using grid optimum.",
                RuntimeWarning,
                stacklevel=2,
            )

        return float(np.exp(best_log_lam_grid))

    # ------------------------------------------------------------------
    # Single-term fitting
    # ------------------------------------------------------------------

    def fit_single_term(
        self,
        C_matrix: np.ndarray,
        J_matrix: np.ndarray,
        S_coeff: np.ndarray,
        y: np.ndarray,
        L: int = 5,
    ) -> dict:
        """Fit a single-term functional logistic regression with REML smoothing.

        Parameters
        ----------
        C_matrix : np.ndarray of shape (n_obs, K)
            Predictor spline basis coefficients for each observation.
        J_matrix : np.ndarray of shape (L, K)
            Inner-product matrix between the coefficient basis and the
            predictor basis.
        S_coeff : np.ndarray of shape (L, L)
            Roughness penalty matrix for the coefficient function basis.
        y : np.ndarray of shape (n_obs,)
            Binary response (0/1).
        L : int, default 5
            Number of coefficient basis functions (must match ``J_matrix.shape[0]``).

        Returns
        -------
        dict
            - ``'alpha'`` (float): fitted intercept.
            - ``'b_coeff'`` (np.ndarray of shape (L,)): coefficient vector.
            - ``'lambda_beta'`` (float): REML-estimated smoothing parameter.
            - ``'converged'`` (bool): whether IRLS converged.
        """
        C_matrix = np.asarray(C_matrix, dtype=np.float64)
        J_matrix = np.asarray(J_matrix, dtype=np.float64)
        S_coeff = np.asarray(S_coeff, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        tilde_X = C_matrix @ J_matrix.T  # (n_obs, L)
        lambda_beta = self._estimate_lambda_reml(tilde_X, y, S_coeff)
        alpha, b_coeff, converged = self._fit_penalized_logistic_irls(
            tilde_X, y, S_coeff, lambda_beta
        )
        return {
            "alpha": alpha,
            "b_coeff": b_coeff,
            "lambda_beta": lambda_beta,
            "converged": converged,
        }

    # ------------------------------------------------------------------
    # Greedy forward selection and full fit
    # ------------------------------------------------------------------

    def _eval_candidate_val_auc(
        self,
        selected_dicts: List[dict],
        candidate: dict,
        y_train: np.ndarray,
        y_val: np.ndarray,
    ) -> float:
        """Fit combined model (selected + candidate) and return validation AUC.

        Uses a lightweight REML lambda estimation (coarser grid) during
        forward selection for computational efficiency.

        Parameters
        ----------
        selected_dicts : list[dict]
            Currently selected terms (dicts with 'J', 'C_train', 'C_val', 'S_coeff').
        candidate : dict
            Candidate term to evaluate.
        y_train : np.ndarray
            Binary training labels.
        y_val : np.ndarray
            Binary validation labels.

        Returns
        -------
        float
            Validation ROC-AUC, or 0.5 on failure.
        """
        trial_terms = selected_dicts + [candidate]

        train_data = [
            {"J": t["J"], "C": t["C_train"], "S": t["S_coeff"]}
            for t in trial_terms
        ]
        val_data = [
            {"J": t["J"], "C": t["C_val"], "S": t["S_coeff"]}
            for t in trial_terms
        ]

        try:
            tilde_X_tr, S_block, _ = self._build_design_matrix(train_data)
            tilde_X_vl, _, _ = self._build_design_matrix(val_data)

            # Coarser REML grid during selection for speed
            lambda_beta = self._estimate_lambda_reml(
                tilde_X_tr, y_train, S_block, n_grid=8
            )
            alpha, b_vec, _ = self._fit_penalized_logistic_irls(
                tilde_X_tr, y_train, S_block, lambda_beta
            )

            eta_val = alpha + tilde_X_vl @ b_vec
            pi_val = expit(eta_val)
            return float(roc_auc_score(y_val, pi_val))
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"Error evaluating candidate '{candidate.get('name', '?')}': {exc}",
                RuntimeWarning,
                stacklevel=3,
            )
            return 0.5

    def fit(
        self,
        terms_list: List[dict],
        y_train: np.ndarray,
        y_val: np.ndarray,
        terms_val_list: Optional[List[dict]] = None,
        auc_threshold: float = 0.0005,
        L: int = 5,
    ) -> "FunctionalLogisticModel":
        """Greedy forward selection of functional terms followed by final REML fit.

        Parameters
        ----------
        terms_list : list[dict]
            Candidate functional terms.  Each dict must contain:
            - ``'name'`` (str): unique term identifier.
            - ``'C_train'`` (np.ndarray of shape (n_train, K)): training basis
              coefficients.
            - ``'C_val'`` (np.ndarray of shape (n_val, K)): validation basis
              coefficients.
            - ``'J'`` (np.ndarray of shape (L, K)): inner-product matrix.
            - ``'S_pred'`` (np.ndarray of shape (K, K)): predictor-basis penalty
              (not used in model fitting, retained for reference).
            - ``'S_coeff'`` (np.ndarray of shape (L, L)): coefficient-basis
              roughness penalty.
            - ``'domain_end'`` (float): upper bound of the functional domain.
        y_train : np.ndarray of shape (n_train,)
            Binary training labels.
        y_val : np.ndarray of shape (n_val,)
            Binary validation labels.
        terms_val_list : list[dict] or None, default None
            Optional separate validation term list.  If *None*, ``'C_val'``
            from ``terms_list`` is used.
        auc_threshold : float, default 0.0005
            Minimum validation AUC improvement to add a term.
        L : int, default 5
            Number of coefficient basis functions (used for basis reconstruction).

        Returns
        -------
        self : FunctionalLogisticModel
            Fitted model instance.

        Raises
        ------
        ValueError
            If ``terms_list`` is empty or contains duplicate names.
        """
        y_train = np.asarray(y_train, dtype=np.float64)
        y_val = np.asarray(y_val, dtype=np.float64)

        if not terms_list:
            raise ValueError("terms_list must not be empty.")

        names = [t["name"] for t in terms_list]
        if len(names) != len(set(names)):
            raise ValueError("terms_list contains duplicate term names.")

        # Build a lookup for val data (use C_val from terms_list unless overridden)
        val_lookup: Dict[str, np.ndarray] = {}
        if terms_val_list is not None:
            for tv in terms_val_list:
                val_lookup[tv["name"]] = np.asarray(tv["C"], dtype=np.float64)

        # Enrich each term dict with arrays
        term_dicts = []
        for t in terms_list:
            td = {
                "name": t["name"],
                "C_train": np.asarray(t["C_train"], dtype=np.float64),
                "C_val": (
                    val_lookup[t["name"]]
                    if t["name"] in val_lookup
                    else np.asarray(t["C_val"], dtype=np.float64)
                ),
                "J": np.asarray(t["J"], dtype=np.float64),
                "S_coeff": np.asarray(t["S_coeff"], dtype=np.float64),
                "domain_end": float(t["domain_end"]),
            }
            term_dicts.append(td)

        # ------ Greedy forward selection ------
        remaining = list(term_dicts)
        selected: List[dict] = []
        current_auc: float = 0.0
        selection_history: List[dict] = []

        while remaining:
            best_auc = current_auc
            best_term: Optional[dict] = None

            for candidate in remaining:
                auc = self._eval_candidate_val_auc(
                    selected, candidate, y_train, y_val
                )
                if auc > best_auc:
                    best_auc = auc
                    best_term = candidate

            delta = best_auc - current_auc
            if best_term is None or delta <= auc_threshold:
                break

            selected.append(best_term)
            remaining.remove(best_term)
            selection_history.append(
                {
                    "term": best_term["name"],
                    "val_auc": best_auc,
                    "delta_auc": delta,
                }
            )
            current_auc = best_auc

        if not selected:
            warnings.warn(
                "No term improved validation AUC beyond the threshold. "
                "The model has no selected terms.",
                UserWarning,
                stacklevel=2,
            )

        self._selection_history = selection_history
        self.selected_terms_ = [t["name"] for t in selected]

        # ------ Final REML fit on training data with selected terms ------
        if selected:
            train_data_final = [
                {"J": t["J"], "C": t["C_train"], "S": t["S_coeff"]}
                for t in selected
            ]
            tilde_X_final, S_block_final, term_slices_final = (
                self._build_design_matrix(train_data_final)
            )
            self.lambda_beta_ = self._estimate_lambda_reml(
                tilde_X_final, y_train, S_block_final
            )
            alpha_final, b_vec_final, _ = self._fit_penalized_logistic_irls(
                tilde_X_final, y_train, S_block_final, self.lambda_beta_
            )
            self.alpha_ = alpha_final

            self.term_coefficients_ = {}
            self.lambda_betas_per_term_ = {}
            self._term_basis_info_ = {}
            for term, (s, e) in zip(selected, term_slices_final):
                name = term["name"]
                self.term_coefficients_[name] = b_vec_final[s:e].copy()
                self.lambda_betas_per_term_[name] = self.lambda_beta_
                self._term_basis_info_[name] = {
                    "domain_end": term["domain_end"],
                    "L": L,
                    "degree": self._BSPLINE_DEGREE,
                }
        else:
            self.alpha_ = 0.0
            self.term_coefficients_ = {}
            self.lambda_betas_per_term_ = {}
            self._term_basis_info_ = {}

        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_proba(
        self, terms_data_list: List[Dict]
    ) -> np.ndarray:
        """Compute predicted default probabilities for new observations.

        Parameters
        ----------
        terms_data_list : list[dict]
            One dict per term.  Only terms whose name appears in
            ``selected_terms_`` are used.  Each dict must contain:
            - ``'name'`` (str): term identifier.
            - ``'C'`` (np.ndarray of shape (n_obs, K)): predictor basis coeff.
            - ``'J'`` (np.ndarray of shape (L, K)): inner-product matrix.

        Returns
        -------
        np.ndarray of shape (n_obs,)
            Predicted probabilities in [0, 1].

        Raises
        ------
        RuntimeError
            If the model has not been fitted (no selected terms and alpha is 0).
        ValueError
            If a selected term is missing from ``terms_data_list``.
        """
        if not self.selected_terms_ and not self.term_coefficients_:
            warnings.warn(
                "No terms were selected; returning constant probability expit(alpha).",
                UserWarning,
                stacklevel=2,
            )
            return np.array([], dtype=np.float64)

        # Index provided data by name
        data_by_name: Dict[str, dict] = {t["name"]: t for t in terms_data_list}
        missing = set(self.selected_terms_) - set(data_by_name)
        if missing:
            raise ValueError(
                f"terms_data_list is missing the following selected terms: {missing}"
            )

        # Determine n_obs from the first selected term
        first = data_by_name[self.selected_terms_[0]]
        n_obs = np.asarray(first["C"], dtype=np.float64).shape[0]

        eta = np.full(n_obs, self.alpha_, dtype=np.float64)
        for name in self.selected_terms_:
            td = data_by_name[name]
            C = np.asarray(td["C"], dtype=np.float64)  # (n_obs, K)
            J = np.asarray(td["J"], dtype=np.float64)  # (L, K)
            b = self.term_coefficients_[name]  # (L,)
            tilde_X_term = C @ J.T  # (n_obs, L)
            eta += tilde_X_term @ b

        return expit(eta).astype(np.float64)

    # ------------------------------------------------------------------
    # Coefficient function evaluation
    # ------------------------------------------------------------------

    def get_coefficient_functions(
        self,
        grid_points_per_term: Union[int, Dict[str, int]],
        coeff_basis_params_per_term: Optional[Dict[str, dict]] = None,
    ) -> Dict[str, dict]:
        """Evaluate each selected term's coefficient function on a dense grid.

        For term ``t``, the coefficient function is:

            beta_t(s) = sum_{l=1}^{L} b_{t,l} phi_l(s)

        where phi_l are the cubic B-spline basis functions and b_t is stored
        in ``term_coefficients_``.

        Parameters
        ----------
        grid_points_per_term : int or dict[str, int]
            Number of grid points per term.  If an int, the same number is
            used for all selected terms.
        coeff_basis_params_per_term : dict[str, dict] or None, default None
            Optional per-term basis parameters overriding the stored values.
            Each inner dict may contain ``'domain_end'`` (float), ``'L'`` (int),
            and ``'degree'`` (int).  If *None*, the parameters stored during
            ``fit`` are used.

        Returns
        -------
        dict[str, dict]
            Maps each selected term name to a dict with:
            - ``'t_grid'`` (np.ndarray): evaluation points.
            - ``'beta'`` (np.ndarray): coefficient function values.

        Raises
        ------
        RuntimeError
            If called before ``fit``.
        KeyError
            If basis parameters for a selected term cannot be found.
        """
        if not self.selected_terms_:
            return {}

        result: Dict[str, dict] = {}
        params_override = coeff_basis_params_per_term or {}

        for name in self.selected_terms_:
            # Resolve basis parameters
            stored = self._term_basis_info_.get(name, {})
            override = params_override.get(name, {})
            combined = {**stored, **override}

            domain_end = float(combined.get("domain_end", 1.0))
            L = int(combined.get("L", 5))
            degree = int(combined.get("degree", self._BSPLINE_DEGREE))

            n_grid = (
                grid_points_per_term
                if isinstance(grid_points_per_term, int)
                else grid_points_per_term[name]
            )
            t_grid = np.linspace(0.0, domain_end, int(n_grid))

            knots = self._make_bspline_knots(domain_end, L, degree)
            Phi = self._eval_bspline_basis(t_grid, knots, L, degree)

            b = self.term_coefficients_[name]
            beta = Phi @ b

            result[name] = {
                "t_grid": t_grid,
                "beta": beta.astype(np.float64),
            }

        return result

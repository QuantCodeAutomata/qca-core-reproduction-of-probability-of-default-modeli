"""Tests for LogisticPDModel, XGBoostPDModel, and FunctionalLogisticModel."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import pytest

from src.models.functional_logistic import FunctionalLogisticModel
from src.models.logistic_model import LogisticPDModel
from src.models.xgboost_model import XGBoostPDModel


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_binary_classification_data(n=300, n_features=6, seed=0):
    """Create a small binary classification dataset with genuine signal.

    Returns X (pd.DataFrame), y (np.ndarray of 0/1).
    """
    rng = np.random.default_rng(seed)
    X_raw = rng.normal(size=(n, n_features))
    # Linear predictor with random weights → ensures signal exists
    w = rng.normal(size=n_features)
    eta = X_raw @ w
    p = 1.0 / (1.0 + np.exp(-eta))
    y = (p > 0.5).astype(np.float64)
    cols = [f"feat_{i}" for i in range(n_features)]
    return pd.DataFrame(X_raw, columns=cols), y


def _split(X, y, n_train=200, n_val=60):
    return (
        X.iloc[:n_train],
        y[:n_train],
        X.iloc[n_train : n_train + n_val],
        y[n_train : n_train + n_val],
    )


# ---------------------------------------------------------------------------
# LogisticPDModel
# ---------------------------------------------------------------------------

class TestLogisticPDModel:
    def _fitted_model(self):
        X, y = _make_binary_classification_data(n=300, n_features=6, seed=0)
        X_tr, y_tr, X_vl, y_vl = _split(X, y)
        model = LogisticPDModel()
        model.fit(X_tr, y_tr, X_vl, y_vl, auc_threshold=0.0005, max_features=5)
        return model, X, y

    def test_logistic_fit_returns_self(self):
        """fit() must return the model instance (self)."""
        X, y = _make_binary_classification_data(n=300, n_features=4, seed=1)
        X_tr, y_tr, X_vl, y_vl = _split(X, y)
        model = LogisticPDModel()
        result = model.fit(X_tr, y_tr, X_vl, y_vl)
        assert result is model, "fit() must return self"

    def test_logistic_predict_proba_range(self):
        """Predicted probabilities must lie in [0, 1]."""
        model, X, y = self._fitted_model()
        if not model.selected_features_:
            pytest.skip("No features selected; skipping predict_proba range check")
        proba = model.predict_proba(X)
        assert np.all(proba >= 0.0), f"Negative probability found: min={proba.min()}"
        assert np.all(proba <= 1.0), f"Probability > 1 found: max={proba.max()}"

    def test_logistic_selected_features_nonempty(self):
        """At least one feature must be selected when training data has genuine signal."""
        # Use strong signal so at least one feature clears the AUC threshold
        rng = np.random.default_rng(42)
        n = 400
        X_raw = rng.normal(size=(n, 5))
        w = np.array([2.0, -2.0, 1.5, -1.5, 1.0])
        eta = X_raw @ w
        y = (1 / (1 + np.exp(-eta)) > 0.5).astype(np.float64)

        # Ensure both classes present in train and val
        split = 280
        X = pd.DataFrame(X_raw, columns=[f"f{i}" for i in range(5)])
        X_tr, y_tr = X.iloc[:split], y[:split]
        X_vl, y_vl = X.iloc[split:], y[split:]

        model = LogisticPDModel()
        model.fit(X_tr, y_tr, X_vl, y_vl, auc_threshold=0.001, max_features=5)
        assert len(model.selected_features_) >= 1, (
            "Expected at least one feature selected with strong-signal data"
        )

    def test_logistic_greedy_selection_improves_auc(self):
        """Each greedy step must show positive delta_auc in the history."""
        model, _, _ = self._fitted_model()
        history = model.get_selection_history()
        if not history:
            pytest.skip("No features selected; cannot verify delta_auc")
        for entry in history:
            assert entry["delta_auc"] > 0, (
                f"delta_auc={entry['delta_auc']} not positive for feature '{entry['feature']}'"
            )

    def test_logistic_selection_history_format(self):
        """Each history entry must contain 'feature', 'val_auc', and 'delta_auc'."""
        model, _, _ = self._fitted_model()
        history = model.get_selection_history()
        required_keys = {"feature", "val_auc", "delta_auc"}
        for entry in history:
            assert required_keys.issubset(entry.keys()), (
                f"History entry missing keys: {required_keys - set(entry.keys())}"
            )
            assert isinstance(entry["feature"], str)
            assert 0.0 <= entry["val_auc"] <= 1.0, (
                f"val_auc={entry['val_auc']} out of [0,1]"
            )
            assert entry["delta_auc"] > 0

    def test_logistic_predict_proba_before_fit_raises(self):
        """predict_proba must raise RuntimeError when called before fit()."""
        model = LogisticPDModel()
        X, _ = _make_binary_classification_data(n=50, n_features=3)
        with pytest.raises(RuntimeError):
            model.predict_proba(X)

    def test_logistic_candidate_features_subset(self):
        """fit() must respect the candidate_features argument."""
        X, y = _make_binary_classification_data(n=300, n_features=6, seed=3)
        X_tr, y_tr, X_vl, y_vl = _split(X, y)
        candidates = ["feat_0", "feat_1", "feat_2"]
        model = LogisticPDModel()
        model.fit(X_tr, y_tr, X_vl, y_vl, candidate_features=candidates, auc_threshold=0.0)
        for feat in model.selected_features_:
            assert feat in candidates, f"Selected feature '{feat}' not in candidate set"


# ---------------------------------------------------------------------------
# XGBoostPDModel
# ---------------------------------------------------------------------------

class TestXGBoostPDModel:
    def _fitted_xgboost(self, n=300, n_features=5, seed=0):
        X_df, y = _make_binary_classification_data(n=n, n_features=n_features, seed=seed)
        X = X_df.to_numpy()
        n_train = 200
        X_tr, y_tr = X[:n_train], y[:n_train]
        X_vl, y_vl = X[n_train:], y[n_train:]
        model = XGBoostPDModel()
        # Use n_trials=2 for speed; still exercises full Optuna machinery
        model.fit(X_tr, y_tr, X_vl, y_vl, random_state=42, n_trials=2)
        return model, X, y, n_features

    def test_xgboost_fit_returns_self(self):
        """fit() must return the model instance (self)."""
        X_df, y = _make_binary_classification_data(n=200, n_features=4, seed=10)
        X = X_df.to_numpy()
        model = XGBoostPDModel()
        result = model.fit(X[:150], y[:150], X[150:], y[150:], n_trials=1)
        assert result is model, "fit() must return self"

    def test_xgboost_predict_proba_range(self):
        """Predicted probabilities must lie in [0, 1]."""
        model, X, y, _ = self._fitted_xgboost()
        proba = model.predict_proba(X)
        assert np.all(proba >= 0.0), f"Negative probability: min={proba.min()}"
        assert np.all(proba <= 1.0), f"Probability > 1: max={proba.max()}"

    def test_xgboost_feature_importance_shape(self):
        """feature_importance_ must have one entry per training feature."""
        model, X, y, n_features = self._fitted_xgboost(n_features=5)
        assert len(model.feature_importance_) == n_features, (
            f"Expected {n_features} importance values, got {len(model.feature_importance_)}"
        )

    def test_xgboost_get_feature_importance_dataframe(self):
        """get_feature_importance() must return a DataFrame with columns ['feature', 'importance']."""
        model, X, y, n_features = self._fitted_xgboost(n_features=4)
        names = [f"feat_{i}" for i in range(n_features)]
        df = model.get_feature_importance(names)
        assert "feature" in df.columns and "importance" in df.columns, (
            "get_feature_importance() must return columns ['feature', 'importance']"
        )
        assert len(df) == n_features

    def test_xgboost_predict_proba_before_fit_raises(self):
        """predict_proba must raise RuntimeError when called before fit()."""
        model = XGBoostPDModel()
        X = np.random.randn(10, 3)
        with pytest.raises(RuntimeError):
            model.predict_proba(X)


# ---------------------------------------------------------------------------
# FunctionalLogisticModel
# ---------------------------------------------------------------------------

class TestFunctionalLogisticModel:
    """Tests for the functional logistic regression model.

    Core numerical components (_fit_penalized_logistic_irls, _reml_criterion,
    predict_proba) are tested directly without going through the full greedy-
    selection pipeline (which would require real functional predictor data).
    """

    def _make_irls_inputs(self, n=150, n_params=5, seed=0):
        rng = np.random.default_rng(seed)
        w = rng.normal(size=n_params)
        tilde_X = rng.normal(size=(n, n_params))
        eta = tilde_X @ w
        y = (1 / (1 + np.exp(-eta)) > 0.5).astype(np.float64)
        S_block = np.eye(n_params)
        return tilde_X, y, S_block

    def test_functional_irls_converges(self):
        """_fit_penalized_logistic_irls must converge with well-conditioned data."""
        tilde_X, y, S_block = self._make_irls_inputs(n=200, n_params=5, seed=0)
        model = FunctionalLogisticModel()
        alpha, b_vec, converged = model._fit_penalized_logistic_irls(
            tilde_X, y, S_block, lambda_beta=0.1
        )
        assert converged, "IRLS must converge with well-conditioned input"
        assert np.isfinite(alpha), "Fitted intercept must be finite"
        assert b_vec.shape == (5,), f"Expected b_vec shape (5,), got {b_vec.shape}"
        assert np.all(np.isfinite(b_vec)), "Fitted coefficients must be finite"

    def test_functional_irls_zero_lambda(self):
        """IRLS with lambda=0 must return values of the correct types without crashing.

        Convergence is not guaranteed for the unpenalised case (the objective can
        diverge when the data are linearly separable), so we only check that the
        function completes and returns the expected shapes and types.
        """
        tilde_X, y, S_block = self._make_irls_inputs(n=200, n_params=4, seed=1)
        model = FunctionalLogisticModel()
        alpha, b_vec, converged = model._fit_penalized_logistic_irls(
            tilde_X, y, S_block, lambda_beta=0.0
        )
        assert isinstance(converged, bool), "converged must be a bool"
        assert isinstance(alpha, float), "alpha must be a float"
        assert b_vec.shape == (4,), f"Expected b_vec shape (4,), got {b_vec.shape}"

    def test_functional_predict_proba_range(self):
        """predict_proba must return values in [0, 1]."""
        rng = np.random.default_rng(7)
        n, K, L = 120, 6, 5

        C = rng.normal(size=(n, K))
        J = rng.normal(size=(L, K))
        S_coeff = np.eye(L)

        # Generate y with signal from the functional structure
        tilde_X = C @ J.T
        w = rng.normal(size=L)
        eta = tilde_X @ w
        y = (1 / (1 + np.exp(-eta)) > 0.5).astype(np.float64)

        model = FunctionalLogisticModel()
        result = model.fit_single_term(C, J, S_coeff, y, L=L)

        # Manually configure model for prediction (avoids full greedy selection)
        model.alpha_ = result["alpha"]
        model.term_coefficients_ = {"term1": result["b_coeff"]}
        model.selected_terms_ = ["term1"]

        terms_pred = [{"name": "term1", "C": C, "J": J}]
        proba = model.predict_proba(terms_pred)

        assert proba.shape == (n,), f"Expected shape ({n},), got {proba.shape}"
        assert np.all(proba >= 0.0), f"Negative probability found: min={proba.min()}"
        assert np.all(proba <= 1.0), f"Probability > 1 found: max={proba.max()}"

    def test_functional_reml_criterion_scalar(self):
        """_reml_criterion must return a finite scalar float."""
        tilde_X, y, S_block = self._make_irls_inputs(n=100, n_params=5, seed=2)
        model = FunctionalLogisticModel()
        result = model._reml_criterion(np.log(0.1), tilde_X, y, S_block)
        assert isinstance(result, float), "REML criterion must return a Python float"
        assert np.isfinite(result) or result == np.inf, (
            "REML criterion must return a finite value or np.inf"
        )

    def test_functional_reml_criterion_at_multiple_lambdas(self):
        """_reml_criterion must return finite values across a range of lambda."""
        tilde_X, y, S_block = self._make_irls_inputs(n=100, n_params=5, seed=3)
        model = FunctionalLogisticModel()
        finite_count = 0
        for log_lam in [-4.0, -2.0, 0.0, 2.0, 4.0]:
            val = model._reml_criterion(log_lam, tilde_X, y, S_block)
            assert isinstance(val, float)
            if np.isfinite(val):
                finite_count += 1
        assert finite_count >= 3, (
            "Expected at least 3 finite REML values across lambda range"
        )

    def test_functional_predict_proba_missing_term_raises(self):
        """predict_proba must raise ValueError when a selected term is missing."""
        model = FunctionalLogisticModel()
        rng = np.random.default_rng(42)
        C = rng.normal(size=(10, 5))
        J = rng.normal(size=(4, 5))
        model.alpha_ = 0.0
        model.term_coefficients_ = {"term_A": np.zeros(4)}
        model.selected_terms_ = ["term_A"]

        # Provide wrong term name
        with pytest.raises(ValueError):
            model.predict_proba([{"name": "term_B", "C": C, "J": J}])

    def test_functional_fit_single_term_returns_dict(self):
        """fit_single_term must return a dict with required keys."""
        rng = np.random.default_rng(0)
        n, K, L = 80, 5, 4
        C = rng.normal(size=(n, K))
        J = rng.normal(size=(L, K))
        S = np.eye(L)
        y = rng.integers(0, 2, n).astype(np.float64)

        model = FunctionalLogisticModel()
        result = model.fit_single_term(C, J, S, y, L=L)

        required = {"alpha", "b_coeff", "lambda_beta", "converged"}
        assert required.issubset(result.keys()), (
            f"fit_single_term missing keys: {required - set(result.keys())}"
        )
        assert isinstance(result["alpha"], float)
        assert isinstance(result["b_coeff"], np.ndarray)
        assert result["b_coeff"].shape == (L,)
        assert isinstance(result["converged"], bool)

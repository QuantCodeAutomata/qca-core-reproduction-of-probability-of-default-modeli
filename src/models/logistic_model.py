"""Scalar logistic regression model with greedy forward feature selection for PD modeling."""

from __future__ import annotations

import warnings
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


class LogisticPDModel:
    """Scalar logistic regression PD model with greedy forward feature selection.

    Fits a logistic regression model by greedily selecting features that
    maximise validation AUC, then refits the final model on training data
    only using the selected feature set.

    Attributes
    ----------
    selected_features_ : list[str]
        Ordered list of feature names chosen during forward selection.
    coef_ : np.ndarray of shape (n_selected_features,)
        Coefficients of the final fitted logistic regression model.
    intercept_ : float
        Intercept of the final fitted logistic regression model.
    scaler_ : StandardScaler
        Scaler fitted on the training data of the selected features.
    """

    def __init__(self) -> None:
        self.selected_features_: List[str] = []
        self.coef_: np.ndarray = np.array([], dtype=np.float64)
        self.intercept_: float = 0.0
        self.scaler_: StandardScaler = StandardScaler()
        self._selection_history: List[dict] = []
        self._model: Optional[LogisticRegression] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fit_candidate(
        self,
        X_train_scaled: np.ndarray,
        y_train: np.ndarray,
        X_val_scaled: np.ndarray,
        y_val: np.ndarray,
        random_state: int,
    ) -> float:
        """Fit a LogisticRegression on scaled data and return validation AUC.

        Parameters
        ----------
        X_train_scaled : np.ndarray
            Scaled training feature matrix.
        y_train : np.ndarray
            Binary training labels.
        X_val_scaled : np.ndarray
            Scaled validation feature matrix.
        y_val : np.ndarray
            Binary validation labels.
        random_state : int
            Random state for reproducibility.

        Returns
        -------
        float
            Validation ROC-AUC score.
        """
        lr = LogisticRegression(
            solver="liblinear",
            max_iter=1000,
            random_state=random_state,
        )
        lr.fit(X_train_scaled, y_train)
        y_prob = lr.predict_proba(X_val_scaled)[:, 1]
        return float(roc_auc_score(y_val, y_prob))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        candidate_features: Optional[List[str]] = None,
        auc_threshold: float = 0.0005,
        max_features: int = 10,
        random_state: int = 42,
    ) -> "LogisticPDModel":
        """Greedy forward feature selection maximising validation AUC.

        At each step the feature that yields the largest validation AUC
        improvement is added, provided the improvement exceeds
        ``auc_threshold``.  Selection stops when no candidate clears the
        threshold or ``max_features`` features have been chosen.  The final
        model is fitted on training data only (not train + val) to avoid
        leakage.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training feature matrix with named columns.
        y_train : np.ndarray
            Binary training labels (0 = non-default, 1 = default).
        X_val : pd.DataFrame
            Validation feature matrix with named columns.
        y_val : np.ndarray
            Binary validation labels.
        candidate_features : list[str] or None
            Column names to consider during selection.  If *None*, all
            columns of ``X_train`` are used.
        auc_threshold : float, default 0.0005
            Minimum AUC improvement required to add a feature.
        max_features : int, default 10
            Maximum number of features to select.
        random_state : int, default 42
            Random seed for LogisticRegression.

        Returns
        -------
        self : LogisticPDModel
            Fitted model instance.

        Raises
        ------
        ValueError
            If ``X_train`` and ``X_val`` do not share the same columns, or if
            ``candidate_features`` contains names absent from ``X_train``.
        """
        y_train = np.asarray(y_train, dtype=np.float64)
        y_val = np.asarray(y_val, dtype=np.float64)

        if candidate_features is None:
            candidate_features = list(X_train.columns)
        else:
            missing = set(candidate_features) - set(X_train.columns)
            if missing:
                raise ValueError(
                    f"candidate_features contains columns not in X_train: {missing}"
                )

        remaining: List[str] = list(candidate_features)
        selected: List[str] = []
        history: List[dict] = []
        current_auc: float = 0.0

        while remaining and len(selected) < max_features:
            best_auc: float = current_auc
            best_feature: Optional[str] = None

            for feat in remaining:
                trial_features = selected + [feat]

                # Scale using training data only
                scaler_tmp = StandardScaler()
                X_tr_s = scaler_tmp.fit_transform(
                    X_train[trial_features].to_numpy(dtype=np.float64)
                )
                X_vl_s = scaler_tmp.transform(
                    X_val[trial_features].to_numpy(dtype=np.float64)
                )

                try:
                    auc = self._fit_candidate(
                        X_tr_s, y_train, X_vl_s, y_val, random_state
                    )
                except Exception as exc:  # noqa: BLE001
                    warnings.warn(
                        f"Skipping feature '{feat}' due to fitting error: {exc}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    continue

                if auc > best_auc:
                    best_auc = auc
                    best_feature = feat

            delta = best_auc - current_auc
            if best_feature is None or delta <= auc_threshold:
                break

            selected.append(best_feature)
            remaining.remove(best_feature)
            history.append(
                {
                    "feature": best_feature,
                    "val_auc": best_auc,
                    "delta_auc": delta,
                }
            )
            current_auc = best_auc

        if not selected:
            warnings.warn(
                "No feature improved validation AUC beyond the threshold. "
                "Model has no selected features.",
                UserWarning,
                stacklevel=2,
            )

        # Refit final model on training data only
        self.selected_features_ = selected
        self._selection_history = history

        if selected:
            X_tr_final = X_train[selected].to_numpy(dtype=np.float64)
            self.scaler_ = StandardScaler()
            X_tr_scaled = self.scaler_.fit_transform(X_tr_final)

            self._model = LogisticRegression(
                solver="liblinear",
                max_iter=1000,
                random_state=random_state,
            )
            self._model.fit(X_tr_scaled, y_train)
            self.coef_ = self._model.coef_.flatten().astype(np.float64)
            self.intercept_ = float(self._model.intercept_[0])
        else:
            self._model = None
            self.coef_ = np.array([], dtype=np.float64)
            self.intercept_ = 0.0

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return predicted probability of default.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix containing at least the selected feature columns.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted default probabilities in [0, 1].

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        ValueError
            If required feature columns are missing from ``X``.
        """
        if self._model is None:
            raise RuntimeError(
                "Model is not fitted. Call fit() before predict_proba()."
            )
        missing = set(self.selected_features_) - set(X.columns)
        if missing:
            raise ValueError(
                f"Input X is missing required feature columns: {missing}"
            )
        X_arr = X[self.selected_features_].to_numpy(dtype=np.float64)
        X_scaled = self.scaler_.transform(X_arr)
        return self._model.predict_proba(X_scaled)[:, 1].astype(np.float64)

    def get_selected_features(self) -> List[str]:
        """Return the list of features selected during forward selection.

        Returns
        -------
        list[str]
            Ordered list of selected feature names (first added = index 0).
        """
        return list(self.selected_features_)

    def get_selection_history(self) -> List[dict]:
        """Return the step-by-step forward selection log.

        Returns
        -------
        list[dict]
            Each dict contains:
            - ``'feature'`` (str): feature added at this step.
            - ``'val_auc'`` (float): validation AUC after adding the feature.
            - ``'delta_auc'`` (float): AUC improvement over the previous step.
        """
        return [dict(entry) for entry in self._selection_history]

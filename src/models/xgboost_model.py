"""XGBoost PD model with Optuna-based hyperparameter tuning."""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.getLogger("xgboost").setLevel(logging.WARNING)


class XGBoostPDModel:
    """XGBoost probability-of-default model with Optuna hyperparameter tuning.

    Hyperparameters are selected by maximising validation ROC-AUC using an
    Optuna TPE study.  The final model is refitted on training data only
    (using the best hyperparameters and early stopping) to avoid leakage.

    Attributes
    ----------
    model_ : xgb.XGBClassifier
        The fitted XGBoost classifier.
    best_params_ : dict
        Best hyperparameter configuration found by Optuna.
    feature_importance_ : np.ndarray
        Feature importances from the final fitted model (gain-based).
    """

    def __init__(self) -> None:
        self.model_: Optional[xgb.XGBClassifier] = None
        self.best_params_: dict = {}
        self.feature_importance_: np.ndarray = np.array([], dtype=np.float64)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_objective(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        random_state: int,
    ):
        """Build an Optuna objective closure over the provided data splits.

        Parameters
        ----------
        X_train : np.ndarray
            Training feature matrix.
        y_train : np.ndarray
            Binary training labels.
        X_val : np.ndarray
            Validation feature matrix.
        y_val : np.ndarray
            Binary validation labels.
        random_state : int
            Random seed for XGBoost.

        Returns
        -------
        Callable
            Optuna objective function that returns negative validation AUC
            (Optuna minimises by default).
        """

        def objective(trial: optuna.Trial) -> float:
            params = {
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree", 0.6, 1.0
                ),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "reg_alpha": trial.suggest_float(
                    "reg_alpha", 1e-8, 1.0, log=True
                ),
                "reg_lambda": trial.suggest_float(
                    "reg_lambda", 1e-8, 10.0, log=True
                ),
                "n_estimators": 1000,
                "early_stopping_rounds": 50,
                "random_state": random_state,
                "verbosity": 0,
            }

            clf = xgb.XGBClassifier(**params)
            clf.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            y_prob = clf.predict_proba(X_val)[:, 1]
            auc = float(roc_auc_score(y_val, y_prob))
            return -auc  # minimise negative AUC

        return objective

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        random_state: int = 42,
        n_trials: int = 50,
    ) -> "XGBoostPDModel":
        """Tune hyperparameters with Optuna and refit the best model.

        Runs an Optuna study to maximise validation ROC-AUC over
        ``n_trials`` trials, then refits the model on training data using
        the best hyperparameters with early stopping on the validation set.

        Parameters
        ----------
        X_train : np.ndarray
            Training feature matrix of shape (n_train, n_features).
        y_train : np.ndarray
            Binary training labels of shape (n_train,).
        X_val : np.ndarray
            Validation feature matrix of shape (n_val, n_features).
        y_val : np.ndarray
            Binary validation labels of shape (n_val,).
        random_state : int, default 42
            Master random seed for the Optuna sampler and XGBoost.
        n_trials : int, default 50
            Number of Optuna trials to run.

        Returns
        -------
        self : XGBoostPDModel
            Fitted model instance.

        Raises
        ------
        ValueError
            If training or validation arrays are empty.
        """
        X_train = np.asarray(X_train, dtype=np.float64)
        y_train = np.asarray(y_train, dtype=np.float64)
        X_val = np.asarray(X_val, dtype=np.float64)
        y_val = np.asarray(y_val, dtype=np.float64)

        if X_train.size == 0 or X_val.size == 0:
            raise ValueError("Training and validation arrays must not be empty.")

        sampler = optuna.samplers.TPESampler(seed=random_state)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        objective = self._make_objective(
            X_train, y_train, X_val, y_val, random_state
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best = study.best_params.copy()
        self.best_params_ = best

        # Refit on training data with best params and early stopping
        final_params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "max_depth": best["max_depth"],
            "learning_rate": best["learning_rate"],
            "subsample": best["subsample"],
            "colsample_bytree": best["colsample_bytree"],
            "min_child_weight": best["min_child_weight"],
            "reg_alpha": best["reg_alpha"],
            "reg_lambda": best["reg_lambda"],
            "n_estimators": 1000,
            "early_stopping_rounds": 50,
            "random_state": random_state,
            "verbosity": 0,
        }
        self.model_ = xgb.XGBClassifier(**final_params)
        self.model_.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        self.feature_importance_ = np.asarray(
            self.model_.feature_importances_, dtype=np.float64
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return predicted probability of default.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted default probabilities in [0, 1].

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        if self.model_ is None:
            raise RuntimeError(
                "Model is not fitted. Call fit() before predict_proba()."
            )
        X = np.asarray(X, dtype=np.float64)
        return self.model_.predict_proba(X)[:, 1].astype(np.float64)

    def get_feature_importance(
        self, feature_names: List[str]
    ) -> pd.DataFrame:
        """Return a DataFrame of feature importances sorted in descending order.

        Parameters
        ----------
        feature_names : list[str]
            Names corresponding to the columns used during training.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ``['feature', 'importance']``, sorted by
            ``importance`` descending.

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        ValueError
            If the length of ``feature_names`` does not match the number of
            training features.
        """
        if self.model_ is None:
            raise RuntimeError(
                "Model is not fitted. Call fit() before get_feature_importance()."
            )
        if len(feature_names) != len(self.feature_importance_):
            raise ValueError(
                f"Length of feature_names ({len(feature_names)}) does not match "
                f"number of model features ({len(self.feature_importance_)})."
            )
        df = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": self.feature_importance_,
            }
        )
        return df.sort_values("importance", ascending=False).reset_index(drop=True)

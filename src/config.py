"""Central configuration for the PD modelling repository.

All key parameters — data splits, model hyperparameters, and paper
reproduction targets — are collected here so that every experiment script
imports from a single authoritative source.
"""

from __future__ import annotations

CONFIG: dict = {
    # ------------------------------------------------------------------ #
    # Reproducibility
    # ------------------------------------------------------------------ #
    "random_seed": 42,
    "bootstrap_seed": 123,

    # ------------------------------------------------------------------ #
    # Sample-period boundaries (development and out-of-time)
    # ------------------------------------------------------------------ #
    "dev_start": "2017-08-31",
    "dev_end": "2019-12-31",
    "oot_start": "2021-01-31",
    "oot_end": "2024-12-31",

    # ------------------------------------------------------------------ #
    # Split sizes
    # ------------------------------------------------------------------ #
    "train_size": 73152,
    "val_size": 31352,
    "test_size": 25285,

    # ------------------------------------------------------------------ #
    # Target construction
    # ------------------------------------------------------------------ #
    "min_lookback_months": 12,
    "default_horizon_months": 12,

    # ------------------------------------------------------------------ #
    # Scalar (tabular) feature configuration
    # ------------------------------------------------------------------ #
    "scalar_variables": [
        "gearing",
        "relative_overdraft",
        "available_funds",
        "overdraft",
        "utilization",
    ],
    "scalar_horizons": [0, 3, 6, 9, 12],

    # ------------------------------------------------------------------ #
    # Functional feature configuration
    # ------------------------------------------------------------------ #
    "functional_horizons": [3, 6, 9, 12],
    "functional_lambda_x_candidates": [0, 1e-4, 1e-3, 1e-2, 1e-1, 1],
    "functional_basis_types": ["cubic_spline", "mean_spline"],

    # ------------------------------------------------------------------ #
    # Functional logistic regression
    # ------------------------------------------------------------------ #
    "L_coeff_basis": 5,          # Number of B-spline basis functions for beta(s)
    "auc_improvement_threshold": 0.0005,

    # ------------------------------------------------------------------ #
    # Pre-processing
    # ------------------------------------------------------------------ #
    "winsorization_lower": 0.02,
    "winsorization_upper": 0.98,
    "denominator_floor_dkk": 1000.0,

    # ------------------------------------------------------------------ #
    # Loss function parameters
    # ------------------------------------------------------------------ #
    "log_loss_clip_eps": 1e-15,
    "focal_loss_gamma": 2.0,
    "focal_loss_alpha": 0.25,

    # ------------------------------------------------------------------ #
    # Statistical testing
    # ------------------------------------------------------------------ #
    "delong_alpha": 0.05,

    # ------------------------------------------------------------------ #
    # Bootstrap inference
    # ------------------------------------------------------------------ #
    "n_bootstrap": 200,
    "bootstrap_dense_grid_points": 200,

    # ------------------------------------------------------------------ #
    # Synthetic-data experiment
    # ------------------------------------------------------------------ #
    "n_synthetic_borrowers": 800,

    # ------------------------------------------------------------------ #
    # XGBoost tuning
    # ------------------------------------------------------------------ #
    "xgboost_n_trials": 15,

    # ------------------------------------------------------------------ #
    # Paper reproduction targets (Table 3 / Figure 5 of the reference paper)
    # ------------------------------------------------------------------ #
    "paper_target_aucs": {
        "test": {
            "functional": 0.8996,
            "logistic": 0.8902,
            "xgboost": 0.8940,
        },
        "oot": {
            "functional": 0.8901,
            "logistic": 0.8823,
            "xgboost": 0.8852,
        },
    },
}

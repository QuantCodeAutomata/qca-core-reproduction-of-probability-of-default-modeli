# Repository Knowledge — PD Modelling

## Purpose
Reproduces a paper comparing scalar logistic regression, XGBoost, and functional
logistic regression for Probability of Default (PD) modelling.

## Source layout (`src/`)
| File | Contents |
|------|----------|
| `config.py` | Single `CONFIG` dict (30 keys) with all experiment parameters |
| `evaluation.py` | Metric functions + DeLong paired AUC test + Holm-Bonferroni |
| `calendar_utils.py` | Danish banking-day calendar; Easter (Meeus/Jones/Butcher); holiday list; relative-time grid τ |
| `preprocessing.py` | LOCF EOD reconstruction; account/borrower/household aggregation; sampling dates; winsorization |
| `feature_engineering.py` | 50 scalar features; functional predictors (cubic_spline / mean_spline); B-spline helpers |
| `models/__init__.py` | Empty |
| `models/logistic_model.py` | `LogisticPDModel` — greedy forward selection via val AUC |
| `models/xgboost_model.py` | `XGBoostPDModel` — Optuna TPE tuning (n_trials=30 default) |
| `models/functional_logistic.py` | `FunctionalLogisticModel` — IRLS + REML + greedy selection |

## Data layer (`data/`)
| File | Contents |
|------|----------|
| `synthetic_generator.py` | `generate_synthetic_banking_data(n_borrowers, seed, start_date, end_date)` → `{'accounts', 'balances', 'defaults'}` |
| | `save_synthetic_data(data_dict, output_dir)` → writes three CSVs |

## Key implementation details

### synthetic_generator
- AR(1) per account type: deposit (α≈0.93, positive), credit (α≈0.90, clipped to [−limit, 0]),
  loan (amortising, payment-driven), investment (α≈0.96, upward drift).
- Default generation: monthly pass, base prob=0.004, stress factor = 1 + 5·max(util−0.70, 0).
  Once defaulted, borrower is removed from at-risk pool.
- Balances stored as **monthly end-of-month snapshots**; preprocessing LOCF fills daily gaps.

### calendar_utils
- `generate_banking_days`: Mon–Fri minus DK public holidays.
- Danish holidays computed from Easter (Meeus/Jones/Butcher algorithm):
  - Good Friday = Easter − 2, Easter Monday = Easter + 1, Ascension = Easter + 39,
    Whit Monday = Easter + 50, Store Bededag = Easter + 19 (**only for year < 2024**),
    plus fixed dates: Jan 1, Dec 24/25/26/31.
- `get_month_end_banking_days`: returns pd.Series with PeriodIndex (freq='M') → Timestamp.
- `get_relative_time_grid`: τ_j^(m) = j/k_m + (m−1); m=1 is oldest month, m=M is most recent.

### preprocessing
- `reconstruct_eod_states`: builds (account_id, date) grid sorted by **date** (not account_id),
  then uses `pd.merge_asof(by='account_id', direction='backward')` for LOCF in one pass.
  **Critical**: left DataFrame must be sorted by 'date' globally for merge_asof to work.
- `assign_sampling_dates`: 70/30 dev/OOT split by default; validity window = [bd_start + 12 months,
  bd_end − 12 months].
- `create_default_labels`: cross-joins sampling with defaults, checks (T, T+horizon_months].

### feature_engineering
- Scalar features: 5 vars × 2 levels × 5 horizons = 50 columns.
  Column naming: `'{level}_{var}_M{horizon}'` e.g. `'bw_gearing_M12'`, `'hh_utilization_M0'`.
  Level prefixes are `'bw'` and `'hh'` (no trailing underscore in the name segment).
- `build_functional_predictors`: for `cubic_spline`, knots built once with `_make_clamped_knots`;
  roughness penalty matrix S computed once and reused across all borrowers.
  Penalised solve: `(B^T B + λ S) c = B^T y`; falls back to lstsq if matrix is singular.
- `compute_spline_basis_matrix`: clamped B-spline (degree+1 repeated knots at boundaries).
  Rows sum to 1.0 (partition of unity) inside the domain.
- `compute_roughness_penalty_matrix`: Gauss–Legendre quadrature (n_quad=200 default);
  result is symmetric PSD of shape (n_basis, n_basis).
- `compute_inner_product_matrix`: J shape is (L, K) = (len(phi_basis), len(psi_basis)).

### LogisticPDModel
- Greedy forward selection over `candidate_features` using validation AUC.
- `auc_threshold=0.0005`, `max_features=10` by default.
- Final model fitted on **train only** (not train+val) — leakage safety.
- Uses `StandardScaler` fitted on training columns only.
- sklearn `LogisticRegression(solver='liblinear', max_iter=1000)`.

### XGBoostPDModel
- XGBoost ≥ 2.x API: `early_stopping_rounds` goes in the **constructor**, not in `.fit()`.
- `use_label_encoder` parameter was removed in XGBoost 2.x — do not add it.
- Optuna study minimises **negative** AUC (direction="minimize").
- `n_estimators=1000` with `early_stopping_rounds=50` in both tuning and final fit.

### FunctionalLogisticModel
- Inner-product design matrix: `tilde_X_term = C @ J.T` → shape `(n_obs, L)`.
- IRLS augments with an intercept column: `X_aug = [ones | tilde_X]`.
- Penalty block `S_aug = block_diag(0, lambda * S_block)` (intercept unpenalised).
- REML criterion minimised over log-spaced grid [1e-6, 1e3] then refined with
  `scipy.optimize.minimize_scalar(method='bounded')`.
- B-spline knot vector for L=5 cubic basis: degree+1 clamped knots at each end
  plus `L - degree - 1 = 1` internal knot → 9 total knots.
- `terms_list` dicts require keys: `name`, `C_train`, `C_val`, `J`, `S_pred`,
  `S_coeff`, `domain_end`.

### Evaluation
- `delong_auc_variance` returns `(auc1, auc2, delta_auc, z_stat, p_value)`.
- `holm_bonferroni_adjust` returns adjusted p-values in **original** order.
- `evaluate_model_suite` returns DataFrame with index=model_name,
  columns=[`auc`, `brier`, `log_loss`, `weighted_log_loss`, `focal_loss`].

## Dependencies
Core: `numpy`, `pandas`, `scipy`, `scikit-learn`, `xgboost`, `optuna`.
Install optuna if missing: `pip install optuna`.

## Experiment scripts (`exp/`)
All four experiments share the same data pipeline and are runnable from the project root.

| Script | Purpose | Key output |
|--------|---------|------------|
| `exp_1_main.py` | Main 3-model comparison (logistic, XGBoost, functional) | `results/exp1_metrics.csv`, `results/exp1_delong.csv`, `results/exp1_coeff_functions.png`, `results/exp1_auc_comparison.png` |
| `exp_2_month_end.py` | Month-end-only functional predictor robustness | `results/exp2_metrics.csv`, `results/exp2_delong.csv`, `results/exp2_auc_comparison.png`, `results/exp2_coeff_functions.png` |
| `exp_3_bootstrap.py` | Bootstrap 90%/95% CI bands for β(s) | `results/exp3_bootstrap_coeff_functions.png`, `results/exp3_bootstrap_summary.csv`; add `--fast` for B=50 |
| `exp_4_basis_robustness.py` | Benchmark vs Fourier / polynomial / mixed predictor bases | `results/exp4_metrics.csv`, `results/exp4_comparison.png` |

### Functional term construction pattern (all experiments)
1. `build_functional_predictors(...)` → `(C_matrix, basis_info)` for each split.
2. Impute NaN rows with training column means.
3. Build coefficient-basis knots: `_make_clamped_knots(domain_end, L=5)`.
4. `compute_roughness_penalty_matrix(coeff_knots, ...)` → `S_coeff` (5×5).
5. For `cubic_spline` predictor: `compute_inner_product_matrix(phi_params, psi_params)` → `J` (5×K).
6. For `mean_spline` predictor: compute `J[l,k] = ∫_{k-1}^{k} φ_l(s) ds` via G-L quadrature.
7. Assemble term dict with keys `{name, C_train, C_val, J, S_pred, S_coeff, domain_end}` for `FunctionalLogisticModel.fit`.
8. For `predict_proba`: pass `{name, C, J}` per term.

### Bootstrap (exp_3)
After the full `FunctionalLogisticModel.fit`, fix `lambda_beta_` and call
`model._fit_penalized_logistic_irls(tilde_X_boot, y_boot, S_block, lambda_beta_fixed)`
for each resample to avoid repeated REML estimation. Uses `joblib.Parallel` if available.

### Alternative bases (exp_4)
- Fourier predictor: `[1, sin(2π·k·t/M), cos(2π·k·t/M), ...]`; roughness penalty is analytic diagonal.
- Polynomial predictor: `[1, t, t², t³]`; roughness penalty computed numerically.
- Mixed model: cubic-spline `C` matrices (same as benchmark) but Fourier coefficient basis for `J` and `S_coeff`.

## Python path
Add project root to `sys.path` before importing:
```python
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```

## Test suite (`tests/`)
108 tests across 5 files; run with `python -m pytest tests/ -v`.

| File | Tests | Coverage |
|------|-------|----------|
| `test_calendar.py` | 17 | All calendar_utils functions; holiday exclusion; relative-time grid properties |
| `test_evaluation.py` | 34 | Every metric in evaluation.py; DeLong exact properties; Holm-Bonferroni math |
| `test_preprocessing.py` | 21 | Full pipeline: LOCF, formulas, gearing floor, label window, winsorization, sampling |
| `test_features.py` | 17 | 50 scalar features; B-spline POU; PSD roughness penalty; inner-product shape; functional predictor shapes |
| `test_models.py` | 19 | All three model classes; greedy selection; IRLS convergence; REML scalar; predict_proba range |

### Shared fixtures (`conftest.py`)
All fixtures are `scope="session"` (pipeline runs once per session):
- `synthetic_data` — 100 borrowers, seed=0
- `banking_days` — 2015-01-01 to 2025-12-31
- `month_end_days` — derived from banking_days
- `small_sample_data` — full pipeline dict with keys: `accounts_df`, `balances_df`,
  `defaults_df`, `eod_states`, `account_daily`, `borrower_daily`, `hh_daily`,
  `sampling_df`, `labeled_df`

### Test design notes
- No mocks — all tests exercise real code paths.
- `test_functional_irls_zero_lambda`: convergence not asserted for λ=0 because
  IRLS can diverge on separable data (unpenalised); only return-type correctness is checked.
- XGBoost tests use `n_trials=2` for speed (full Optuna machinery still exercised).
- `_make_clamped_knots` is imported directly from `feature_engineering` for
  roughness-penalty and inner-product tests.

# Probability of Default Modelling – Functional vs Scalar vs XGBoost

Reproduction of a retail-banking PD paper comparing three modelling approaches
on daily end-of-banking-day behavioural data aligned in relative banking-day time.

> **Note.** The original study uses proprietary Danish retail-banking records.
> This repository generates 3 000 synthetic borrowers that mirror the structural
> properties of the paper's dataset and reproduces the full methodology
> end-to-end for validation purposes.

---

## Repository Structure

```
repo/
├── data/
│   └── synthetic_generator.py   # Synthetic borrower / account / balance / default data
│
├── src/
│   ├── calendar_utils.py         # Danish banking-day calendar & relative-time mapping
│   ├── config.py                 # Central CONFIG dict (seeds, sample sizes, paths)
│   ├── preprocessing.py          # LOCF EOD reconstruction, derived vars, winsorization
│   ├── feature_engineering.py    # Scalar month-end features + functional predictors
│   ├── evaluation.py             # AUC, Brier, log loss, weighted log loss, focal loss
│   └── models/
│       ├── logistic_model.py     # Greedy-forward scalar logistic regression
│       ├── xgboost_model.py      # XGBoost with Optuna hyperparameter optimisation
│       └── functional_logistic.py # Penalised functional logistic regression (REML)
│
├── exp/
│   ├── exp_1_main.py             # Core comparison: Logistic / XGBoost / Functional
│   ├── exp_2_month_end.py        # Robustness: month-end-only functional model
│   ├── exp_3_bootstrap.py        # Bootstrap 90/95% pointwise confidence bands
│   └── exp_4_basis_robustness.py # Alternative basis families (Fourier / local poly / mixed)
│
├── tests/
│   ├── conftest.py
│   ├── test_preprocessing.py
│   ├── test_features.py
│   ├── test_models.py
│   └── test_evaluation.py
│
├── results/
│   ├── RESULTS.md                # Full metric tables and interpretation
│   ├── exp1_metrics.csv / exp1_delong.csv / *.png
│   ├── exp2_metrics.csv / exp2_delong.csv / *.png
│   ├── exp3_bootstrap_summary.csv / *.png
│   └── exp4_metrics.csv / *.png
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Methodology Overview

### Data pipeline (per experiment)
1. **Banking-day calendar** – identifies every valid Danish banking day and
   month-end banking day over the full raw-data period.
2. **Sampling date assignment** – one random end-of-month date T per borrower,
   with ≥ 12 months lookback and ≥ 12 months outcome horizon.
3. **EOD state reconstruction** – `pd.merge_asof` LOCF per (account, day).
4. **Derived variables** – available funds, utilisation, overdraft, gearing,
   relative overdraft; zero-denominator rules documented in `preprocessing.py`.
5. **Winsorization** – 2nd / 98th pctile cutoffs from training values at T,
   applied consistently to all splits.
6. **Scalar features** – 50 candidates (5 variables × 2 levels × 5 horizons).
7. **Functional predictors** – daily paths on relative time τ ∈ (0, M],
   smoothed with cubic B-splines or mean splines; coefficient functions with
   L = 5 basis functions, λ_β by REML.

### Models
| Model | Selection | Key detail |
|-------|-----------|------------|
| Scalar Logistic | Greedy forward (ΔAUCᵥₐₗ > 0.0005) | sklearn `LogisticRegression` |
| XGBoost | All 50 features; Optuna 50 trials | Early stopping on val AUC |
| Functional Logistic | Greedy forward over 280 functional candidates | IRLS + REML; L=5 |

### Evaluation
- Primary: **AUC** (scikit-learn)
- Secondary: Brier score, log loss, weighted log loss (w₁ ≈ 1/default rate),
  focal loss (γ=2, α=0.25)
- Inference: paired **DeLong** test with **Holm–Bonferroni** correction

---

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Run all four experiments
python exp/exp_1_main.py
python exp/exp_2_month_end.py
python exp/exp_3_bootstrap.py
python exp/exp_4_basis_robustness.py

# Run tests
pytest tests/ -v
```

Results (metrics CSVs + plots) are written to `results/`.

---

## Environment

Python 3.12.  Key dependencies: `numpy`, `pandas`, `scipy`, `scikit-learn`,
`xgboost`, `optuna`, `joblib`, `matplotlib`, `seaborn`.

See `requirements.txt` for the full pinned list.

---

## Results Summary

See [`results/RESULTS.md`](results/RESULTS.md) for full metric tables,
DeLong statistics, and interpretation of each experiment.

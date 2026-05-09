---
description: Required robustness tests for any ML-based strategy or filter before deployment
globs: experiments/run_*ml*.py, experiments/run_*xgb*.py, experiments/run_*lgbm*.py
---

# ML Robustness Test Requirements

Any experiment that uses machine learning (XGBoost, LightGBM, neural nets, etc.) for trade filtering, signal generation, or exit optimization MUST include the following 5 robustness tests before claiming results are deployable.

## Mandatory Tests

1. **Independent Holdout** — Train on data before a cutoff (e.g., 2023-01-01), evaluate on data after. AUC must exceed 0.65 and filtered Sharpe must beat baseline.

2. **Parameter Perturbation** — Run at least 8 hyperparameter variants (depth, learning rate, n_estimators). AUC std must be < 0.05 across variants.

3. **Random Label Shuffle** — Permute labels 20 times, retrain each. Real AUC must be z > 3.0 above the shuffle distribution (p < 0.05).

4. **Feature Ablation** — Test subgroups (tech-only, macro-only, time-only, full). Verify which feature groups contribute genuine signal vs noise.

5. **Per-Fold AUC Stability** — Report per-fold AUC from walk-forward. Min fold AUC > 0.58, coefficient of variation < 0.20.

## Pass Criteria

- **4/5 or 5/5 PASS**: Model is robust, may proceed to deployment planning
- **3/5 PASS**: Investigate failures, may deploy with extra caution
- **2 or fewer PASS**: Do NOT deploy. Simplify model or gather more data.

## Template

Use `experiments/run_r92_ml_exit_robustness.py` as the reference implementation for these tests.

#!/usr/bin/env python3
"""
R170b — ML Exit V3: Mandatory Robustness Suite (per .cursor/rules/ml-robustness-tests.md)

Runs the same 5 tests as R92 on each R170 P6 strategy's trade-level feature matrix
(engine + features aligned with run_r170_ml_exit_v3.py).

Tests:
  1. Independent Holdout — train before 2023-01-01, test after; AUC > 0.65 and
     filtered Sharpe > baseline Sharpe on holdout exits.
  2. Parameter Perturbation — ≥8 XGB variants; PASS if std(AUC) < 0.05 (walk-forward OOS).
  3. Random Label Shuffle — 20 permutations; real AUC z > 3 vs shuffle distribution, p < 0.05.
  4. Feature Ablation — tech-only / macro-only / time-only / full walk-forward AUC.
  5. Per-Fold AUC Stability — min fold AUC > 0.58, CV < 0.20.

Pass summary: 4/5 or 5/5 strategies may proceed per strategy; aggregate counts printed.

Reference: experiments/run_r92_ml_exit_robustness.py (walk-forward structure).
"""
from __future__ import annotations

import json
import sys
import os
import time
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

from sklearn.metrics import roc_auc_score

# Reuse R170 pipeline
from experiments.run_r170_ml_exit_v3 import (
    HAS_XGB,
    STRAT_ORDER,
    STRATEGY_BT_MAP,
    P6_CAPS,
    UNIT_LOT,
    SPREAD,
    ALL_FEATURES,
    TECH_FEATURES,
    MACRO_FEATURES,
    TIME_FEATURES,
    XGB_PARAMS,
    ML_THRESHOLD,
    load_h1,
    load_macro,
    build_h1_indicators,
    build_trade_features,
    _compute_stats,
    _sharpe,
    _trades_to_daily,
    KFOLD_FOLDS,
)

OUTPUT_DIR = Path('results/r170b_ml_robustness')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HOLDOUT_SPLIT = '2023-01-01'
N_SHUFFLE = 20
RNG = np.random.RandomState(42)

# ≥8 hyperparameter variants (depth / lr / n_estimators)
PARAM_GRID = [
    {'max_depth': 3, 'learning_rate': 0.05, 'n_estimators': 300},
    {'max_depth': 4, 'learning_rate': 0.05, 'n_estimators': 300},
    {'max_depth': 5, 'learning_rate': 0.03, 'n_estimators': 300},
    {'max_depth': 5, 'learning_rate': 0.05, 'n_estimators': 100},
    {'max_depth': 5, 'learning_rate': 0.05, 'n_estimators': 300},
    {'max_depth': 5, 'learning_rate': 0.05, 'n_estimators': 500},
    {'max_depth': 5, 'learning_rate': 0.08, 'n_estimators': 300},
    {'max_depth': 6, 'learning_rate': 0.05, 'n_estimators': 300},
]


def make_xgb(**overrides):
    from xgboost import XGBClassifier
    kw = {**XGB_PARAMS, **overrides}
    kw.setdefault('use_label_encoder', False)
    kw.setdefault('verbosity', 0)
    return XGBClassifier(**kw)


def _prep_xy(feat_df: pd.DataFrame, feature_cols: list[str]):
    X = feat_df[feature_cols].copy()
    y = feat_df['label'].values
    entry_times = pd.to_datetime(feat_df['entry_time'])
    if hasattr(entry_times.dt, 'tz') and entry_times.dt.tz is not None:
        entry_times = entry_times.dt.tz_localize(None)
    return X, y, entry_times


def _median_impute(train: pd.DataFrame, test: pd.DataFrame):
    med = train.median()
    Xt = train.fillna(med)
    Xv = test.fillna(med)
    const = [c for c in Xt.columns if Xt[c].nunique() <= 1]
    if const:
        Xt = Xt.drop(columns=const)
        Xv = Xv.drop(columns=const)
    return Xt, Xv


def train_walkforward_oos_auc(X: pd.DataFrame, y: np.ndarray, entry_times: pd.Series, model) -> dict:
    """Expanding train before fold start, test inside fold (same time logic as R92)."""
    oos_pred = np.full(len(y), np.nan)
    fold_aucs: list[float] = []

    for fold_name, fold_start, fold_end in KFOLD_FOLDS:
        fs = pd.Timestamp(fold_start)
        fe = pd.Timestamp(fold_end)
        train_m = entry_times < fs
        test_m = (entry_times >= fs) & (entry_times < fe)
        if train_m.sum() < 40 or test_m.sum() < 8:
            continue
        X_tr, X_te = _median_impute(X.loc[train_m], X.loc[test_m])
        y_tr, y_te = y[train_m.values], y[test_m.values]
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            continue
        if X_tr.shape[1] == 0:
            continue
        try:
            m = deepcopy(model)
            m.fit(X_tr.values, y_tr)
            prob = m.predict_proba(X_te.values)[:, 1]
            idx = np.where(test_m.values)[0]
            oos_pred[idx] = prob
            fold_aucs.append(float(roc_auc_score(y_te, prob)))
        except Exception:
            continue

    valid = ~np.isnan(oos_pred)
    overall = None
    if valid.sum() > 20 and len(np.unique(y[valid])) > 1:
        overall = float(roc_auc_score(y[valid], oos_pred[valid]))
    return {'overall_auc': overall, 'fold_aucs': fold_aucs, 'oos_pred': oos_pred}


def test1_holdout(X: pd.DataFrame, y: np.ndarray, entry_times: pd.Series,
                  feat_df: pd.DataFrame) -> dict:
    split_ts = pd.Timestamp(HOLDOUT_SPLIT)
    train_m = entry_times < split_ts
    test_m = entry_times >= split_ts
    if train_m.sum() < 40 or test_m.sum() < 15:
        return {'passed': False, 'reason': 'insufficient_holdout_split'}

    X_tr, X_te = _median_impute(X.loc[train_m], X.loc[test_m])
    y_tr, y_te = y[train_m.values], y[test_m.values]
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2 or X_tr.shape[1] == 0:
        return {'passed': False, 'reason': 'degenerate_split'}

    model = make_xgb()
    model.fit(X_tr.values, y_tr)
    prob = model.predict_proba(X_te.values)[:, 1]
    auc = float(roc_auc_score(y_te, prob))

    test_df = feat_df.loc[test_m].reset_index(drop=True)
    baseline_pnls = test_df['pnl'].values

    def sharpe_from_pnls_exit(pnls, exits):
        trades = [{'pnl': float(p), 'exit_time': ex} for p, ex in zip(pnls, exits)]
        return _compute_stats(trades)['sharpe']

    exits = test_df['exit_time'].tolist()
    b_sh = sharpe_from_pnls_exit(baseline_pnls, exits)
    filt_pnls = []
    filt_exits = []
    for i in range(len(prob)):
        if prob[i] >= ML_THRESHOLD:
            filt_pnls.append(baseline_pnls[i])
            filt_exits.append(exits[i])
    f_sh = sharpe_from_pnls_exit(np.array(filt_pnls), filt_exits)

    passed = auc > 0.65 and f_sh > b_sh
    return {
        'passed': passed,
        'holdout_auc': round(auc, 4),
        'baseline_sharpe': round(b_sh, 4),
        'filtered_sharpe': round(f_sh, 4),
        'n_train': int(train_m.sum()),
        'n_test': int(test_m.sum()),
    }


def test2_param_perturb(X: pd.DataFrame, y: np.ndarray, entry_times: pd.Series) -> dict:
    aucs = []
    for params in PARAM_GRID:
        model = make_xgb(**params)
        r = train_walkforward_oos_auc(X, y, entry_times, model)
        if r['overall_auc'] is not None:
            aucs.append(r['overall_auc'])
    if len(aucs) < 4:
        return {'passed': False, 'reason': 'insufficient_variants', 'aucs': aucs}
    std = float(np.std(aucs))
    passed = std < 0.05  # project rule
    return {
        'passed': passed,
        'std_auc': round(std, 4),
        'mean_auc': round(float(np.mean(aucs)), 4),
        'min_auc': round(float(np.min(aucs)), 4),
        'n_variants': len(aucs),
        'all_aucs': [round(a, 4) for a in aucs],
    }


def test3_shuffle(X: pd.DataFrame, y: np.ndarray, entry_times: pd.Series) -> dict:
    base = make_xgb()
    real = train_walkforward_oos_auc(X, y, entry_times, base)
    real_auc = real['overall_auc']
    if real_auc is None:
        return {'passed': False, 'reason': 'no_real_auc'}

    shuffled = []
    for i in range(N_SHUFFLE):
        y_perm = RNG.permutation(y)
        m = make_xgb()
        r = train_walkforward_oos_auc(X, y_perm, entry_times, m)
        if r['overall_auc'] is not None:
            shuffled.append(r['overall_auc'])
        if (i + 1) % 5 == 0:
            print(f'      shuffle {i+1}/{N_SHUFFLE}...', flush=True)

    if len(shuffled) < 8:
        return {'passed': False, 'reason': 'insufficient_shuffle_aucs'}

    mu, sd = float(np.mean(shuffled)), float(np.std(shuffled))
    z = (real_auc - mu) / sd if sd > 1e-12 else 0.0
    p_emp = sum(1 for s in shuffled if s >= real_auc) / len(shuffled)
    passed = z > 3.0 and p_emp < 0.05
    return {
        'passed': passed,
        'real_auc': round(real_auc, 4),
        'shuffle_mean': round(mu, 4),
        'shuffle_std': round(sd, 4),
        'z_score': round(z, 2),
        'p_value': round(p_emp, 4),
    }


def test4_ablation(X: pd.DataFrame, y: np.ndarray, entry_times: pd.Series,
                   feature_cols: list[str]) -> dict:
    groups = {
        'tech_only': [c for c in TECH_FEATURES if c in feature_cols],
        'macro_only': [c for c in MACRO_FEATURES if c in feature_cols],
        'time_only': [c for c in TIME_FEATURES if c in feature_cols],
        'full': feature_cols,
    }
    out = {}
    for name, cols in groups.items():
        if not cols:
            out[name] = None
            continue
        X_sub = X[cols].copy()
        model = make_xgb()
        r = train_walkforward_oos_auc(X_sub, y, entry_times, model)
        out[name] = r['overall_auc']

    full_a, tech_a, time_a = out.get('full'), out.get('tech_only'), out.get('time_only')
    external_adds = full_a is not None and tech_a is not None and full_a > tech_a
    time_ok = time_a is not None and time_a < 0.60
    passed = external_adds and time_ok
    return {
        'passed': passed,
        'full_auc': round(full_a, 4) if full_a else None,
        'tech_only_auc': round(tech_a, 4) if tech_a else None,
        'macro_only_auc': round(out['macro_only'], 4) if out.get('macro_only') else None,
        'time_only_auc': round(time_a, 4) if time_a else None,
        'external_adds_value': external_adds,
        'time_not_predictive': time_ok,
    }


def test5_fold_stability(X: pd.DataFrame, y: np.ndarray, entry_times: pd.Series) -> dict:
    model = make_xgb()
    r = train_walkforward_oos_auc(X, y, entry_times, model)
    folds = r['fold_aucs']
    if len(folds) < 3:
        return {'passed': False, 'reason': f'only_{len(folds)}_folds', 'fold_aucs': folds}

    mean_a = float(np.mean(folds))
    std_a = float(np.std(folds))
    cv = std_a / mean_a if mean_a > 0 else 999.0
    min_a = float(np.min(folds))
    passed = min_a > 0.58 and cv < 0.20
    return {
        'passed': passed,
        'fold_aucs': [round(x, 4) for x in folds],
        'mean_auc': round(mean_a, 4),
        'cv': round(cv, 4),
        'min_auc': round(min_a, 4),
    }


def run_strategy(strat_name: str, h1_df: pd.DataFrame, h1_ind: pd.DataFrame,
                 macro_df, feature_cols: list[str]) -> dict:
    print(f"\n  {'='*66}", flush=True)
    print(f"  Strategy: {strat_name}", flush=True)
    print(f"  {'='*66}", flush=True)

    bt_fn, bt_kw = STRATEGY_BT_MAP[strat_name]
    cap = P6_CAPS[strat_name]
    trades = bt_fn(h1_df, spread=SPREAD, lot=UNIT_LOT, maxloss_cap=cap, **bt_kw)
    feat_df = build_trade_features(trades, h1_ind, macro_df)
    if len(feat_df) < 80:
        print(f'    [SKIP] too few samples ({len(feat_df)})', flush=True)
        return {'strategy': strat_name, 'status': 'skipped', 'reason': 'too_few_samples'}

    use_cols = [c for c in feature_cols if c in feat_df.columns]
    if len(use_cols) < 5:
        print(f'    [SKIP] too few feature cols ({len(use_cols)})', flush=True)
        return {'strategy': strat_name, 'status': 'skipped', 'reason': 'too_few_features'}

    X, y, entry_times = _prep_xy(feat_df, use_cols)

    print(f'    samples={len(y)}, win_rate={y.mean()*100:.1f}%', flush=True)

    t1 = test1_holdout(X, y, entry_times, feat_df)
    print(f"    T1 Holdout: AUC={t1.get('holdout_auc')} base_Sharpe={t1.get('baseline_sharpe')} "
          f"filt_Sharpe={t1.get('filtered_sharpe')} -> {'PASS' if t1.get('passed') else 'FAIL'}", flush=True)

    t2 = test2_param_perturb(X, y, entry_times)
    print(f"    T2 Param perturb: std={t2.get('std_auc')} mean={t2.get('mean_auc')} -> "
          f"{'PASS' if t2.get('passed') else 'FAIL'}", flush=True)

    t3 = test3_shuffle(X, y, entry_times)
    print(f"    T3 Shuffle: z={t3.get('z_score')} p={t3.get('p_value')} -> "
          f"{'PASS' if t3.get('passed') else 'FAIL'}", flush=True)

    t4 = test4_ablation(X, y, entry_times, use_cols)
    print(f"    T4 Ablation: full={t4.get('full_auc')} tech={t4.get('tech_only_auc')} -> "
          f"{'PASS' if t4.get('passed') else 'FAIL'}", flush=True)

    t5 = test5_fold_stability(X, y, entry_times)
    print(f"    T5 Fold stability: min={t5.get('min_auc')} CV={t5.get('cv')} -> "
          f"{'PASS' if t5.get('passed') else 'FAIL'}", flush=True)

    tests = {'test1_holdout': t1, 'test2_param': t2, 'test3_shuffle': t3, 'test4_ablation': t4, 'test5_fold': t5}
    n_pass = sum(1 for v in tests.values() if isinstance(v, dict) and v.get('passed'))
    return {
        'strategy': strat_name,
        'status': 'ok',
        'n_trades_feat': len(feat_df),
        'tests': tests,
        'n_pass': n_pass,
        'deploy_band': 'robust' if n_pass >= 4 else 'do_not_deploy',
    }


def main():
    if not HAS_XGB:
        print('ERROR: xgboost required.', flush=True)
        sys.exit(1)

    t0 = time.time()
    print('=' * 70, flush=True)
    print('  R170b — ML Exit V3 Robustness (5 mandatory tests × 6 strategies)', flush=True)
    print('=' * 70, flush=True)

    h1_df = load_h1()
    macro_df = load_macro()
    h1_ind = build_h1_indicators(h1_df)
    feature_cols = [c for c in ALL_FEATURES]

    results = []
    for sn in STRAT_ORDER:
        results.append(run_strategy(sn, h1_df, h1_ind, macro_df, feature_cols))

    # Aggregate
    ok = [r for r in results if r.get('status') == 'ok']
    agg = {f'test{k}_pass_rate': 0.0 for k in range(1, 6)}
    test_keys = ['test1_holdout', 'test2_param', 'test3_shuffle', 'test4_ablation', 'test5_fold']
    if ok:
        for i, key in enumerate(test_keys):
            passes = sum(1 for r in ok if r['tests'][key].get('passed'))
            agg[f'test{i+1}_pass_rate'] = round(passes / len(ok), 3)

    summary = {
        'elapsed_s': round(time.time() - t0, 1),
        'strategies': results,
        'aggregate_pass_rates': agg,
        'note': 'Per ml-robustness-tests.md: 4/5 or 5/5 PASS per strategy for deployment planning.',
    }

    out_path = OUTPUT_DIR / 'r170b_robustness_summary.json'
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print('\n' + '=' * 70, flush=True)
    print('  SUMMARY', flush=True)
    print('=' * 70, flush=True)
    for r in results:
        if r.get('status') != 'ok':
            print(f"  {r.get('strategy')}: SKIP ({r.get('reason')})", flush=True)
        else:
            print(f"  {r['strategy']}: {r['n_pass']}/5 PASS -> {r['deploy_band']}", flush=True)
    print(f"\n  Saved: {out_path}", flush=True)
    print(f"  Elapsed: {summary['elapsed_s']}s", flush=True)


if __name__ == '__main__':
    main()

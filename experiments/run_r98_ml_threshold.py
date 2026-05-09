#!/usr/bin/env python3
"""
R98 — ML Exit Threshold Optimization
======================================
Tests different ML probability thresholds for L8_MAX trade filtering.

Steps:
  1. Run L8_MAX backtest to get ~18,800 trades
  2. Compute H1 indicators and build features (same as R92-B)
  3. Walk-forward ML training to get OOS predictions for all trades
  4. Sweep thresholds [0.40 .. 0.65], compute metrics per threshold
  5. K-Fold validation on best 3 thresholds
  6. Sensitivity: Sharpe change per 0.01 threshold shift
  7. Save results/r98_ml_threshold/r98_results.json
"""
import sys, os, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("results/r98_ml_threshold")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100
SPREAD = 0.30
UNIT_LOT = 0.01

CAPS = {'L8_MAX': 35, 'PSAR': 5, 'TSMOM': 0, 'SESS_BO': 35}

THRESHOLDS = [0.40, 0.45, 0.48, 0.50, 0.52, 0.55, 0.60, 0.65]

FOLDS = [
    ("Fold1", "2015-01-01", "2017-01-01"),
    ("Fold2", "2017-01-01", "2019-01-01"),
    ("Fold3", "2019-01-01", "2021-01-01"),
    ("Fold4", "2021-01-01", "2023-01-01"),
    ("Fold5", "2023-01-01", "2025-01-01"),
    ("Fold6", "2025-01-01", "2026-05-01"),
]

ML_FEATURES = [
    'atr_14', 'adx_14', 'rsi_14', 'rsi_2',
    'kc_breakout_strength', 'volume_ratio', 'atr_percentile',
    'ema9_ema21_cross', 'close_ema100_dist',
    'hour_of_day', 'day_of_week', 'direction',
]


# ═══════════════════════════════════════════════════════════════
# Stats helpers
# ═══════════════════════════════════════════════════════════════

def _trades_to_daily(trades):
    if not trades:
        return np.array([0.0])
    daily = {}
    for t in trades:
        d = pd.Timestamp(t['exit_time']).date()
        daily[d] = daily.get(d, 0) + t['pnl']
    if not daily:
        return np.array([0.0])
    return np.array([daily[k] for k in sorted(daily.keys())])


def _sharpe(arr):
    if len(arr) < 10 or np.std(arr, ddof=1) == 0:
        return 0.0
    return float(np.mean(arr) / np.std(arr, ddof=1) * np.sqrt(252))


def _max_dd(arr):
    cum = np.cumsum(arr)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    return float(dd.max()) if len(dd) > 0 else 0.0


def _compute_stats(trades):
    if not trades:
        return {'n_trades': 0, 'sharpe': 0, 'max_dd': 0, 'pnl': 0, 'wr': 0, 'filter_rate': 0}
    daily = _trades_to_daily(trades)
    pnls = [t['pnl'] for t in trades]
    n = len(trades)
    return {
        'n_trades': n,
        'sharpe': round(_sharpe(daily), 3),
        'max_dd': round(_max_dd(daily), 2),
        'pnl': round(sum(pnls), 2),
        'wr': round(sum(1 for p in pnls if p > 0) / n * 100, 1),
    }


# ═══════════════════════════════════════════════════════════════
# L8_MAX backtest
# ═══════════════════════════════════════════════════════════════

def bt_l8_max(data_bundle, spread, lot, maxloss_cap=35):
    from backtest.runner import run_variant, LIVE_PARITY_KWARGS
    kw = {**LIVE_PARITY_KWARGS, 'maxloss_cap': maxloss_cap,
          'spread_cost': spread, 'initial_capital': 2000,
          'min_lot_size': lot, 'max_lot_size': lot}
    result = run_variant(data_bundle, "L8_MAX", verbose=False, **kw)
    raw_trades = result.get('_trades', [])
    trades = []
    for t in raw_trades:
        trades.append({
            'dir': t.direction, 'entry': t.entry_price, 'exit': t.exit_price,
            'entry_time': t.entry_time, 'exit_time': t.exit_time,
            'entry_bar': 0,
            'pnl': t.pnl, 'reason': t.exit_reason, 'bars': 0,
        })
    return trades


# ═══════════════════════════════════════════════════════════════
# ML model
# ═══════════════════════════════════════════════════════════════

def get_xgb_model(n_estimators=300, max_depth=5, learning_rate=0.05):
    try:
        import xgboost as xgb
        try:
            m = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth,
                learning_rate=learning_rate, subsample=0.8, colsample_bytree=0.8,
                eval_metric='logloss', tree_method='hist', device='cuda',
                random_state=42, verbosity=0)
            m.fit(np.random.randn(10, 5), np.random.randint(0, 2, 10))
            return xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth,
                learning_rate=learning_rate, subsample=0.8, colsample_bytree=0.8,
                eval_metric='logloss', tree_method='hist', device='cuda',
                random_state=42, verbosity=0)
        except Exception:
            return xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth,
                learning_rate=learning_rate, subsample=0.8, colsample_bytree=0.8,
                eval_metric='logloss', tree_method='hist', random_state=42, verbosity=0)
    except ImportError:
        return None


def walk_forward_oos_predictions(X, y, entry_times):
    """Walk-forward: train on all prior folds, predict current fold. Returns OOS probs for all trades."""
    et = entry_times.copy()
    if hasattr(et, 'dt') and et.dt.tz is not None:
        et = et.dt.tz_localize(None)
    elif hasattr(et, 'tz') and et.tz is not None:
        et = et.tz_localize(None)

    X_use = X[[c for c in ML_FEATURES if c in X.columns]].copy()
    oos_preds = np.full(len(y), np.nan)

    for _, fold_start, fold_end in FOLDS:
        fs = pd.Timestamp(fold_start)
        fe = pd.Timestamp(fold_end)
        train_mask = et < fs
        test_mask = (et >= fs) & (et < fe)
        if train_mask.sum() < 50 or test_mask.sum() < 20:
            continue

        Xtr = X_use[train_mask].copy()
        ytr = y[train_mask]
        Xte = X_use[test_mask].copy()

        med = Xtr.median()
        Xtr = Xtr.fillna(med)
        Xte = Xte.fillna(med)
        const = [c for c in Xtr.columns if Xtr[c].nunique() <= 1]
        if const:
            Xtr = Xtr.drop(columns=const)
            Xte = Xte.drop(columns=const)
        if len(Xtr.columns) == 0:
            continue

        try:
            model = get_xgb_model()
            if model is None:
                continue
            model.fit(Xtr, ytr)
            probs = model.predict_proba(Xte)[:, 1]
            oos_preds[np.where(test_mask)[0]] = probs
        except Exception:
            continue

    return oos_preds


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 80, flush=True)
    print("  R98 — ML Exit Threshold Optimization", flush=True)
    print("=" * 80, flush=True)

    # ── Step 1: Load data & run L8_MAX backtest ──
    print("\n  Loading data...", flush=True)
    from backtest.runner import load_m15, load_h1_aligned, H1_CSV_PATH, DataBundle

    m15_raw = load_m15()
    h1_df = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])
    print(f"    H1: {len(h1_df)} bars ({h1_df.index[0].date()} ~ {h1_df.index[-1].date()})")

    print("  Loading L8_MAX DataBundle...", flush=True)
    bundle = DataBundle.load_custom()

    print("  Running L8_MAX backtest...", flush=True)
    trades = bt_l8_max(bundle, SPREAD, UNIT_LOT, maxloss_cap=CAPS['L8_MAX'])
    base_stats = _compute_stats(trades)
    print(f"    {base_stats['n_trades']} trades, Sharpe={base_stats['sharpe']:.3f}, WR={base_stats['wr']:.1f}%")

    # ── Step 2: Compute H1 indicators & build features ──
    print("\n  Computing H1 indicators...", flush=True)
    from experiments.run_r92b_multi_strategy import compute_h1_indicators, build_features_for_trades
    h1_indicators = compute_h1_indicators(h1_df)

    ext_path = Path("data/external/aligned_daily.csv")
    external_daily = None
    if ext_path.exists():
        external_daily = pd.read_csv(ext_path, parse_dates=['Date'], index_col='Date')
        external_daily.index = external_daily.index.normalize()
        if external_daily.index.tz is not None:
            external_daily.index = external_daily.index.tz_localize(None)
        print(f"    External daily: {len(external_daily)} rows")

    print("  Building features...", flush=True)
    X, y, valid_indices = build_features_for_trades(trades, h1_indicators, external_daily)
    print(f"    Features: {X.shape[1]} cols, {len(X)} samples (win_rate={y.mean()*100:.1f}%)")

    entry_times = pd.Series([pd.Timestamp(trades[vi]['entry_time']) for vi in valid_indices])

    # ── Step 3: Walk-forward ML predictions ──
    print("\n  Walk-forward ML training (OOS predictions)...", flush=True)
    oos_preds = walk_forward_oos_predictions(X, y, entry_times)
    valid_oos = ~np.isnan(oos_preds)
    n_valid = int(valid_oos.sum())
    print(f"    OOS predictions for {n_valid}/{len(oos_preds)} trades")

    if n_valid < 100:
        print("  ERROR: Too few OOS predictions. Aborting.", flush=True)
        return

    # ── Step 4: Threshold sweep ──
    print("\n" + "=" * 70, flush=True)
    print("  Phase 1: Threshold Sweep", flush=True)
    print("=" * 70, flush=True)

    valid_trades = [trades[valid_indices[i]] for i in range(len(valid_indices)) if valid_oos[i]]
    valid_probs = oos_preds[valid_oos]
    base_valid_stats = _compute_stats(valid_trades)

    print(f"\n    Baseline (no filter): {base_valid_stats['n_trades']} trades, "
          f"Sharpe={base_valid_stats['sharpe']:.3f}, WR={base_valid_stats['wr']:.1f}%\n")

    threshold_results = {}
    print(f"    {'Threshold':>10}  {'N_trades':>8}  {'Filter%':>8}  {'Sharpe':>8}  {'MaxDD':>8}  "
          f"{'PnL':>10}  {'WR%':>6}", flush=True)
    print(f"    {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*6}")

    for thr in THRESHOLDS:
        filtered = [valid_trades[i] for i in range(len(valid_trades)) if valid_probs[i] >= thr]
        stats = _compute_stats(filtered)
        filter_rate = round((1 - len(filtered) / len(valid_trades)) * 100, 1) if valid_trades else 0
        stats['filter_rate'] = filter_rate
        stats['threshold'] = thr
        threshold_results[str(thr)] = stats

        print(f"    {thr:>10.2f}  {stats['n_trades']:>8}  {filter_rate:>7.1f}%  {stats['sharpe']:>8.3f}  "
              f"${stats['max_dd']:>7.1f}  ${stats['pnl']:>9.1f}  {stats['wr']:>5.1f}%", flush=True)

    # ── Step 5: K-Fold on best 3 thresholds ──
    print("\n" + "=" * 70, flush=True)
    print("  Phase 2: K-Fold Validation on Best 3 Thresholds", flush=True)
    print("=" * 70, flush=True)

    ranked = sorted(threshold_results.items(), key=lambda x: x[1]['sharpe'], reverse=True)
    best_3 = [float(k) for k, _ in ranked[:3]]
    print(f"    Best 3 thresholds: {best_3}")

    kfold_results = {}
    et_naive = entry_times.copy()
    if hasattr(et_naive, 'dt') and et_naive.dt.tz is not None:
        et_naive = et_naive.dt.tz_localize(None)
    elif hasattr(et_naive, 'tz') and et_naive.tz is not None:
        et_naive = et_naive.tz_localize(None)

    for thr in best_3:
        print(f"\n    Threshold={thr:.2f}:", flush=True)
        fold_sharpes = []

        for fold_name, fold_start, fold_end in FOLDS:
            fs = pd.Timestamp(fold_start)
            fe = pd.Timestamp(fold_end)

            fold_mask = (et_naive >= fs) & (et_naive < fe)
            fold_valid_mask = fold_mask & valid_oos
            if fold_valid_mask.sum() < 10:
                fold_sharpes.append(0.0)
                continue

            fold_trades_all = [trades[valid_indices[i]]
                               for i in range(len(valid_indices)) if fold_valid_mask.iloc[i]]
            fold_probs = oos_preds[np.where(fold_valid_mask)[0]]

            filtered = [fold_trades_all[j] for j in range(len(fold_trades_all))
                        if fold_probs[j] >= thr]
            daily = _trades_to_daily(filtered)
            fold_sharpes.append(_sharpe(daily))

        positive = sum(1 for s in fold_sharpes if s > 0)
        mean_sh = float(np.mean(fold_sharpes))
        kfold_results[str(thr)] = {
            'threshold': thr,
            'fold_sharpes': [round(s, 3) for s in fold_sharpes],
            'positive_folds': positive,
            'mean_sharpe': round(mean_sh, 3),
            'pass_4of6': positive >= 4,
        }
        print(f"      {positive}/6 positive, mean Sharpe={mean_sh:.3f}")
        print(f"      Folds: {[f'{s:.3f}' for s in fold_sharpes]}", flush=True)

    # ── Step 6: Sensitivity analysis ──
    print("\n" + "=" * 70, flush=True)
    print("  Phase 3: Sensitivity Analysis", flush=True)
    print("=" * 70, flush=True)

    sensitivity = {}
    fine_thresholds = np.arange(0.40, 0.66, 0.01)
    sharpe_curve = []

    for thr in fine_thresholds:
        filtered = [valid_trades[i] for i in range(len(valid_trades)) if valid_probs[i] >= thr]
        daily = _trades_to_daily(filtered)
        sh = _sharpe(daily)
        sharpe_curve.append({'threshold': round(float(thr), 2), 'sharpe': round(sh, 3),
                             'n_trades': len(filtered)})

    if len(sharpe_curve) >= 3:
        sharpes_arr = [s['sharpe'] for s in sharpe_curve]
        gradients = np.diff(sharpes_arr) / 0.01
        max_gradient_idx = int(np.argmax(np.abs(gradients)))
        max_gradient_thr = sharpe_curve[max_gradient_idx]['threshold']
        max_gradient_val = float(gradients[max_gradient_idx])

        peak_idx = int(np.argmax(sharpes_arr))
        peak_thr = sharpe_curve[peak_idx]['threshold']
        peak_sharpe = sharpes_arr[peak_idx]

        stable_zone_start = None
        stable_zone_end = None
        for i in range(len(gradients)):
            if abs(gradients[i]) < 0.05:
                if stable_zone_start is None:
                    stable_zone_start = sharpe_curve[i]['threshold']
                stable_zone_end = sharpe_curve[i + 1]['threshold']
            else:
                if stable_zone_start is not None:
                    break

        sensitivity = {
            'sharpe_curve': sharpe_curve,
            'peak_threshold': peak_thr,
            'peak_sharpe': round(peak_sharpe, 3),
            'max_gradient_threshold': max_gradient_thr,
            'max_gradient_per_001': round(max_gradient_val, 4),
            'stable_zone': [stable_zone_start, stable_zone_end] if stable_zone_start else None,
        }

        print(f"    Peak Sharpe: {peak_sharpe:.3f} at threshold={peak_thr:.2f}")
        print(f"    Max sensitivity: {max_gradient_val:+.4f} Sharpe per 0.01 at threshold={max_gradient_thr:.2f}")
        if stable_zone_start:
            print(f"    Stable zone: [{stable_zone_start:.2f}, {stable_zone_end:.2f}]")
    else:
        print("    Insufficient data for sensitivity analysis")

    # ── Summary & Save ──
    elapsed = time.time() - t0
    print(f"\n{'='*80}", flush=True)
    print(f"  R98 COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)

    best_overall = ranked[0]
    best_thr = float(best_overall[0])
    best_sh = best_overall[1]['sharpe']
    kf = kfold_results.get(str(best_thr), {})
    kf_pass = kf.get('pass_4of6', False)
    print(f"  Best threshold: {best_thr:.2f} (Sharpe={best_sh:.3f}, K-Fold {'PASS' if kf_pass else 'FAIL'})")
    print(f"  Baseline Sharpe: {base_valid_stats['sharpe']:.3f}")
    print(f"{'='*80}", flush=True)

    output = {
        'experiment': 'R98 ML Exit Threshold Optimization',
        'elapsed_s': round(elapsed, 1),
        'baseline': {
            'n_trades': base_valid_stats['n_trades'],
            'sharpe': base_valid_stats['sharpe'],
            'wr': base_valid_stats['wr'],
        },
        'threshold_sweep': threshold_results,
        'kfold_validation': kfold_results,
        'sensitivity': sensitivity,
        'recommendation': {
            'best_threshold': best_thr,
            'best_sharpe': best_sh,
            'kfold_pass': kf_pass,
            'verdict': f"Use threshold={best_thr:.2f}" if kf_pass else "Keep default 0.50 (best failed K-Fold)",
        },
    }
    with open(OUTPUT_DIR / "r98_results.json", 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Saved: {OUTPUT_DIR}/r98_results.json", flush=True)


if __name__ == "__main__":
    main()

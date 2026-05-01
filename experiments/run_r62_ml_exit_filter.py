#!/usr/bin/env python3
"""
R62 — ML Exit Quality Filter (XGBoost/LightGBM Walk-Forward)
==============================================================
Train an ML model to predict at entry time whether a Keltner trade will be
profitable.  This is NOT for replacing entry signals (already saturated per
core insight #6), but for scoring trade quality to potentially adjust exit
parameters (e.g. MaxHold).

Pipeline:
  1. Run full Keltner backtest → get all trades with entry_time / pnl
  2. Compute 12 features from H1 data at each trade's entry bar
  3. Binary label: y = 1 if pnl > 0 else 0
  4. Walk-Forward: train 2015–2021, test 2022–2026-04
  5. XGBoost + LightGBM → AUC-ROC, calibration, feature importance
  6. Per-quintile analysis: avg PnL & win rate per predicted-prob quintile
  7. If AUC > 0.60: conditional exit simulation (wider/tighter MaxHold)
"""

# ---------------------------------------------------------------------------
# Dependency bootstrap
# ---------------------------------------------------------------------------
try:
    import xgboost as xgb
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'xgboost', 'lightgbm', 'scikit-learn'])
    import xgboost as xgb

try:
    import lightgbm as lgb
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'lightgbm'])
    import lightgbm as lgb

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'scikit-learn'])
    from sklearn.metrics import roc_auc_score

import sys, os, io, time, json, multiprocessing as mp
import numpy as np
import pandas as pd
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

OUTPUT_DIR = Path("results/r62_ml_exit_filter")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_WORKERS = 6
FIXED_LOT = 0.05
SPREAD = 0.30

FOLDS = [
    ("Fold1", "2015-01-01", "2017-01-01"),
    ("Fold2", "2017-01-01", "2019-01-01"),
    ("Fold3", "2019-01-01", "2021-01-01"),
    ("Fold4", "2021-01-01", "2023-01-01"),
    ("Fold5", "2023-01-01", "2025-01-01"),
    ("Fold6", "2025-01-01", "2026-05-01"),
]

TRAIN_END = "2021-12-31"
TEST_START = "2022-01-01"

FEATURE_NAMES = [
    'atr_14', 'adx_14', 'rsi_14', 'rsi_2',
    'kc_breakout_strength', 'volume_ratio', 'atr_percentile',
    'ema9_ema21_cross', 'close_ema100_dist',
    'hour_of_day', 'day_of_week', 'direction',
]

def fmt(v):
    return f"${v:>10,.0f}" if v >= 0 else f"-${abs(v):>9,.0f}"


def save_text(filename, text):
    path = OUTPUT_DIR / filename
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"  [Saved] {path}", flush=True)


def save_checkpoint(data, filename):
    path = OUTPUT_DIR / filename
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, default=str)
    print(f"  [Checkpoint] {path}", flush=True)


def load_checkpoint(filename):
    path = OUTPUT_DIR / filename
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


# ═══════════════════════════════════════════════════════════════
# H1 feature computation helpers
# ═══════════════════════════════════════════════════════════════

def _compute_rsi(close, period):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _compute_atr(high, low, close, period):
    tr = pd.DataFrame({
        'hl': high - low,
        'hc': (high - close.shift(1)).abs(),
        'lc': (low - close.shift(1)).abs(),
    }).max(axis=1)
    return tr.rolling(period).mean()


def _compute_adx(df, period=14):
    high, low, close = df['High'], df['Low'], df['Close']
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr = pd.DataFrame({'hl': high - low, 'hc': (high - close.shift(1)).abs(),
                        'lc': (low - close.shift(1)).abs()}).max(axis=1)
    atr_s = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr_s)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr_s)
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    return dx.rolling(period).mean()


def prepare_h1_features(h1_df):
    """Pre-compute all feature columns on H1 DataFrame."""
    df = h1_df.copy()
    close = df['Close']; high = df['High']; low = df['Low']
    vol = df['Volume'] if 'Volume' in df.columns else pd.Series(0, index=df.index)

    df['atr_14'] = _compute_atr(high, low, close, 14)
    df['adx_14'] = _compute_adx(df, 14)
    df['rsi_14'] = _compute_rsi(close, 14)
    df['rsi_2'] = _compute_rsi(close, 2)

    ema_kc = close.ewm(span=25, adjust=False).mean()
    atr_kc = df['atr_14']
    kc_upper = ema_kc + 1.2 * atr_kc
    df['kc_breakout_strength'] = (close - kc_upper) / atr_kc.replace(0, np.nan)

    vol_ma20 = vol.rolling(20).mean()
    df['volume_ratio'] = vol / vol_ma20.replace(0, np.nan)

    df['atr_percentile'] = df['atr_14'].rolling(100, min_periods=20).rank(pct=True)

    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    df['ema9_ema21_cross'] = (ema9 - ema21) / atr_kc.replace(0, np.nan)

    ema100 = close.ewm(span=100, adjust=False).mean()
    df['close_ema100_dist'] = (close - ema100) / atr_kc.replace(0, np.nan)

    df['hour_of_day'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek

    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def extract_features_for_trade(h1_feat, entry_time, direction_str):
    """Look up feature values from H1 DataFrame at the bar <= entry_time."""
    h1_time = pd.Timestamp(entry_time).floor('h')
    mask = h1_feat.index <= h1_time
    if mask.sum() == 0:
        return None
    idx = h1_feat.index[mask][-1]
    row = h1_feat.loc[idx]

    dir_val = 1 if direction_str in ('BUY', 'LONG', 1) else -1
    feats = {}
    for f in FEATURE_NAMES:
        if f == 'direction':
            feats[f] = dir_val
        elif f in row.index:
            feats[f] = float(row[f]) if not pd.isna(row[f]) else np.nan
        else:
            feats[f] = np.nan
    return feats


# ═══════════════════════════════════════════════════════════════
# Step 1: Run full Keltner backtest → trades
# ═══════════════════════════════════════════════════════════════

def get_keltner_trades(m15_df, h1_df):
    print("  [Step 1] Running full Keltner backtest...", flush=True)
    from backtest.runner import DataBundle, run_variant, LIVE_PARITY_KWARGS
    kw = {
        **LIVE_PARITY_KWARGS,
        'maxloss_cap': 37,
        'spread_cost': SPREAD, 'initial_capital': 2000,
        'min_lot_size': 0.03, 'max_lot_size': 0.03,
    }
    data = DataBundle(m15_df, h1_df)
    result = run_variant(data, "R62_BASE", verbose=True, **kw)
    raw = result.get('_trades', [])

    trades = []
    for t in raw:
        entry_time = t.entry_time if hasattr(t, 'entry_time') else t.get('entry_time', '')
        exit_time = t.exit_time if hasattr(t, 'exit_time') else t.get('exit_time', '')
        pnl = t.pnl if hasattr(t, 'pnl') else t.get('pnl', 0)
        direction = t.direction if hasattr(t, 'direction') else t.get('direction', '')
        bars_held = t.bars_held if hasattr(t, 'bars_held') else t.get('bars_held', 0)
        exit_reason = t.exit_reason if hasattr(t, 'exit_reason') else t.get('exit_reason', '')
        trades.append({
            'entry_time': str(entry_time),
            'exit_time': str(exit_time),
            'pnl': float(pnl),
            'direction': str(direction),
            'bars_held': int(bars_held),
            'exit_reason': str(exit_reason),
        })

    print(f"    Total trades: {len(trades)}", flush=True)
    print(f"    Win rate: {sum(1 for t in trades if t['pnl'] > 0)/len(trades)*100:.1f}%", flush=True)
    print(f"    Total PnL: {fmt(sum(t['pnl'] for t in trades))}", flush=True)
    return trades


# ═══════════════════════════════════════════════════════════════
# Step 2: Build feature matrix
# ═══════════════════════════════════════════════════════════════

def build_feature_matrix(trades, h1_feat):
    print("  [Step 2] Building feature matrix...", flush=True)
    rows = []
    skipped = 0
    for t in trades:
        feats = extract_features_for_trade(h1_feat, t['entry_time'], t['direction'])
        if feats is None:
            skipped += 1
            continue
        feats['pnl'] = t['pnl']
        feats['y'] = 1 if t['pnl'] > 0 else 0
        feats['entry_time'] = t['entry_time']
        feats['exit_time'] = t.get('exit_time', t['entry_time'])
        feats['bars_held'] = t['bars_held']
        feats['exit_reason'] = t['exit_reason']
        rows.append(feats)

    df = pd.DataFrame(rows)
    if skipped > 0:
        print(f"    Skipped {skipped} trades (no H1 data match)", flush=True)
    print(f"    Feature matrix: {len(df)} trades x {len(FEATURE_NAMES)} features", flush=True)
    print(f"    NaN counts:", flush=True)
    for f in FEATURE_NAMES:
        nan_ct = df[f].isna().sum()
        if nan_ct > 0:
            print(f"      {f}: {nan_ct}", flush=True)

    df[FEATURE_NAMES] = df[FEATURE_NAMES].fillna(df[FEATURE_NAMES].median())
    return df


# ═══════════════════════════════════════════════════════════════
# Step 3: Walk-Forward Split
# ═══════════════════════════════════════════════════════════════

def split_train_test(df):
    df['entry_ts'] = pd.to_datetime(df['entry_time'])
    train = df[df['entry_ts'] <= TRAIN_END].copy()
    test = df[df['entry_ts'] >= TEST_START].copy()
    print(f"  [Step 3] Walk-Forward Split:", flush=True)
    print(f"    Train: {len(train)} trades (... to {TRAIN_END}), WR={train['y'].mean()*100:.1f}%", flush=True)
    print(f"    Test:  {len(test)} trades ({TEST_START} to ...), WR={test['y'].mean()*100:.1f}%", flush=True)
    return train, test


# ═══════════════════════════════════════════════════════════════
# Step 4-5: Train & Evaluate Models
# ═══════════════════════════════════════════════════════════════

def train_and_evaluate(train_df, test_df):
    print("\n  [Step 4-5] Training & Evaluating Models...", flush=True)

    X_train = train_df[FEATURE_NAMES].values
    y_train = train_df['y'].values
    X_test = test_df[FEATURE_NAMES].values
    y_test = test_df['y'].values

    results = {}
    models = {}

    # XGBoost
    print("    Training XGBoost...", flush=True)
    xgb_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric='logloss',
        random_state=42,
    )
    xgb_model.fit(X_train, y_train, verbose=False)
    xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
    xgb_auc = roc_auc_score(y_test, xgb_prob)
    print(f"      XGBoost AUC: {xgb_auc:.4f}", flush=True)
    results['xgb'] = {'auc': round(xgb_auc, 4), 'model_name': 'XGBoost'}
    models['xgb'] = (xgb_model, xgb_prob)

    # LightGBM
    print("    Training LightGBM...", flush=True)
    lgb_model = lgb.LGBMClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbose=-1,
    )
    lgb_model.fit(X_train, y_train)
    lgb_prob = lgb_model.predict_proba(X_test)[:, 1]
    lgb_auc = roc_auc_score(y_test, lgb_prob)
    print(f"      LightGBM AUC: {lgb_auc:.4f}", flush=True)
    results['lgb'] = {'auc': round(lgb_auc, 4), 'model_name': 'LightGBM'}
    models['lgb'] = (lgb_model, lgb_prob)

    # Pick best
    best_key = 'xgb' if xgb_auc >= lgb_auc else 'lgb'
    best_model, best_prob = models[best_key]
    best_auc = results[best_key]['auc']
    print(f"    Best model: {results[best_key]['model_name']} (AUC={best_auc})", flush=True)

    return results, best_key, best_model, best_prob, best_auc


# ═══════════════════════════════════════════════════════════════
# Step 6: Feature importance
# ═══════════════════════════════════════════════════════════════

def report_feature_importance(best_model, best_key):
    print("\n  [Step 6] Feature Importance:", flush=True)
    if best_key == 'xgb':
        importances = best_model.feature_importances_
    else:
        importances = best_model.feature_importances_

    pairs = sorted(zip(FEATURE_NAMES, importances), key=lambda x: -x[1])
    lines = ["Feature Importance (best model)\n"]
    lines.append(f"{'Rank':>4} {'Feature':>25} {'Importance':>12} {'%':>8}")
    lines.append("-" * 55)
    total_imp = sum(importances)
    for i, (fname, imp) in enumerate(pairs, 1):
        pct = imp / total_imp * 100 if total_imp > 0 else 0
        lines.append(f"{i:>4} {fname:>25} {imp:>12.4f} {pct:>7.1f}%")
        print(f"    {i:>2}. {fname:>25}: {pct:.1f}%", flush=True)

    report = "\n".join(lines)
    save_text("feature_importance.txt", report)
    return pairs


# ═══════════════════════════════════════════════════════════════
# Step 7: Calibration — predicted prob vs actual win rate
# ═══════════════════════════════════════════════════════════════

def report_calibration(test_df, best_prob):
    print("\n  [Step 7] Calibration (10 deciles):", flush=True)
    df = test_df.copy()
    df['pred_prob'] = best_prob

    df['decile'] = pd.qcut(df['pred_prob'], 10, labels=False, duplicates='drop')
    lines = ["Calibration: Predicted Probability vs Actual Win Rate\n"]
    lines.append(f"{'Decile':>7} {'PredProb':>12} {'ActualWR':>10} {'N':>6} {'AvgPnL':>12}")
    lines.append("-" * 55)
    for d in sorted(df['decile'].unique()):
        grp = df[df['decile'] == d]
        avg_pred = grp['pred_prob'].mean()
        actual_wr = grp['y'].mean()
        avg_pnl = grp['pnl'].mean()
        lines.append(f"{d:>7} {avg_pred:>12.4f} {actual_wr:>10.3f} {len(grp):>6} {fmt(avg_pnl):>12}")
        print(f"    D{d}: pred={avg_pred:.3f} actual_wr={actual_wr:.3f} "
              f"N={len(grp)} avg_pnl={fmt(avg_pnl)}", flush=True)

    report = "\n".join(lines)
    save_text("calibration.txt", report)
    return df


# ═══════════════════════════════════════════════════════════════
# Step 8: Quintile analysis
# ═══════════════════════════════════════════════════════════════

def report_quintile_analysis(test_df, best_prob):
    print("\n  [Step 8] Per-Quintile Analysis:", flush=True)
    df = test_df.copy()
    df['pred_prob'] = best_prob

    df['quintile'] = pd.qcut(df['pred_prob'], 5, labels=['Q1(Low)', 'Q2', 'Q3', 'Q4', 'Q5(High)'],
                             duplicates='drop')
    lines = ["Per-Quintile Analysis: Predicted Probability → PnL & WR\n"]
    lines.append(f"{'Quintile':>10} {'N':>6} {'WinRate':>8} {'AvgPnL':>12} {'TotalPnL':>12} {'AvgProb':>10}")
    lines.append("-" * 65)
    monotonic = True
    prev_wr = None
    for q in ['Q1(Low)', 'Q2', 'Q3', 'Q4', 'Q5(High)']:
        if q not in df['quintile'].values:
            continue
        grp = df[df['quintile'] == q]
        wr = grp['y'].mean()
        avg_pnl = grp['pnl'].mean()
        total_pnl = grp['pnl'].sum()
        avg_prob = grp['pred_prob'].mean()
        lines.append(f"{q:>10} {len(grp):>6} {wr:>8.3f} {fmt(avg_pnl):>12} "
                     f"{fmt(total_pnl):>12} {avg_prob:>10.3f}")
        print(f"    {q}: N={len(grp)}, WR={wr:.3f}, AvgPnL={fmt(avg_pnl)}, "
              f"TotalPnL={fmt(total_pnl)}", flush=True)
        if prev_wr is not None and wr < prev_wr - 0.02:
            monotonic = False
        prev_wr = wr

    if monotonic:
        lines.append("\n*** MONOTONIC RELATIONSHIP DETECTED — model has predictive value for exit tuning ***")
        print("    >>> MONOTONIC: higher predicted prob → higher win rate", flush=True)
    else:
        lines.append("\n*** No clear monotonic relationship — model may have limited value ***")
        print("    >>> NON-MONOTONIC: relationship not clean", flush=True)

    report = "\n".join(lines)
    save_text("quintile_analysis.txt", report)
    return monotonic


# ═══════════════════════════════════════════════════════════════
# Step 9: Conditional exit simulation (if AUC > 0.60)
# ═══════════════════════════════════════════════════════════════

def conditional_exit_simulation(test_df, best_prob, best_auc):
    if best_auc <= 0.60:
        print(f"\n  [Step 9] SKIPPED — AUC {best_auc:.4f} <= 0.60 threshold", flush=True)
        save_text("conditional_exit_sim.txt",
                  f"SKIPPED: AUC={best_auc:.4f} <= 0.60, insufficient predictive power.")
        return

    print(f"\n  [Step 9] Conditional Exit Simulation (AUC={best_auc:.4f} > 0.60):", flush=True)
    df = test_df.copy()
    df['pred_prob'] = best_prob

    baseline_pnl = df['pnl'].values
    baseline_total = float(baseline_pnl.sum())

    daily_base = {}
    for _, row in df.iterrows():
        d = pd.Timestamp(row['exit_time']).date()
        daily_base[d] = daily_base.get(d, 0) + row['pnl']
    base_arr = np.array(list(daily_base.values()))
    baseline_sharpe = float(base_arr.mean() / base_arr.std() * np.sqrt(252)) if base_arr.std() > 0 else 0

    # Simulate: high-quality trades get wider hold (factor 1.4 PnL proxy),
    #           low-quality trades get tighter hold (factor 0.6 PnL proxy)
    # This is a rough approximation: wider MaxHold ≈ trades that would timeout
    # get more time to hit trailing, so winning trades improve marginally.
    adj_pnl = df['pnl'].copy()
    high_q = df['pred_prob'] > 0.7
    low_q = df['pred_prob'] < 0.3

    timeout_mask = df['exit_reason'] == 'Timeout'
    adj_pnl.loc[high_q & timeout_mask & (df['pnl'] > 0)] *= 1.2
    adj_pnl.loc[high_q & timeout_mask & (df['pnl'] < 0)] *= 0.8
    adj_pnl.loc[low_q & timeout_mask & (df['pnl'] < 0)] *= 0.7
    adj_pnl.loc[low_q & timeout_mask & (df['pnl'] > 0)] *= 0.9

    adj_total = float(adj_pnl.sum())

    daily_adj = {}
    for i, (_, row) in enumerate(df.iterrows()):
        d = pd.Timestamp(row['exit_time']).date()
        daily_adj[d] = daily_adj.get(d, 0) + adj_pnl.iloc[i]
    adj_arr = np.array(list(daily_adj.values()))
    adj_sharpe = float(adj_arr.mean() / adj_arr.std() * np.sqrt(252)) if adj_arr.std() > 0 else 0

    lines = [f"Conditional Exit Simulation (AUC={best_auc:.4f})\n"]
    lines.append(f"High-quality trades (prob > 0.7): {high_q.sum()} ({high_q.mean()*100:.1f}%)")
    lines.append(f"Low-quality trades (prob < 0.3):  {low_q.sum()} ({low_q.mean()*100:.1f}%)")
    lines.append(f"Timeout trades in test set: {timeout_mask.sum()}")
    lines.append(f"\n{'Metric':>20} {'Baseline':>12} {'Adjusted':>12} {'Delta':>12}")
    lines.append("-" * 60)
    lines.append(f"{'Total PnL':>20} {fmt(baseline_total):>12} {fmt(adj_total):>12} "
                 f"{fmt(adj_total - baseline_total):>12}")
    lines.append(f"{'Sharpe':>20} {baseline_sharpe:>12.3f} {adj_sharpe:>12.3f} "
                 f"{adj_sharpe - baseline_sharpe:>12.3f}")
    lines.append(f"\nNote: PnL adjustment is a rough proxy. Actual improvement requires "
                 f"re-running backtest with modified MaxHold per trade.")

    if adj_sharpe > baseline_sharpe:
        lines.append(f"\n*** POSITIVE SIGNAL: Conditional exit improves Sharpe by "
                     f"{adj_sharpe - baseline_sharpe:.3f} ***")
        print(f"    >>> POSITIVE: Sharpe {baseline_sharpe:.3f} → {adj_sharpe:.3f} "
              f"(+{adj_sharpe - baseline_sharpe:.3f})", flush=True)
    else:
        lines.append(f"\n*** NO IMPROVEMENT: Conditional exit does not help ***")
        print(f"    >>> NO IMPROVEMENT: Sharpe {baseline_sharpe:.3f} → {adj_sharpe:.3f}", flush=True)

    report = "\n".join(lines)
    print(report, flush=True)
    save_text("conditional_exit_sim.txt", report)


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 80)
    print("  R62: ML Exit Quality Filter (XGBoost/LightGBM Walk-Forward)")
    print("=" * 80)

    from backtest.runner import DataBundle
    data = DataBundle.load_default()
    h1_df = data.h1_df.copy()
    m15_df = data.m15_df.copy()
    print(f"  H1: {len(h1_df)} bars | M15: {len(m15_df)} bars")

    # Step 1: Get trades
    trades = get_keltner_trades(m15_df, h1_df)

    # Step 2: H1 features + matrix
    print("\n  Preparing H1 feature columns...", flush=True)
    h1_feat = prepare_h1_features(h1_df)
    feat_df = build_feature_matrix(trades, h1_feat)
    if len(feat_df) < 50:
        print("  ABORT: Too few trades with valid features", flush=True)
        return

    # Step 3: Split
    train_df, test_df = split_train_test(feat_df)
    if len(train_df) < 30 or len(test_df) < 30:
        print("  ABORT: Insufficient train/test samples", flush=True)
        return

    # Step 4-5: Train & evaluate
    model_results, best_key, best_model, best_prob, best_auc = train_and_evaluate(train_df, test_df)

    # Step 6: Feature importance
    fi_pairs = report_feature_importance(best_model, best_key)

    # Step 7: Calibration
    cal_df = report_calibration(test_df, best_prob)

    # Step 8: Quintile
    is_monotonic = report_quintile_analysis(test_df, best_prob)

    # Step 9: Conditional exit (if AUC > 0.60)
    conditional_exit_simulation(test_df, best_prob, best_auc)

    # Summary
    elapsed = time.time() - t0
    lines = [f"R62 SUMMARY (elapsed {elapsed/60:.1f} min)\n"]
    lines.append(f"Total trades: {len(feat_df)}")
    lines.append(f"Train: {len(train_df)} | Test: {len(test_df)}")
    lines.append(f"\nModel AUCs:")
    for k, v in model_results.items():
        lines.append(f"  {v['model_name']}: {v['auc']:.4f}")
    lines.append(f"\nBest: {model_results[best_key]['model_name']} (AUC={best_auc:.4f})")
    lines.append(f"Monotonic quintile: {'YES' if is_monotonic else 'NO'}")
    lines.append(f"\nTop 5 features:")
    for i, (fname, imp) in enumerate(fi_pairs[:5], 1):
        total_imp = sum(x[1] for x in fi_pairs)
        lines.append(f"  {i}. {fname}: {imp/total_imp*100:.1f}%")

    if best_auc > 0.60:
        lines.append(f"\nConditional exit simulation: see conditional_exit_sim.txt")
    else:
        lines.append(f"\nConditional exit: SKIPPED (AUC {best_auc:.4f} <= 0.60)")

    report = "\n".join(lines)
    print(f"\n{'='*80}")
    print(report)
    print(f"{'='*80}")
    save_text("r62_summary.txt", report)

    save_checkpoint({
        'model_results': model_results,
        'best_model': best_key,
        'best_auc': best_auc,
        'n_train': len(train_df),
        'n_test': len(test_df),
        'is_monotonic': is_monotonic,
        'feature_importance': {fname: float(imp) for fname, imp in fi_pairs},
    }, "r62_results.json")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R145 — XGBoost Entry Signal Refinement
========================================
R137 showed AUC=0.74 but threshold 0.6 was too strict, triggering very few
signals.  This experiment sweeps thresholds, integrates ML as a per-strategy
filter, evaluates walk-forward PnL, checks feature-importance stability, and
measures AUC degradation over time.

Phases:
  1. Load data & build 19-feature matrix (same as R137)
  2. Threshold sweep (0.50 → 0.65) with signal count, AUC, backtest PnL
  3. Strategy-specific ML filter (PSAR / TSMOM / SESS_BO)
  4. Walk-Forward PnL (6mo train / 2mo test, sliding 2mo, 2019-2026)
  5. Feature importance stability (Jaccard across folds)
  6. AUC degradation by year (concept drift)
  7. K-Fold 5-fold of best combo
"""
import sys, os, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

import xgboost as xgb
from backtest.runner import load_csv

OUTPUT_DIR = Path("results/r145_xgb_refinement")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100; SPREAD = 0.30; UNIT_LOT = 0.01
CAPS = {'PSAR': 5, 'TSMOM': 0, 'SESS_BO': 35}
FORWARD_BARS = 4
N_LAGS = 8

ALIGNED_CSV = Path("data/external/aligned_daily.csv")
H1_CANDIDATES = [
    Path("data/download/xauusd-h1-bid-2015-01-01-2026-04-27.csv"),
    Path("data/download/xauusd-h1-bid-2015-01-01-2026-04-10.csv"),
    Path("data/download/xauusd-h1-bid-2015-01-01-2026-03-25.csv"),
]

THRESHOLDS = [0.50, 0.52, 0.55, 0.58, 0.60, 0.65]

XGB_PARAMS = dict(
    n_estimators=200, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric='auc', use_label_encoder=False, verbosity=0,
)

CONTINUOUS_FEATURES = [
    'ret_1', 'ret_4', 'ret_8', 'ret_24', 'ATR14', 'RSI14', 'RSI2',
    'KC_pos', 'MACD_hist', 'volume', 'EMA20_trend', 'daily_range',
    'daily_mom', 'VIX', 'DXY', 'US10Y', 'real_yield',
]
CATEGORICAL_FEATURES = ['hour', 'dow']
ALL_FEATURES = CONTINUOUS_FEATURES + CATEGORICAL_FEATURES

t0 = time.time()


# ═══════════════════════════════════════════════════════════════
# Shared helpers — indicators
# ═══════════════════════════════════════════════════════════════

def compute_atr(df, period=14):
    h, l, c = df['High'], df['Low'], df['Close']
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_kc_position(close, period=25, atr_mult=1.2, atr_period=14):
    kc_mid = close.ewm(span=period).mean()
    atr = (close.rolling(atr_period).max() - close.rolling(atr_period).min()).rolling(atr_period).mean()
    kc_upper = kc_mid + atr_mult * atr
    kc_lower = kc_mid - atr_mult * atr
    width = kc_upper - kc_lower
    pos = (close - kc_lower) / width.replace(0, np.nan)
    return pos.fillna(0.5)


def add_psar(df, af_step=0.01, af_max=0.05):
    h, l, c = df['High'].values, df['Low'].values, df['Close'].values
    n = len(df)
    psar = np.empty(n); psar[:] = np.nan
    af = af_step; rising = True; ep = h[0]; psar[0] = l[0]
    for i in range(1, n):
        prev = psar[i-1]
        if rising:
            psar[i] = prev + af * (ep - prev)
            psar[i] = min(psar[i], l[i-1], l[max(0, i-2)])
            if l[i] < psar[i]:
                rising = False; psar[i] = ep; ep = l[i]; af = af_step
            else:
                if h[i] > ep: ep = h[i]; af = min(af + af_step, af_max)
        else:
            psar[i] = prev + af * (ep - prev)
            psar[i] = max(psar[i], h[i-1], h[max(0, i-2)])
            if h[i] > psar[i]:
                rising = True; psar[i] = ep; ep = h[i]; af = af_step
            else:
                if l[i] < ep: ep = l[i]; af = min(af + af_step, af_max)
    df['PSAR'] = psar


# ═══════════════════════════════════════════════════════════════
# Shared helpers — trade execution
# ═══════════════════════════════════════════════════════════════

def _mk(pos, exit_px, exit_time, reason, bar_i, pnl):
    return {'dir': pos['dir'], 'entry': pos['entry'], 'exit': exit_px,
            'entry_time': pos['time'], 'exit_time': exit_time,
            'pnl': pnl, 'reason': reason, 'bars': bar_i - pos['bar']}


def _run_exit(pos, i, hi, lo, cl, spread, lot, pv, times,
              sl_atr, tp_atr, trail_act, trail_dist, max_hold, cap):
    atr = pos['atr']
    sl = atr * sl_atr; tp = atr * tp_atr; bars = i - pos['bar']
    if pos['dir'] == 'BUY':
        pnl_now = (cl - pos['entry'] - spread) * lot * pv
        if cap > 0 and pnl_now < -cap:
            return _mk(pos, cl, times[i], "MaxLossCap", i, -cap)
        if lo <= pos['entry'] - sl:
            return _mk(pos, pos['entry'] - sl, times[i], "SL", i, -sl * lot * pv)
        if hi >= pos['entry'] + tp:
            return _mk(pos, pos['entry'] + tp, times[i], "TP", i, tp * lot * pv)
        extreme = pos.get('extreme', pos['entry']); extreme = max(extreme, hi); pos['extreme'] = extreme
        if extreme - pos['entry'] >= atr * trail_act:
            trail_price = extreme - atr * trail_dist
            if cl <= trail_price:
                return _mk(pos, trail_price, times[i], "Trail", i, (trail_price - pos['entry'] - spread) * lot * pv)
        if bars >= max_hold:
            return _mk(pos, cl, times[i], "TimeExit", i, pnl_now)
    else:
        pnl_now = (pos['entry'] - cl - spread) * lot * pv
        if cap > 0 and pnl_now < -cap:
            return _mk(pos, cl, times[i], "MaxLossCap", i, -cap)
        if hi >= pos['entry'] + sl:
            return _mk(pos, pos['entry'] + sl, times[i], "SL", i, -sl * lot * pv)
        if lo <= pos['entry'] - tp:
            return _mk(pos, pos['entry'] - tp, times[i], "TP", i, tp * lot * pv)
        extreme = pos.get('extreme', pos['entry']); extreme = min(extreme, lo); pos['extreme'] = extreme
        if pos['entry'] - extreme >= atr * trail_act:
            trail_price = extreme + atr * trail_dist
            if cl >= trail_price:
                return _mk(pos, trail_price, times[i], "Trail", i, (pos['entry'] - trail_price - spread) * lot * pv)
        if bars >= max_hold:
            return _mk(pos, cl, times[i], "TimeExit", i, pnl_now)
    return None


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
    eq = np.cumsum(arr)
    return float((np.maximum.accumulate(eq) - eq).max()) if len(eq) > 1 else 0.0


def _compute_stats(trades):
    if not trades:
        return {'n': 0, 'sharpe': 0, 'pnl': 0, 'wr': 0, 'max_dd': 0}
    daily = _trades_to_daily(trades)
    pnls = [t['pnl'] for t in trades]
    n = len(trades)
    return {
        'n': n,
        'sharpe': round(_sharpe(daily), 3),
        'pnl': round(sum(pnls), 2),
        'wr': round(sum(1 for p in pnls if p > 0) / n * 100, 1),
        'max_dd': round(_max_dd(daily), 2),
    }


# ═══════════════════════════════════════════════════════════════
# Strategy backtests
# ═══════════════════════════════════════════════════════════════

def bt_psar(h1_df, spread, lot, cap, params=None):
    if params is None:
        params = {'sl_atr': 4.5, 'tp_atr': 16.0, 'trail_act': 0.20, 'trail_dist': 0.04, 'max_hold': 20}
    df = h1_df.copy(); add_psar(df); df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR', 'PSAR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; psar = df['PSAR'].values; times = df.index; n = len(df)
    trades = []; pos = None; last_exit = -999
    for i in range(1, n):
        if pos is not None:
            result = _run_exit(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                               params['sl_atr'], params['tp_atr'], params['trail_act'],
                               params['trail_dist'], params['max_hold'], cap)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if i - last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        if psar[i-1] > c[i-1] and psar[i] < c[i]:
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif psar[i-1] < c[i-1] and psar[i] > c[i]:
            pos = {'dir': 'SELL', 'entry': c[i]-spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_tsmom(h1_df, spread, lot, cap,
             fast=480, slow=720, sl_atr=4.5, tp_atr=6.0,
             trail_act=0.14, trail_dist=0.025, max_hold=20):
    df = h1_df.copy(); df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; times = df.index; n = len(df)
    max_lb = max(fast, slow)
    score = np.full(n, np.nan)
    for i in range(max_lb, n):
        s = 0.0
        if c[i-fast] > 0: s += 0.5 * np.sign(c[i]/c[i-fast] - 1.0)
        if c[i-slow] > 0: s += 0.5 * np.sign(c[i]/c[i-slow] - 1.0)
        score[i] = s
    trades = []; pos = None; last_exit = -999
    for i in range(max_lb+1, n):
        if pos is not None:
            result = _run_exit(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                               sl_atr, tp_atr, trail_act, trail_dist, max_hold, cap)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            if pos['dir'] == 'BUY' and score[i] < 0:
                pnl = (c[i]-pos['entry']-spread)*lot*PV
                trades.append(_mk(pos, c[i], times[i], "Reversal", i, pnl)); pos = None; last_exit = i; continue
            elif pos['dir'] == 'SELL' and score[i] > 0:
                pnl = (pos['entry']-c[i]-spread)*lot*PV
                trades.append(_mk(pos, c[i], times[i], "Reversal", i, pnl)); pos = None; last_exit = i; continue
            continue
        if i-last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        if np.isnan(score[i]) or np.isnan(score[i-1]): continue
        if score[i] > 0 and score[i-1] <= 0:
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif score[i] < 0 and score[i-1] >= 0:
            pos = {'dir': 'SELL', 'entry': c[i]-spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_sess_bo(h1_df, spread, lot, cap,
               session_hour=12, lookback=4, sl_atr=4.5, tp_atr=4.0,
               trail_act=0.14, trail_dist=0.025, max_hold=20):
    df = h1_df.copy(); df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; hours = df.index.hour; times = df.index; n = len(df)
    trades = []; pos = None; last_exit = -999
    for i in range(lookback, n):
        if pos is not None:
            result = _run_exit(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                               sl_atr, tp_atr, trail_act, trail_dist, max_hold, cap)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if hours[i] != session_hour: continue
        if i - last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        hh = max(h[i-j] for j in range(1, lookback+1))
        ll = min(lo[i-j] for j in range(1, lookback+1))
        if c[i] > hh:
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif c[i] < ll:
            pos = {'dir': 'SELL', 'entry': c[i]-spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


# ═══════════════════════════════════════════════════════════════
# XGBoost helpers
# ═══════════════════════════════════════════════════════════════

def _add_lag_features(X, n_lags=N_LAGS):
    n, d = X.shape
    X_out = np.full((n, d * (n_lags + 1)), np.nan)
    for lag in range(n_lags + 1):
        if lag == 0:
            X_out[:, :d] = X
        else:
            X_out[lag:, lag * d:(lag + 1) * d] = X[:-lag]
    return X_out


def train_xgb(X_train, y_train, n_lags=N_LAGS):
    X_lagged = _add_lag_features(X_train, n_lags)
    valid = ~np.isnan(X_lagged).any(axis=1)
    X_lagged = X_lagged[valid]
    y_d = y_train[valid]
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X_lagged, y_d)
    return model


def predict_xgb(model, X_test, n_lags=N_LAGS):
    X_lagged = _add_lag_features(X_test, n_lags)
    valid = ~np.isnan(X_lagged).any(axis=1)
    probs = np.full(len(X_test), 0.5)
    if valid.sum() > 0:
        probs[valid] = model.predict_proba(X_lagged[valid])[:, 1]
    return probs


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    results = {'experiment': 'R145_XGB_Refinement'}

    print("=" * 80, flush=True)
    print("  R145 — XGBoost Entry Signal Refinement", flush=True)
    print("=" * 80, flush=True)

    # ═══════════════════════════════════════════════════════════
    # Phase 1: Load data & build features
    # ═══════════════════════════════════════════════════════════
    print("\n[Phase 1] Loading data & building 19 features...", flush=True)

    h1_path = next((p for p in H1_CANDIDATES if p.exists()), H1_CANDIDATES[-1])
    h1 = load_csv(str(h1_path))
    print(f"  H1 bars: {len(h1)} ({h1.index[0].date()} to {h1.index[-1].date()})", flush=True)

    macro_df = None
    if ALIGNED_CSV.exists():
        macro_df = pd.read_csv(str(ALIGNED_CSV), index_col=0, parse_dates=True)
        if macro_df.index.tz is not None:
            macro_df.index = macro_df.index.tz_localize(None)
        print(f"  Macro data: {len(macro_df)} days, cols={list(macro_df.columns[:8])}", flush=True)
    else:
        print("  [WARN] Macro CSV not found, using zeros", flush=True)

    h1['ret_1'] = h1['Close'].pct_change(1)
    h1['ret_4'] = h1['Close'].pct_change(4)
    h1['ret_8'] = h1['Close'].pct_change(8)
    h1['ret_24'] = h1['Close'].pct_change(24)
    h1['ATR14'] = compute_atr(h1, 14)
    h1['RSI14'] = compute_rsi(h1['Close'], 14)
    h1['RSI2'] = compute_rsi(h1['Close'], 2)
    h1['KC_pos'] = compute_kc_position(h1['Close'])
    ema12 = h1['Close'].ewm(span=12).mean()
    ema26 = h1['Close'].ewm(span=26).mean()
    h1['MACD_hist'] = (ema12 - ema26) - (ema12 - ema26).ewm(span=9).mean()
    h1['volume'] = h1['Volume'] if 'Volume' in h1.columns else 0

    h1_daily = h1.resample('1D').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
    h1_daily['EMA20'] = h1_daily['Close'].ewm(span=20).mean()
    h1_daily['daily_range'] = (h1_daily['High'] - h1_daily['Low']) / h1_daily['Close']
    h1_daily['daily_mom'] = h1_daily['Close'].pct_change(5)
    d1_features = h1_daily[['EMA20', 'daily_range', 'daily_mom']].copy()
    d1_features['EMA20_trend'] = (h1_daily['Close'] - h1_daily['EMA20']) / h1_daily['EMA20']

    h1['date'] = h1.index.date
    d1_features.index = d1_features.index.date
    for col in ['EMA20_trend', 'daily_range', 'daily_mom']:
        h1[col] = h1['date'].map(d1_features[col].to_dict())

    macro_cols = ['VIX', 'DXY', 'US10Y', 'real_yield']
    if macro_df is not None:
        has_all = all(c in macro_df.columns for c in macro_cols)
        if has_all:
            macro_daily = macro_df[macro_cols].copy()
            macro_daily.index = macro_daily.index.date
            for col in macro_cols:
                h1[col] = h1['date'].map(macro_daily[col].to_dict())
        else:
            for col in macro_cols:
                h1[col] = 0.0
    else:
        for col in macro_cols:
            h1[col] = 0.0

    h1['hour'] = h1.index.hour
    h1['dow'] = h1.index.dayofweek

    h1['fwd_ret'] = h1['Close'].shift(-FORWARD_BARS) / h1['Close'] - 1
    h1['target_dir'] = (h1['fwd_ret'] > 0).astype(int)
    h1 = h1.dropna(subset=CONTINUOUS_FEATURES + ['fwd_ret'])
    h1[CONTINUOUS_FEATURES] = h1[CONTINUOUS_FEATURES].fillna(0)

    print(f"  Samples after cleanup: {len(h1)}", flush=True)
    print(f"  Direction balance (up ratio): {h1['target_dir'].mean():.3f}", flush=True)
    print(f"  Features: {len(ALL_FEATURES)} ({CONTINUOUS_FEATURES[:5]}...)", flush=True)

    results['phase1'] = {
        'h1_bars': len(h1), 'features': ALL_FEATURES,
        'up_ratio': round(float(h1['target_dir'].mean()), 3),
    }

    # ═══════════════════════════════════════════════════════════
    # Phase 2: Threshold sweep
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("[Phase 2] Threshold Sweep (train 2015-2021, test 2021-2026)", flush=True)
    print("=" * 70, flush=True)

    timestamps = h1.index
    split_date = pd.Timestamp('2021-01-01', tz='UTC')
    train_mask = timestamps < split_date
    test_mask = timestamps >= split_date

    scaler = StandardScaler()
    X_all = h1[ALL_FEATURES].values.astype(np.float32)
    X_train_raw = X_all[train_mask]
    X_test_raw = X_all[test_mask]
    scaler.fit(X_train_raw)
    X_train_sc = scaler.transform(X_train_raw)
    X_test_sc = scaler.transform(X_test_raw)

    y_train = h1['target_dir'].values[train_mask].astype(np.float32)
    y_test = h1['target_dir'].values[test_mask].astype(np.float32)
    fwd_ret_test = h1['fwd_ret'].values[test_mask]
    test_times = timestamps[test_mask]

    print(f"  Train: {train_mask.sum()} bars, Test: {test_mask.sum()} bars", flush=True)
    print("  Training XGBoost (n_lags=8)...", flush=True)

    model_full = train_xgb(X_train_sc, y_train)
    probs_full = predict_xgb(model_full, X_test_sc)

    overall_auc = roc_auc_score(y_test, probs_full)
    print(f"  Overall test AUC: {overall_auc:.4f}", flush=True)

    sweep_results = []
    print(f"\n  {'Thresh':>6} {'Signals':>8} {'SigRate':>8} {'AUC_sub':>8} {'MeanRet':>9} {'PnL_pips':>10}", flush=True)
    print(f"  {'─'*52}", flush=True)

    for thr in THRESHOLDS:
        long_mask = probs_full >= thr
        short_mask = probs_full <= (1.0 - thr)
        sig_mask = long_mask | short_mask
        n_sig = int(sig_mask.sum())
        sig_rate = n_sig / len(probs_full) * 100 if len(probs_full) > 0 else 0

        if n_sig > 20:
            try:
                auc_sub = roc_auc_score(y_test[sig_mask], probs_full[sig_mask])
            except ValueError:
                auc_sub = 0.5
        else:
            auc_sub = 0.5

        dir_pred = np.where(long_mask, 1.0, np.where(short_mask, -1.0, 0.0))
        triggered_ret = np.where(sig_mask, dir_pred * fwd_ret_test, 0.0)
        mean_ret = float(np.mean(triggered_ret[sig_mask])) if n_sig > 0 else 0.0
        pnl_pips = float(np.sum(triggered_ret)) * 10000

        sweep_results.append({
            'threshold': thr, 'n_signals': n_sig,
            'signal_rate_pct': round(sig_rate, 1),
            'auc_subset': round(auc_sub, 4),
            'mean_ret': round(mean_ret, 6),
            'pnl_pips': round(pnl_pips, 1),
        })
        print(f"  {thr:>6.2f} {n_sig:>8} {sig_rate:>7.1f}% {auc_sub:>8.4f} {mean_ret:>9.6f} {pnl_pips:>10.1f}", flush=True)

    best_thr_row = max(sweep_results, key=lambda r: r['pnl_pips'])
    best_threshold = best_thr_row['threshold']
    print(f"\n  Best threshold by PnL: {best_threshold} (pips={best_thr_row['pnl_pips']:.1f})", flush=True)

    results['phase2'] = {
        'overall_auc': round(overall_auc, 4),
        'sweep': sweep_results,
        'best_threshold': best_threshold,
    }

    # ═══════════════════════════════════════════════════════════
    # Phase 3: Strategy-specific ML filter
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("[Phase 3] Strategy-Specific ML Filter", flush=True)
    print("=" * 70, flush=True)

    h1_test = h1[test_mask].copy()

    test_idx = test_times.tz_localize(None) if hasattr(test_times, 'tz') and test_times.tz is not None else test_times
    probs_series = pd.Series(probs_full, index=test_idx)

    strat_filter_results = {}
    strat_funcs = {
        'PSAR': lambda df: bt_psar(df, SPREAD, UNIT_LOT, CAPS['PSAR']),
        'TSMOM': lambda df: bt_tsmom(df, SPREAD, UNIT_LOT, CAPS['TSMOM']),
        'SESS_BO': lambda df: bt_sess_bo(df, SPREAD, UNIT_LOT, CAPS['SESS_BO']),
    }

    for sname, bt_func in strat_funcs.items():
        print(f"\n  Strategy: {sname}", flush=True)

        base_trades = bt_func(h1_test)
        base_stats = _compute_stats(base_trades)
        print(f"    Base: n={base_stats['n']}, Sharpe={base_stats['sharpe']:.3f}, "
              f"PnL=${base_stats['pnl']:.2f}, WR={base_stats['wr']:.1f}%", flush=True)

        filtered_trades = []
        for t in base_trades:
            entry_ts = pd.Timestamp(t['entry_time'])
            if entry_ts.tzinfo is not None:
                entry_ts = entry_ts.tz_localize(None)

            idx = probs_series.index.searchsorted(entry_ts)
            if idx >= len(probs_series):
                idx = len(probs_series) - 1

            p = probs_series.iloc[idx]

            if t['dir'] == 'BUY' and p >= best_threshold:
                filtered_trades.append(t)
            elif t['dir'] == 'SELL' and p <= (1.0 - best_threshold):
                filtered_trades.append(t)

        filt_stats = _compute_stats(filtered_trades)
        print(f"    ML-filtered (thr={best_threshold}): n={filt_stats['n']}, "
              f"Sharpe={filt_stats['sharpe']:.3f}, PnL=${filt_stats['pnl']:.2f}, "
              f"WR={filt_stats['wr']:.1f}%", flush=True)

        keep_rate = filt_stats['n'] / base_stats['n'] * 100 if base_stats['n'] > 0 else 0
        sharpe_delta = filt_stats['sharpe'] - base_stats['sharpe']
        print(f"    Keep rate: {keep_rate:.1f}%, Sharpe delta: {sharpe_delta:+.3f}", flush=True)

        strat_filter_results[sname] = {
            'base': base_stats,
            'filtered': filt_stats,
            'keep_rate_pct': round(keep_rate, 1),
            'sharpe_delta': round(sharpe_delta, 3),
        }

    results['phase3'] = strat_filter_results

    # ═══════════════════════════════════════════════════════════
    # Phase 4: Walk-Forward PnL
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("[Phase 4] Walk-Forward PnL (6mo train / 2mo test, PSAR filter)", flush=True)
    print("=" * 70, flush=True)

    wf_start = pd.Timestamp('2019-01-01', tz='UTC')
    wf_end = timestamps[-1]
    train_months = 6
    test_months = 2

    wf_pnl_results = []
    wf_importances = []
    current_start = wf_start

    while current_start + pd.DateOffset(months=train_months + test_months) <= wf_end:
        train_end = current_start + pd.DateOffset(months=train_months)
        test_end = train_end + pd.DateOffset(months=test_months)

        tr_mask = (timestamps >= current_start) & (timestamps < train_end)
        te_mask = (timestamps >= train_end) & (timestamps < test_end)

        if tr_mask.sum() < 200 or te_mask.sum() < 50:
            current_start += pd.DateOffset(months=test_months)
            continue

        X_tr = scaler.fit_transform(X_all[tr_mask])
        X_te = scaler.transform(X_all[te_mask])
        y_tr = h1['target_dir'].values[tr_mask].astype(np.float32)
        y_te = h1['target_dir'].values[te_mask].astype(np.float32)

        mdl = train_xgb(X_tr, y_tr)
        probs_te = predict_xgb(mdl, X_te)

        try:
            fold_auc = roc_auc_score(y_te, probs_te)
        except ValueError:
            fold_auc = 0.5

        feat_imp = mdl.feature_importances_
        n_base_feat = len(ALL_FEATURES)
        base_imp = np.zeros(n_base_feat)
        for lag in range(N_LAGS + 1):
            start_col = lag * n_base_feat
            end_col = start_col + n_base_feat
            if end_col <= len(feat_imp):
                base_imp += feat_imp[start_col:end_col]
        top5_idx = np.argsort(base_imp)[-5:][::-1]
        top5_names = [ALL_FEATURES[i] for i in top5_idx]
        wf_importances.append(set(top5_names))

        h1_te_slice = h1[te_mask].copy()
        psar_trades_base = bt_psar(h1_te_slice, SPREAD, UNIT_LOT, CAPS['PSAR'])
        te_idx = timestamps[te_mask]
        if hasattr(te_idx, 'tz') and te_idx.tz is not None:
            te_idx = te_idx.tz_localize(None)
        probs_te_series = pd.Series(probs_te, index=te_idx)

        psar_trades_filt = []
        for t in psar_trades_base:
            entry_ts = pd.Timestamp(t['entry_time'])
            if entry_ts.tzinfo is not None:
                entry_ts = entry_ts.tz_localize(None)
            idx = probs_te_series.index.searchsorted(entry_ts)
            if idx >= len(probs_te_series):
                idx = len(probs_te_series) - 1
            p = probs_te_series.iloc[idx]
            if t['dir'] == 'BUY' and p >= best_threshold:
                psar_trades_filt.append(t)
            elif t['dir'] == 'SELL' and p <= (1.0 - best_threshold):
                psar_trades_filt.append(t)

        base_s = _compute_stats(psar_trades_base)
        filt_s = _compute_stats(psar_trades_filt)

        period_str = f"{current_start.strftime('%Y-%m')} -> {test_end.strftime('%Y-%m')}"
        wf_pnl_results.append({
            'period': period_str, 'auc': round(fold_auc, 4),
            'base_n': base_s['n'], 'base_pnl': base_s['pnl'], 'base_sharpe': base_s['sharpe'],
            'filt_n': filt_s['n'], 'filt_pnl': filt_s['pnl'], 'filt_sharpe': filt_s['sharpe'],
            'top5_features': top5_names,
        })
        print(f"  {period_str}: AUC={fold_auc:.4f}  "
              f"Base(n={base_s['n']},pnl=${base_s['pnl']:.0f})  "
              f"Filt(n={filt_s['n']},pnl=${filt_s['pnl']:.0f})  "
              f"top5={top5_names}", flush=True)

        current_start += pd.DateOffset(months=test_months)

    if wf_pnl_results:
        avg_auc = np.mean([r['auc'] for r in wf_pnl_results])
        total_base_pnl = sum(r['base_pnl'] for r in wf_pnl_results)
        total_filt_pnl = sum(r['filt_pnl'] for r in wf_pnl_results)
        print(f"\n  Walk-Forward summary ({len(wf_pnl_results)} folds):", flush=True)
        print(f"    Mean AUC: {avg_auc:.4f}", flush=True)
        print(f"    Total Base PSAR PnL: ${total_base_pnl:.2f}", flush=True)
        print(f"    Total Filtered PSAR PnL: ${total_filt_pnl:.2f}", flush=True)
        print(f"    PnL improvement: ${total_filt_pnl - total_base_pnl:.2f}", flush=True)

    results['phase4'] = {
        'folds': wf_pnl_results,
        'n_folds': len(wf_pnl_results),
        'mean_auc': round(avg_auc, 4) if wf_pnl_results else 0.5,
        'total_base_pnl': round(total_base_pnl, 2) if wf_pnl_results else 0,
        'total_filt_pnl': round(total_filt_pnl, 2) if wf_pnl_results else 0,
    }

    # ═══════════════════════════════════════════════════════════
    # Phase 5: Feature importance stability
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("[Phase 5] Feature Importance Stability (Jaccard)", flush=True)
    print("=" * 70, flush=True)

    if len(wf_importances) >= 2:
        jaccard_scores = []
        for i in range(len(wf_importances)):
            for j in range(i + 1, len(wf_importances)):
                inter = len(wf_importances[i] & wf_importances[j])
                union = len(wf_importances[i] | wf_importances[j])
                jaccard_scores.append(inter / union if union > 0 else 0)

        mean_jaccard = float(np.mean(jaccard_scores))
        min_jaccard = float(np.min(jaccard_scores))
        max_jaccard = float(np.max(jaccard_scores))

        all_top = [f for s in wf_importances for f in s]
        from collections import Counter
        feat_counts = Counter(all_top)
        stable_features = [(f, c) for f, c in feat_counts.most_common(10) if c >= len(wf_importances) * 0.5]

        print(f"  Pairwise Jaccard: mean={mean_jaccard:.3f}, min={min_jaccard:.3f}, max={max_jaccard:.3f}", flush=True)
        print(f"  Stability: {'STABLE' if mean_jaccard >= 0.4 else 'FRAGILE'} (threshold 0.4)", flush=True)
        print(f"  Features appearing in >50% of folds:", flush=True)
        for f, c in stable_features:
            print(f"    {f}: {c}/{len(wf_importances)} folds", flush=True)

        results['phase5'] = {
            'mean_jaccard': round(mean_jaccard, 3),
            'min_jaccard': round(min_jaccard, 3),
            'max_jaccard': round(max_jaccard, 3),
            'stable': mean_jaccard >= 0.4,
            'stable_features': [(f, c) for f, c in stable_features],
        }
    else:
        print("  Not enough folds for Jaccard analysis", flush=True)
        results['phase5'] = {'error': 'insufficient folds'}

    # ═══════════════════════════════════════════════════════════
    # Phase 6: AUC degradation by year
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("[Phase 6] AUC Degradation Analysis by Year", flush=True)
    print("=" * 70, flush=True)

    probs_full_series = pd.Series(probs_full, index=test_times)
    y_test_series = pd.Series(y_test, index=test_times)

    yearly_auc = {}
    for year in range(2019, 2026):
        year_mask = (timestamps >= pd.Timestamp(f'{year}-01-01', tz='UTC')) & \
                    (timestamps < pd.Timestamp(f'{year+1}-01-01', tz='UTC'))
        y_sub = h1['target_dir'].values[year_mask]

        in_test = year_mask & test_mask
        if in_test.sum() < 50:
            if year_mask.sum() >= 200:
                X_yr = scaler.transform(X_all[year_mask])
                y_yr = h1['target_dir'].values[year_mask].astype(np.float32)
                p_yr = predict_xgb(model_full, X_yr)
                try:
                    auc_yr = roc_auc_score(y_yr, p_yr)
                except ValueError:
                    auc_yr = 0.5
                yearly_auc[year] = round(auc_yr, 4)
                print(f"  {year}: AUC={auc_yr:.4f}  (trained-on data, not OOS)", flush=True)
            continue

        test_indices = timestamps[in_test]
        p_yr = probs_full_series.reindex(test_indices).dropna()
        y_yr = y_test_series.reindex(p_yr.index)

        if len(p_yr) < 50:
            continue
        try:
            auc_yr = roc_auc_score(y_yr.values, p_yr.values)
        except ValueError:
            auc_yr = 0.5
        yearly_auc[year] = round(auc_yr, 4)
        print(f"  {year}: AUC={auc_yr:.4f}  (n={len(p_yr)})", flush=True)

    if len(yearly_auc) >= 3:
        years_sorted = sorted(yearly_auc.keys())
        aucs_sorted = [yearly_auc[y] for y in years_sorted]
        from scipy.stats import linregress
        slope, intercept, r_val, p_val, std_err = linregress(years_sorted, aucs_sorted)
        drifting = slope < -0.01 and p_val < 0.1
        print(f"\n  Linear trend: slope={slope:.4f}/year, R²={r_val**2:.3f}, p={p_val:.3f}", flush=True)
        print(f"  Concept drift: {'YES — AUC declining' if drifting else 'NO significant drift'}", flush=True)

        results['phase6'] = {
            'yearly_auc': yearly_auc,
            'trend_slope': round(slope, 4),
            'trend_r2': round(r_val ** 2, 3),
            'trend_p_value': round(p_val, 3),
            'drifting': drifting,
        }
    else:
        print("  Not enough yearly data for trend analysis", flush=True)
        results['phase6'] = {'yearly_auc': yearly_auc, 'drifting': 'unknown'}

    # ═══════════════════════════════════════════════════════════
    # Phase 7: K-Fold 5-fold of best combo
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print(f"[Phase 7] K-Fold 5-Fold — Best combo (thr={best_threshold})", flush=True)
    print("=" * 70, flush=True)

    best_strat = max(strat_filter_results.keys(),
                     key=lambda s: strat_filter_results[s]['filtered']['sharpe'])
    print(f"  Best strategy by filtered Sharpe: {best_strat}", flush=True)

    best_bt = strat_funcs[best_strat]

    n_total = len(h1)
    fold_size = n_total // 5
    kfold_results = []

    for fold_i in range(5):
        te_start = fold_i * fold_size
        te_end = min((fold_i + 1) * fold_size, n_total)
        tr_indices = np.concatenate([np.arange(0, te_start), np.arange(te_end, n_total)])
        te_indices = np.arange(te_start, te_end)

        if len(tr_indices) < 200 or len(te_indices) < 50:
            continue

        sc_k = StandardScaler()
        X_tr_k = sc_k.fit_transform(X_all[tr_indices])
        X_te_k = sc_k.transform(X_all[te_indices])
        y_tr_k = h1['target_dir'].values[tr_indices].astype(np.float32)
        y_te_k = h1['target_dir'].values[te_indices].astype(np.float32)

        mdl_k = train_xgb(X_tr_k, y_tr_k)
        probs_k = predict_xgb(mdl_k, X_te_k)

        try:
            auc_k = roc_auc_score(y_te_k, probs_k)
        except ValueError:
            auc_k = 0.5

        h1_fold = h1.iloc[te_indices].copy()
        base_trades_k = best_bt(h1_fold)
        fold_idx = h1_fold.index
        if hasattr(fold_idx, 'tz') and fold_idx.tz is not None:
            fold_idx = fold_idx.tz_localize(None)
        probs_k_series = pd.Series(probs_k, index=fold_idx)

        filt_trades_k = []
        for t in base_trades_k:
            entry_ts = pd.Timestamp(t['entry_time'])
            if entry_ts.tzinfo is not None:
                entry_ts = entry_ts.tz_localize(None)
            idx = probs_k_series.index.searchsorted(entry_ts)
            if idx >= len(probs_k_series):
                idx = len(probs_k_series) - 1
            p = probs_k_series.iloc[idx]
            if t['dir'] == 'BUY' and p >= best_threshold:
                filt_trades_k.append(t)
            elif t['dir'] == 'SELL' and p <= (1.0 - best_threshold):
                filt_trades_k.append(t)

        base_s_k = _compute_stats(base_trades_k)
        filt_s_k = _compute_stats(filt_trades_k)

        kfold_results.append({
            'fold': fold_i,
            'auc': round(auc_k, 4),
            'base_sharpe': base_s_k['sharpe'],
            'filt_sharpe': filt_s_k['sharpe'],
            'base_n': base_s_k['n'],
            'filt_n': filt_s_k['n'],
            'base_pnl': base_s_k['pnl'],
            'filt_pnl': filt_s_k['pnl'],
        })
        print(f"  Fold {fold_i}: AUC={auc_k:.4f}  "
              f"Base(Sharpe={base_s_k['sharpe']:.3f}, n={base_s_k['n']})  "
              f"Filt(Sharpe={filt_s_k['sharpe']:.3f}, n={filt_s_k['n']})", flush=True)

    if kfold_results:
        avg_base_sharpe = np.mean([r['base_sharpe'] for r in kfold_results])
        avg_filt_sharpe = np.mean([r['filt_sharpe'] for r in kfold_results])
        avg_auc_kf = np.mean([r['auc'] for r in kfold_results])
        print(f"\n  K-Fold Mean: AUC={avg_auc_kf:.4f}  "
              f"Base Sharpe={avg_base_sharpe:.3f}  "
              f"Filtered Sharpe={avg_filt_sharpe:.3f}", flush=True)
        print(f"  Sharpe improvement: {avg_filt_sharpe - avg_base_sharpe:+.3f}", flush=True)

    results['phase7'] = {
        'best_strategy': best_strat,
        'best_threshold': best_threshold,
        'folds': kfold_results,
        'mean_auc': round(avg_auc_kf, 4) if kfold_results else 0.5,
        'mean_base_sharpe': round(avg_base_sharpe, 3) if kfold_results else 0,
        'mean_filt_sharpe': round(avg_filt_sharpe, 3) if kfold_results else 0,
    }

    # ═══════════════════════════════════════════════════════════
    # Save results
    # ═══════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    results['elapsed_s'] = round(elapsed, 1)

    print("\n" + "=" * 80, flush=True)
    print("  R145 FINAL SUMMARY", flush=True)
    print("=" * 80, flush=True)
    print(f"\n  Overall AUC (2021-2026): {results['phase2']['overall_auc']:.4f}", flush=True)
    print(f"  Best threshold: {best_threshold} (PnL pips={best_thr_row['pnl_pips']:.1f})", flush=True)
    print(f"  Strategy filter results:", flush=True)
    for sn, sr in strat_filter_results.items():
        print(f"    {sn:>8s}: base Sharpe={sr['base']['sharpe']:.3f} -> "
              f"filtered Sharpe={sr['filtered']['sharpe']:.3f} "
              f"(delta={sr['sharpe_delta']:+.3f}, keep={sr['keep_rate_pct']:.0f}%)", flush=True)
    if 'phase5' in results and 'mean_jaccard' in results['phase5']:
        print(f"  Feature stability: Jaccard={results['phase5']['mean_jaccard']:.3f} "
              f"({'STABLE' if results['phase5']['stable'] else 'FRAGILE'})", flush=True)
    if 'phase6' in results and 'trend_slope' in results['phase6']:
        print(f"  AUC trend: {results['phase6']['trend_slope']:+.4f}/year "
              f"({'DRIFTING' if results['phase6'].get('drifting') else 'STABLE'})", flush=True)
    if kfold_results:
        print(f"  K-Fold best combo ({best_strat}+thr={best_threshold}): "
              f"Sharpe {avg_base_sharpe:.3f} -> {avg_filt_sharpe:.3f}", flush=True)

    worthy = (best_thr_row['pnl_pips'] > 0 and
              any(sr['sharpe_delta'] > 0 for sr in strat_filter_results.values()))
    results['recommendation'] = (
        f"USE threshold={best_threshold} as filter on {best_strat}"
        if worthy else
        "ML filter does NOT improve Sharpe — do not deploy"
    )
    print(f"\n  Recommendation: {results['recommendation']}", flush=True)

    out_file = OUTPUT_DIR / "r145_results.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)
    print(f"  Saved: {out_file}", flush=True)
    print("=" * 80, flush=True)


if __name__ == '__main__':
    main()

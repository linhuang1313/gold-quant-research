#!/usr/bin/env python3
"""
R151 — XGBoost ML Entry Filter: Independent Validation for SESS_BO
===================================================================
R145 showed XGB filtering for SESS_BO: Sharpe 7.72 -> 11.82 (+4.10) but only
41% trade retention. This experiment does a complete, independent validation.

Phases:
  1. Feature matrix construction from SESS_BO trades
  2. Full-sample AUC and threshold sweep
  3. Walk-Forward validation (8 windows, 3yr train / 1yr test)
  4. K-Fold 5-fold cross-validation
  5. Yearly breakdown (2015-2025)
  6. Feature importance stability
  7. Portfolio impact (PSAR, TSMOM, SESS_BO)

RESEARCH-ONLY — no deployment.
"""
import sys, os, time, json, glob
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

OUTPUT_DIR = Path("results/r151_ml_entry_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100
SPREAD = 0.30
UNIT_LOT = 0.01
CAPS = {'PSAR': 5, 'TSMOM': 0, 'SESS_BO': 35}

XGB_PARAMS = dict(
    n_estimators=200, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric='auc', use_label_encoder=False, verbosity=0,
    random_state=42,
)

CONTINUOUS_FEATURES = [
    'ret_1', 'ret_4', 'ret_8', 'ret_24', 'ATR14', 'RSI14', 'RSI2',
    'KC_pos', 'MACD_hist', 'volume', 'EMA20_trend', 'daily_range',
    'daily_mom', 'VIX', 'DXY', 'US10Y', 'real_yield',
]
CATEGORICAL_FEATURES = ['hour', 'dow']
ALL_FEATURES = CONTINUOUS_FEATURES + CATEGORICAL_FEATURES

THRESHOLDS = [0.45, 0.48, 0.50, 0.52, 0.55, 0.58, 0.60, 0.65]

WF_WINDOWS = [
    ("WF1", "2015-01-01", "2018-01-01", "2018-01-01", "2019-01-01"),
    ("WF2", "2016-01-01", "2019-01-01", "2019-01-01", "2020-01-01"),
    ("WF3", "2017-01-01", "2020-01-01", "2020-01-01", "2021-01-01"),
    ("WF4", "2018-01-01", "2021-01-01", "2021-01-01", "2022-01-01"),
    ("WF5", "2019-01-01", "2022-01-01", "2022-01-01", "2023-01-01"),
    ("WF6", "2020-01-01", "2023-01-01", "2023-01-01", "2024-01-01"),
    ("WF7", "2021-01-01", "2024-01-01", "2024-01-01", "2025-01-01"),
    ("WF8", "2022-01-01", "2025-01-01", "2025-01-01", "2026-05-01"),
]

KFOLDS = [
    ("Fold1", "2015-01-01", "2017-03-01"),
    ("Fold2", "2017-03-01", "2019-06-01"),
    ("Fold3", "2019-06-01", "2021-09-01"),
    ("Fold4", "2021-09-01", "2023-12-01"),
    ("Fold5", "2023-12-01", "2027-01-01"),
]

t0 = time.time()


# ═══════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════

def load_h1():
    candidates = sorted(glob.glob("data/download/xauusd-h1-bid-2015-*.csv"))
    if not candidates:
        raise FileNotFoundError("No xauusd H1 CSV found")
    csv_path = candidates[-1]
    print(f"  Loading H1 data from {csv_path}", flush=True)
    df = pd.read_csv(csv_path)
    df['time'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_localize(None)
    df = df.set_index('time')[['open', 'high', 'low', 'close', 'volume']]
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = df[df['Close'] > 100]
    print(f"  H1 bars: {len(df)}  ({df.index[0]} to {df.index[-1]})", flush=True)
    return df


def load_macro():
    macro_path = Path("data/external/aligned_daily.csv")
    if not macro_path.exists():
        print("  Macro data not found — skipping VIX/DXY/US10Y/real_yield features", flush=True)
        return None
    df = pd.read_csv(macro_path, parse_dates=['Date'], index_col='Date')
    print(f"  Macro data loaded: {len(df)} rows, cols={list(df.columns)}", flush=True)
    return df


# ═══════════════════════════════════════════════════════════════
# Indicators
# ═══════════════════════════════════════════════════════════════

def compute_atr(df, period=14):
    tr = pd.DataFrame({
        'hl': df['High'] - df['Low'],
        'hc': (df['High'] - df['Close'].shift(1)).abs(),
        'lc': (df['Low'] - df['Close'].shift(1)).abs(),
    }).max(axis=1)
    return tr.rolling(period).mean()


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def compute_kc_position(close, period=25, atr_mult=1.2, atr_period=14):
    kc_mid = close.ewm(span=period).mean()
    atr = (close.rolling(atr_period).max() - close.rolling(atr_period).min()).rolling(atr_period).mean()
    kc_upper = kc_mid + atr_mult * atr
    kc_lower = kc_mid - atr_mult * atr
    width = kc_upper - kc_lower
    pos = (close - kc_lower) / width.replace(0, np.nan)
    return pos.fillna(0.5)


def add_psar(df, af_start=0.01, af_max=0.05):
    df = df.copy(); n = len(df)
    psar = np.zeros(n); direction = np.ones(n)
    af = af_start; ep = df['High'].iloc[0]; psar[0] = df['Low'].iloc[0]
    for i in range(1, n):
        prev = psar[i-1]
        if direction[i-1] == 1:
            psar[i] = prev + af * (ep - prev)
            psar[i] = min(psar[i], df['Low'].iloc[i-1], df['Low'].iloc[max(0, i-2)])
            if df['Low'].iloc[i] < psar[i]:
                direction[i] = -1; psar[i] = ep; ep = df['Low'].iloc[i]; af = af_start
            else:
                direction[i] = 1
                if df['High'].iloc[i] > ep:
                    ep = df['High'].iloc[i]; af = min(af+af_start, af_max)
        else:
            psar[i] = prev + af * (ep - prev)
            psar[i] = max(psar[i], df['High'].iloc[i-1], df['High'].iloc[max(0, i-2)])
            if df['High'].iloc[i] > psar[i]:
                direction[i] = 1; psar[i] = ep; ep = df['High'].iloc[i]; af = af_start
            else:
                direction[i] = -1
                if df['Low'].iloc[i] < ep:
                    ep = df['Low'].iloc[i]; af = min(af+af_start, af_max)
    df['PSAR_dir'] = direction; df['ATR'] = compute_atr(df)
    return df


# ═══════════════════════════════════════════════════════════════
# Backtest helpers
# ═══════════════════════════════════════════════════════════════

def _mk(pos, exit_p, exit_time, reason, bar_idx, pnl):
    return {'dir': pos['dir'], 'entry': pos['entry'], 'exit': exit_p,
            'entry_time': pos['time'], 'exit_time': exit_time,
            'pnl': pnl, 'reason': reason, 'bars': bar_idx - pos['bar']}


def _run_exit_with_cap(pos, i, h, lo_v, c, spread, lot, pv, times,
                       sl_atr, tp_atr, trail_act_atr, trail_dist_atr, max_hold,
                       maxloss_cap=0):
    held = i - pos['bar']
    if pos['dir'] == 'BUY':
        pnl_h = (h - pos['entry'] - spread) * lot * pv
        pnl_l = (lo_v - pos['entry'] - spread) * lot * pv
        pnl_c = (c - pos['entry'] - spread) * lot * pv
    else:
        pnl_h = (pos['entry'] - lo_v - spread) * lot * pv
        pnl_l = (pos['entry'] - h - spread) * lot * pv
        pnl_c = (pos['entry'] - c - spread) * lot * pv
    tp_val = tp_atr * pos['atr'] * lot * pv
    sl_val = sl_atr * pos['atr'] * lot * pv
    if pnl_h >= tp_val:
        return _mk(pos, c, times[i], "TP", i, tp_val)
    if pnl_l <= -sl_val:
        return _mk(pos, c, times[i], "SL", i, -sl_val)
    if maxloss_cap > 0 and pnl_c < -maxloss_cap:
        return _mk(pos, c, times[i], "MaxLossCap", i, -maxloss_cap)
    ad = trail_act_atr * pos['atr']; td = trail_dist_atr * pos['atr']
    if pos['dir'] == 'BUY' and h - pos['entry'] >= ad:
        ts_p = h - td
        if lo_v <= ts_p:
            return _mk(pos, c, times[i], "Trail", i, (ts_p - pos['entry'] - spread) * lot * pv)
    elif pos['dir'] == 'SELL' and pos['entry'] - lo_v >= ad:
        ts_p = lo_v + td
        if h >= ts_p:
            return _mk(pos, c, times[i], "Trail", i, (pos['entry'] - ts_p - spread) * lot * pv)
    if held >= max_hold:
        return _mk(pos, c, times[i], "Timeout", i, pnl_c)
    return None


# ═══════════════════════════════════════════════════════════════
# Strategy backtests
# ═══════════════════════════════════════════════════════════════

def bt_sess_bo(h1_df, spread, lot, maxloss_cap=0,
               session_hour=12, lookback=4, sl_atr=4.5, tp_atr=4.0,
               trail_act=0.14, trail_dist=0.025, max_hold=20):
    df = h1_df.copy(); df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; hours = df.index.hour; times = df.index; n = len(df)
    trades = []; pos = None; last_exit = -999
    for i in range(lookback, n):
        if pos is not None:
            result = _run_exit_with_cap(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                                        sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if hours[i] != session_hour: continue
        if i-last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        hh = max(h[i-j] for j in range(1, lookback+1))
        ll = min(lo[i-j] for j in range(1, lookback+1))
        if c[i] > hh:
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif c[i] < ll:
            pos = {'dir': 'SELL', 'entry': c[i]-spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_psar(h1_df, spread, lot, maxloss_cap=0,
            sl_atr=4.5, tp_atr=16.0, trail_act=0.20, trail_dist=0.04, max_hold=20):
    df = add_psar(h1_df).dropna(subset=['PSAR_dir', 'ATR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    pdir = df['PSAR_dir'].values; atr = df['ATR'].values
    times = df.index; n = len(df)
    trades = []; pos = None; last_exit = -999
    for i in range(1, n):
        if pos is not None:
            result = _run_exit_with_cap(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                                        sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if i-last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        if pdir[i-1] == -1 and pdir[i] == 1:
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif pdir[i-1] == 1 and pdir[i] == -1:
            pos = {'dir': 'SELL', 'entry': c[i]-spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_tsmom(h1_df, spread, lot, maxloss_cap=0,
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
            result = _run_exit_with_cap(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                                        sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
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


# ═══════════════════════════════════════════════════════════════
# Stats helpers
# ═══════════════════════════════════════════════════════════════

def trades_to_daily(trades):
    if not trades:
        return pd.Series(dtype=float)
    daily = {}
    for t in trades:
        d = pd.Timestamp(t['exit_time']).date()
        daily[d] = daily.get(d, 0) + t['pnl']
    dates = sorted(daily.keys())
    return pd.Series([daily[d] for d in dates], index=pd.DatetimeIndex(dates))


def sharpe(arr):
    if len(arr) < 10:
        return 0.0
    s = np.std(arr, ddof=1)
    if s == 0:
        return 0.0
    return float(np.mean(arr) / s * np.sqrt(252))


def max_dd(arr):
    if len(arr) == 0:
        return 0.0
    eq = np.cumsum(arr)
    return float((np.maximum.accumulate(eq) - eq).max())


def compute_stats(trades):
    if not trades:
        return {'n': 0, 'sharpe': 0.0, 'pnl': 0.0, 'wr': 0.0, 'max_dd': 0.0}
    daily = trades_to_daily(trades)
    pnls = [t['pnl'] for t in trades]
    n = len(trades)
    return {
        'n': n,
        'sharpe': round(sharpe(daily.values), 3),
        'pnl': round(sum(pnls), 2),
        'wr': round(sum(1 for p in pnls if p > 0) / n * 100, 1),
        'max_dd': round(max_dd(daily.values), 2),
    }


# ═══════════════════════════════════════════════════════════════
# Feature engineering
# ═══════════════════════════════════════════════════════════════

def build_features(h1_df, macro_df=None):
    """Add all 19 features to H1 dataframe in-place and return it."""
    h1 = h1_df.copy()

    h1['ret_1'] = np.log(h1['Close'] / h1['Close'].shift(1))
    h1['ret_4'] = np.log(h1['Close'] / h1['Close'].shift(4))
    h1['ret_8'] = np.log(h1['Close'] / h1['Close'].shift(8))
    h1['ret_24'] = np.log(h1['Close'] / h1['Close'].shift(24))
    h1['ATR14'] = compute_atr(h1, 14)
    h1['RSI14'] = compute_rsi(h1['Close'], 14)
    h1['RSI2'] = compute_rsi(h1['Close'], 2)
    h1['KC_pos'] = compute_kc_position(h1['Close'])

    ema12 = h1['Close'].ewm(span=12).mean()
    ema26 = h1['Close'].ewm(span=26).mean()
    macd_line = ema12 - ema26
    h1['MACD_hist'] = macd_line - macd_line.ewm(span=9).mean()

    h1['volume'] = h1['Volume'] if 'Volume' in h1.columns else 0

    h1_daily = h1.resample('1D').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
    }).dropna()
    h1_daily['EMA20'] = h1_daily['Close'].ewm(span=20).mean()
    h1_daily['daily_range'] = (h1_daily['High'] - h1_daily['Low']) / h1_daily['Close']
    h1_daily['daily_mom'] = h1_daily['Close'] / h1_daily['Close'].shift(1) - 1
    d1_features = h1_daily[['EMA20', 'daily_range', 'daily_mom']].copy()
    d1_features['EMA20_trend'] = (h1_daily['Close'] - h1_daily['EMA20']) / h1_daily['EMA20']

    h1['date'] = h1.index.date
    d1_features.index = d1_features.index.date
    for col in ['EMA20_trend', 'daily_range', 'daily_mom']:
        h1[col] = h1['date'].map(d1_features[col].to_dict())

    macro_cols = ['VIX', 'DXY', 'US10Y', 'real_yield']
    macro_rename = {'VIX_Close': 'VIX', 'DXY_Close': 'DXY', 'US10Y_Close': 'US10Y'}
    if macro_df is not None:
        mdf = macro_df.rename(columns=macro_rename).copy()
        if 'US10Y' in mdf.columns and 'US2Y_Close' in macro_df.columns:
            mdf['real_yield'] = mdf['US10Y'] - macro_df['US2Y_Close']
        elif 'real_yield' not in mdf.columns:
            mdf['real_yield'] = 0.0
        available = [c for c in macro_cols if c in mdf.columns]
        if available:
            macro_daily = mdf[available].copy()
            macro_daily.index = macro_daily.index.date
            for col in available:
                h1[col] = h1['date'].map(macro_daily[col].to_dict())
            for col in macro_cols:
                if col not in available:
                    h1[col] = 0.0
        else:
            for col in macro_cols:
                h1[col] = 0.0
    else:
        for col in macro_cols:
            h1[col] = 0.0

    h1['hour'] = h1.index.hour
    h1['dow'] = h1.index.dayofweek

    h1 = h1.dropna(subset=CONTINUOUS_FEATURES[:9])
    h1[CONTINUOUS_FEATURES] = h1[CONTINUOUS_FEATURES].fillna(0)

    return h1


def get_trade_features(trades, h1_feat):
    """Extract feature vectors at entry_time for each trade. Returns X, y, valid_trades."""
    X_rows = []
    y_labels = []
    valid_trades = []

    for t in trades:
        entry_ts = pd.Timestamp(t['entry_time'])
        if entry_ts.tzinfo is not None:
            entry_ts = entry_ts.tz_localize(None)

        idx = h1_feat.index.searchsorted(entry_ts)
        if idx >= len(h1_feat):
            idx = len(h1_feat) - 1

        closest_ts = h1_feat.index[idx]
        if abs((closest_ts - entry_ts).total_seconds()) > 7200:
            continue

        row = h1_feat.iloc[idx][ALL_FEATURES].values.astype(np.float32)
        if np.isnan(row).sum() > 5:
            continue

        row = np.nan_to_num(row, nan=0.0)
        X_rows.append(row)
        y_labels.append(1 if t['pnl'] > 0 else 0)
        valid_trades.append(t)

    if not X_rows:
        return np.array([]), np.array([]), []

    return np.array(X_rows), np.array(y_labels), valid_trades


def filter_trades_by_ml(trades, h1_feat, model, scaler, threshold=0.50):
    """Apply ML filter to trades. Returns filtered trade list."""
    X, y, valid = get_trade_features(trades, h1_feat)
    if len(X) == 0:
        return []

    X_scaled = scaler.transform(X)
    probs = model.predict_proba(X_scaled)[:, 1]

    filtered = []
    for i, t in enumerate(valid):
        p = probs[i]
        if t['dir'] == 'BUY' and p >= threshold:
            filtered.append(t)
        elif t['dir'] == 'SELL' and p <= (1.0 - threshold):
            filtered.append(t)
    return filtered


# ═══════════════════════════════════════════════════════════════
# Main experiment
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 70, flush=True)
    print("R151 — XGBoost ML Entry Filter: Independent SESS_BO Validation", flush=True)
    print("=" * 70, flush=True)
    print(f"Started: {datetime.now()}", flush=True)

    results = {}

    # ─── Load data ───
    print("\n[Data] Loading H1 data...", flush=True)
    h1_raw = load_h1()
    macro_df = load_macro()

    print("\n[Data] Building features...", flush=True)
    h1 = build_features(h1_raw, macro_df)
    print(f"  Feature matrix: {len(h1)} bars x {len(ALL_FEATURES)} features", flush=True)

    # ─── Run base SESS_BO backtest ───
    print("\n[Base] Running SESS_BO backtest on full data...", flush=True)
    all_trades = bt_sess_bo(h1_raw, SPREAD, UNIT_LOT, CAPS['SESS_BO'])
    base_stats = compute_stats(all_trades)
    print(f"  Total trades: {base_stats['n']}", flush=True)
    print(f"  Base Sharpe: {base_stats['sharpe']:.3f}", flush=True)
    print(f"  Base PnL: ${base_stats['pnl']:.2f}", flush=True)
    print(f"  Win rate: {base_stats['wr']:.1f}%", flush=True)
    print(f"  Max DD: ${base_stats['max_dd']:.2f}", flush=True)

    # ═══════════════════════════════════════════════════════════
    # Phase 1: Feature matrix construction
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("[Phase 1] Feature matrix construction from SESS_BO trades", flush=True)
    print("=" * 70, flush=True)

    X_all, y_all, valid_trades = get_trade_features(all_trades, h1)
    n_trades = len(valid_trades)
    print(f"  Valid trades with features: {n_trades} / {len(all_trades)}", flush=True)
    print(f"  Feature matrix shape: {X_all.shape}", flush=True)
    print(f"  Label distribution: {y_all.mean():.3f} (win rate in labels)", flush=True)

    results['phase1'] = {
        'total_trades': len(all_trades),
        'valid_trades': n_trades,
        'feature_count': len(ALL_FEATURES),
        'features': ALL_FEATURES,
        'label_win_rate': round(float(y_all.mean()), 3),
        'base_stats': base_stats,
    }

    if n_trades < 50:
        print("\n  ERROR: Too few trades for ML analysis. Aborting.", flush=True)
        return

    # ═══════════════════════════════════════════════════════════
    # Phase 2: Full-sample AUC and threshold sweep
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("[Phase 2] Full-sample AUC & threshold sweep (in-sample, NOT for deployment)", flush=True)
    print("=" * 70, flush=True)

    scaler_full = StandardScaler()
    X_scaled = scaler_full.fit_transform(X_all)

    model_full = xgb.XGBClassifier(**XGB_PARAMS)
    model_full.fit(X_scaled, y_all)
    probs_full = model_full.predict_proba(X_scaled)[:, 1]

    try:
        full_auc = roc_auc_score(y_all, probs_full)
    except ValueError:
        full_auc = 0.5

    print(f"  Full-sample AUC: {full_auc:.4f}", flush=True)
    print(f"  (This is in-sample — expect overfit. OOS results in Phase 3 & 4)", flush=True)

    print(f"\n  {'Threshold':>10} {'Kept':>6} {'Keep%':>7} {'Filt Sharpe':>12} {'Filt PnL':>12} {'Base Sharpe':>12}", flush=True)
    print(f"  {'-'*65}", flush=True)

    threshold_results = []
    for thr in THRESHOLDS:
        kept_trades = []
        for i, t in enumerate(valid_trades):
            p = probs_full[i]
            if t['dir'] == 'BUY' and p >= thr:
                kept_trades.append(t)
            elif t['dir'] == 'SELL' and p <= (1.0 - thr):
                kept_trades.append(t)

        filt_stats = compute_stats(kept_trades)
        keep_pct = len(kept_trades) / n_trades * 100

        print(f"  {thr:>10.2f} {len(kept_trades):>6} {keep_pct:>6.1f}% {filt_stats['sharpe']:>12.3f} "
              f"${filt_stats['pnl']:>10.2f} {base_stats['sharpe']:>12.3f}", flush=True)

        threshold_results.append({
            'threshold': thr,
            'kept': len(kept_trades),
            'keep_pct': round(keep_pct, 1),
            'sharpe': filt_stats['sharpe'],
            'pnl': filt_stats['pnl'],
            'wr': filt_stats['wr'],
            'max_dd': filt_stats['max_dd'],
            'sharpe_delta': round(filt_stats['sharpe'] - base_stats['sharpe'], 3),
        })

    feat_imp = model_full.feature_importances_
    top10_idx = np.argsort(feat_imp)[-10:][::-1]
    print(f"\n  Top 10 features (full-sample):", flush=True)
    for rank, idx in enumerate(top10_idx, 1):
        print(f"    {rank:>2}. {ALL_FEATURES[idx]:<15} importance={feat_imp[idx]:.4f}", flush=True)

    results['phase2'] = {
        'full_auc': round(full_auc, 4),
        'threshold_sweep': threshold_results,
        'top10_features': [{'name': ALL_FEATURES[i], 'importance': round(float(feat_imp[i]), 4)} for i in top10_idx],
    }

    # ═══════════════════════════════════════════════════════════
    # Phase 3: Walk-Forward validation
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("[Phase 3] Walk-Forward validation (8 windows, 3yr train / 1yr test)", flush=True)
    print("=" * 70, flush=True)

    wf_results = []
    best_thr = 0.50

    for wf_name, tr_start, tr_end, te_start, te_end in WF_WINDOWS:
        tr_s = pd.Timestamp(tr_start)
        tr_e = pd.Timestamp(tr_end)
        te_s = pd.Timestamp(te_start)
        te_e = pd.Timestamp(te_end)

        train_trades = []
        test_trades = []
        for i, t in enumerate(valid_trades):
            ets = pd.Timestamp(t['entry_time'])
            if ets.tzinfo is not None:
                ets = ets.tz_localize(None)
            if tr_s <= ets < tr_e:
                train_trades.append((i, t))
            if te_s <= ets < te_e:
                test_trades.append((i, t))

        if len(train_trades) < 30 or len(test_trades) < 10:
            print(f"  {wf_name}: SKIP (train={len(train_trades)}, test={len(test_trades)} — too few)", flush=True)
            continue

        tr_idx = [x[0] for x in train_trades]
        te_idx = [x[0] for x in test_trades]
        tr_trades_list = [x[1] for x in train_trades]
        te_trades_list = [x[1] for x in test_trades]

        X_tr = X_all[tr_idx]
        y_tr = y_all[tr_idx]
        X_te = X_all[te_idx]
        y_te = y_all[te_idx]

        scaler_wf = StandardScaler()
        X_tr_s = scaler_wf.fit_transform(X_tr)
        X_te_s = scaler_wf.transform(X_te)

        mdl = xgb.XGBClassifier(**XGB_PARAMS)
        mdl.fit(X_tr_s, y_tr)
        probs_te = mdl.predict_proba(X_te_s)[:, 1]

        try:
            wf_auc = roc_auc_score(y_te, probs_te)
        except ValueError:
            wf_auc = 0.5

        base_test_stats = compute_stats(te_trades_list)

        filtered_te = []
        for j, t in enumerate(te_trades_list):
            p = probs_te[j]
            if t['dir'] == 'BUY' and p >= best_thr:
                filtered_te.append(t)
            elif t['dir'] == 'SELL' and p <= (1.0 - best_thr):
                filtered_te.append(t)

        filt_test_stats = compute_stats(filtered_te)
        keep_pct = filt_test_stats['n'] / base_test_stats['n'] * 100 if base_test_stats['n'] > 0 else 0
        delta = filt_test_stats['sharpe'] - base_test_stats['sharpe']

        print(f"  {wf_name} ({te_start} to {te_end}):", flush=True)
        print(f"    Train: {len(train_trades)} trades  |  Test: {len(test_trades)} trades", flush=True)
        print(f"    AUC: {wf_auc:.4f}", flush=True)
        print(f"    Base Sharpe: {base_test_stats['sharpe']:.3f}  |  Filt Sharpe: {filt_test_stats['sharpe']:.3f}  |  Delta: {delta:+.3f}", flush=True)
        print(f"    Kept: {filt_test_stats['n']}/{base_test_stats['n']} ({keep_pct:.1f}%)", flush=True)

        wf_results.append({
            'window': wf_name,
            'test_start': te_start, 'test_end': te_end,
            'train_n': len(train_trades), 'test_n': len(test_trades),
            'auc': round(wf_auc, 4),
            'base_sharpe': base_test_stats['sharpe'],
            'filt_sharpe': filt_test_stats['sharpe'],
            'sharpe_delta': round(delta, 3),
            'keep_pct': round(keep_pct, 1),
            'base_pnl': base_test_stats['pnl'],
            'filt_pnl': filt_test_stats['pnl'],
        })

    if wf_results:
        avg_auc = np.mean([r['auc'] for r in wf_results])
        avg_delta = np.mean([r['sharpe_delta'] for r in wf_results])
        avg_keep = np.mean([r['keep_pct'] for r in wf_results])
        wins = sum(1 for r in wf_results if r['sharpe_delta'] > 0)
        print(f"\n  Walk-Forward Summary ({len(wf_results)} windows):", flush=True)
        print(f"    Mean AUC: {avg_auc:.4f}", flush=True)
        print(f"    Mean Sharpe delta: {avg_delta:+.3f}", flush=True)
        print(f"    Mean keep %: {avg_keep:.1f}%", flush=True)
        print(f"    Windows with positive delta: {wins}/{len(wf_results)}", flush=True)

    results['phase3'] = {
        'windows': wf_results,
        'n_windows': len(wf_results),
        'mean_auc': round(avg_auc, 4) if wf_results else 0.5,
        'mean_sharpe_delta': round(avg_delta, 3) if wf_results else 0.0,
        'mean_keep_pct': round(avg_keep, 1) if wf_results else 0.0,
        'positive_delta_count': wins if wf_results else 0,
    }

    # ═══════════════════════════════════════════════════════════
    # Phase 4: K-Fold 5-fold validation
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("[Phase 4] K-Fold 5-fold cross-validation", flush=True)
    print("=" * 70, flush=True)

    kfold_results = []

    for fold_name, fold_start, fold_end in KFOLDS:
        fs = pd.Timestamp(fold_start)
        fe = pd.Timestamp(fold_end)

        test_indices = []
        train_indices = []
        for i, t in enumerate(valid_trades):
            ets = pd.Timestamp(t['entry_time'])
            if ets.tzinfo is not None:
                ets = ets.tz_localize(None)
            if fs <= ets < fe:
                test_indices.append(i)
            else:
                train_indices.append(i)

        if len(test_indices) < 10 or len(train_indices) < 30:
            print(f"  {fold_name}: SKIP (test={len(test_indices)}, train={len(train_indices)})", flush=True)
            continue

        X_tr = X_all[train_indices]
        y_tr = y_all[train_indices]
        X_te = X_all[test_indices]
        y_te = y_all[test_indices]

        scaler_kf = StandardScaler()
        X_tr_s = scaler_kf.fit_transform(X_tr)
        X_te_s = scaler_kf.transform(X_te)

        mdl = xgb.XGBClassifier(**XGB_PARAMS)
        mdl.fit(X_tr_s, y_tr)
        probs_te = mdl.predict_proba(X_te_s)[:, 1]

        try:
            fold_auc = roc_auc_score(y_te, probs_te)
        except ValueError:
            fold_auc = 0.5

        te_trades = [valid_trades[i] for i in test_indices]
        base_fold_stats = compute_stats(te_trades)

        filtered_fold = []
        for j, idx in enumerate(test_indices):
            t = valid_trades[idx]
            p = probs_te[j]
            if t['dir'] == 'BUY' and p >= best_thr:
                filtered_fold.append(t)
            elif t['dir'] == 'SELL' and p <= (1.0 - best_thr):
                filtered_fold.append(t)

        filt_fold_stats = compute_stats(filtered_fold)
        keep_pct = filt_fold_stats['n'] / base_fold_stats['n'] * 100 if base_fold_stats['n'] > 0 else 0
        delta = filt_fold_stats['sharpe'] - base_fold_stats['sharpe']

        print(f"  {fold_name} ({fold_start} to {fold_end}):", flush=True)
        print(f"    Train: {len(train_indices)}  |  Test: {len(test_indices)}", flush=True)
        print(f"    AUC: {fold_auc:.4f}", flush=True)
        print(f"    Base Sharpe: {base_fold_stats['sharpe']:.3f}  |  Filt Sharpe: {filt_fold_stats['sharpe']:.3f}  |  Delta: {delta:+.3f}", flush=True)
        print(f"    Kept: {filt_fold_stats['n']}/{base_fold_stats['n']} ({keep_pct:.1f}%)", flush=True)

        kfold_results.append({
            'fold': fold_name,
            'fold_start': fold_start, 'fold_end': fold_end,
            'train_n': len(train_indices), 'test_n': len(test_indices),
            'auc': round(fold_auc, 4),
            'base_sharpe': base_fold_stats['sharpe'],
            'filt_sharpe': filt_fold_stats['sharpe'],
            'sharpe_delta': round(delta, 3),
            'keep_pct': round(keep_pct, 1),
            'base_pnl': base_fold_stats['pnl'],
            'filt_pnl': filt_fold_stats['pnl'],
        })

    if kfold_results:
        avg_auc_kf = np.mean([r['auc'] for r in kfold_results])
        avg_delta_kf = np.mean([r['sharpe_delta'] for r in kfold_results])
        avg_keep_kf = np.mean([r['keep_pct'] for r in kfold_results])
        wins_kf = sum(1 for r in kfold_results if r['sharpe_delta'] > 0)
        print(f"\n  K-Fold Summary ({len(kfold_results)} folds):", flush=True)
        print(f"    Mean AUC: {avg_auc_kf:.4f}", flush=True)
        print(f"    Mean Sharpe delta: {avg_delta_kf:+.3f}", flush=True)
        print(f"    Mean keep %: {avg_keep_kf:.1f}%", flush=True)
        print(f"    Folds with positive delta: {wins_kf}/{len(kfold_results)}", flush=True)

    results['phase4'] = {
        'folds': kfold_results,
        'n_folds': len(kfold_results),
        'mean_auc': round(avg_auc_kf, 4) if kfold_results else 0.5,
        'mean_sharpe_delta': round(avg_delta_kf, 3) if kfold_results else 0.0,
        'mean_keep_pct': round(avg_keep_kf, 1) if kfold_results else 0.0,
        'positive_delta_count': wins_kf if kfold_results else 0,
    }

    # ═══════════════════════════════════════════════════════════
    # Phase 5: Yearly breakdown
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("[Phase 5] Yearly breakdown (2015-2025)", flush=True)
    print("=" * 70, flush=True)

    scaler_full2 = StandardScaler()
    X_scaled2 = scaler_full2.fit_transform(X_all)
    model_full2 = xgb.XGBClassifier(**XGB_PARAMS)
    model_full2.fit(X_scaled2, y_all)
    probs_full2 = model_full2.predict_proba(X_scaled2)[:, 1]

    print(f"\n  {'Year':>6} {'Base N':>7} {'Filt N':>7} {'Keep%':>7} {'Base Sharpe':>12} {'Filt Sharpe':>12} {'Delta':>8}", flush=True)
    print(f"  {'-'*65}", flush=True)

    yearly_results = []
    for year in range(2015, 2026):
        yr_start = pd.Timestamp(f"{year}-01-01")
        yr_end = pd.Timestamp(f"{year+1}-01-01")

        yr_trades = []
        yr_filtered = []
        for j, t in enumerate(valid_trades):
            ets = pd.Timestamp(t['entry_time'])
            if ets.tzinfo is not None:
                ets = ets.tz_localize(None)
            if yr_start <= ets < yr_end:
                yr_trades.append(t)
                p = probs_full2[j]
                if t['dir'] == 'BUY' and p >= best_thr:
                    yr_filtered.append(t)
                elif t['dir'] == 'SELL' and p <= (1.0 - best_thr):
                    yr_filtered.append(t)

        base_yr = compute_stats(yr_trades)
        filt_yr = compute_stats(yr_filtered)
        keep_pct = filt_yr['n'] / base_yr['n'] * 100 if base_yr['n'] > 0 else 0
        delta = filt_yr['sharpe'] - base_yr['sharpe']

        print(f"  {year:>6} {base_yr['n']:>7} {filt_yr['n']:>7} {keep_pct:>6.1f}% "
              f"{base_yr['sharpe']:>12.3f} {filt_yr['sharpe']:>12.3f} {delta:>+8.3f}", flush=True)

        yearly_results.append({
            'year': year,
            'base_n': base_yr['n'], 'filt_n': filt_yr['n'],
            'keep_pct': round(keep_pct, 1),
            'base_sharpe': base_yr['sharpe'], 'filt_sharpe': filt_yr['sharpe'],
            'sharpe_delta': round(delta, 3),
            'base_pnl': base_yr['pnl'], 'filt_pnl': filt_yr['pnl'],
        })

    results['phase5'] = {'yearly': yearly_results}

    # ═══════════════════════════════════════════════════════════
    # Phase 6: Feature importance stability
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("[Phase 6] Feature importance stability across splits", flush=True)
    print("=" * 70, flush=True)

    split_points = [0.2, 0.4, 0.5, 0.6, 0.8]
    all_top5_sets = []
    feature_freq = {}

    for sp_idx, sp in enumerate(split_points):
        cut = int(len(X_all) * sp)
        tr_end = cut
        te_start = cut
        te_end = min(cut + int(len(X_all) * 0.2), len(X_all))

        if sp <= 0.5:
            X_tr_sp = X_all[:tr_end]
            y_tr_sp = y_all[:tr_end]
        else:
            X_tr_sp = np.vstack([X_all[:cut], X_all[te_end:]])
            y_tr_sp = np.concatenate([y_all[:cut], y_all[te_end:]])

        if len(X_tr_sp) < 30:
            continue

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr_sp)
        mdl = xgb.XGBClassifier(**XGB_PARAMS)
        mdl.fit(X_tr_s, y_tr_sp)

        imp = mdl.feature_importances_
        top5_idx = np.argsort(imp)[-5:][::-1]
        top5_names = [ALL_FEATURES[i] for i in top5_idx]
        top5_set = set(top5_names)
        all_top5_sets.append(top5_set)

        for fn in top5_names:
            feature_freq[fn] = feature_freq.get(fn, 0) + 1

        print(f"  Split {sp_idx+1} (cutoff={sp:.0%}): top5 = {top5_names}", flush=True)

    if len(all_top5_sets) >= 2:
        jaccard_scores = []
        for i in range(len(all_top5_sets)):
            for j in range(i+1, len(all_top5_sets)):
                inter = len(all_top5_sets[i] & all_top5_sets[j])
                union = len(all_top5_sets[i] | all_top5_sets[j])
                jaccard_scores.append(inter / union if union > 0 else 0)
        mean_jaccard = np.mean(jaccard_scores)
    else:
        mean_jaccard = 0.0

    stable_features = sorted(feature_freq.items(), key=lambda x: -x[1])
    print(f"\n  Mean Jaccard similarity: {mean_jaccard:.3f}", flush=True)
    print(f"  Feature frequency in top-5 across splits:", flush=True)
    for fn, cnt in stable_features:
        print(f"    {fn:<15} appeared {cnt}/{len(all_top5_sets)} times", flush=True)

    results['phase6'] = {
        'mean_jaccard': round(mean_jaccard, 3),
        'feature_frequency': {fn: cnt for fn, cnt in stable_features},
        'n_splits': len(all_top5_sets),
    }

    # ═══════════════════════════════════════════════════════════
    # Phase 7: Portfolio impact assessment
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("[Phase 7] Portfolio impact — ML filter on PSAR, TSMOM, SESS_BO", flush=True)
    print("=" * 70, flush=True)

    strat_funcs = {
        'SESS_BO': lambda df: bt_sess_bo(df, SPREAD, UNIT_LOT, CAPS['SESS_BO']),
        'PSAR': lambda df: bt_psar(df, SPREAD, UNIT_LOT, CAPS['PSAR']),
        'TSMOM': lambda df: bt_tsmom(df, SPREAD, UNIT_LOT, CAPS['TSMOM']),
    }

    portfolio_results = {}
    combined_base_daily = pd.Series(dtype=float)
    combined_filt_sess_only_daily = pd.Series(dtype=float)
    combined_filt_all_daily = pd.Series(dtype=float)

    for sname, bt_func in strat_funcs.items():
        print(f"\n  Strategy: {sname}", flush=True)

        s_trades = bt_func(h1_raw)
        s_base_stats = compute_stats(s_trades)
        print(f"    Base: n={s_base_stats['n']}, Sharpe={s_base_stats['sharpe']:.3f}, "
              f"PnL=${s_base_stats['pnl']:.2f}", flush=True)

        X_s, y_s, valid_s = get_trade_features(s_trades, h1)

        if len(X_s) >= 50:
            sc_s = StandardScaler()
            X_s_scaled = sc_s.fit_transform(X_s)
            mdl_s = xgb.XGBClassifier(**XGB_PARAMS)
            mdl_s.fit(X_s_scaled, y_s)
            probs_s = mdl_s.predict_proba(X_s_scaled)[:, 1]

            filt_s_trades = []
            for j, t in enumerate(valid_s):
                p = probs_s[j]
                if t['dir'] == 'BUY' and p >= best_thr:
                    filt_s_trades.append(t)
                elif t['dir'] == 'SELL' and p <= (1.0 - best_thr):
                    filt_s_trades.append(t)

            s_filt_stats = compute_stats(filt_s_trades)
        else:
            filt_s_trades = s_trades
            s_filt_stats = s_base_stats
            print(f"    (Too few trades for ML — using base trades)", flush=True)

        keep_pct = s_filt_stats['n'] / s_base_stats['n'] * 100 if s_base_stats['n'] > 0 else 0
        delta = s_filt_stats['sharpe'] - s_base_stats['sharpe']

        print(f"    Filtered: n={s_filt_stats['n']}, Sharpe={s_filt_stats['sharpe']:.3f}, "
              f"PnL=${s_filt_stats['pnl']:.2f}", flush=True)
        print(f"    Keep: {keep_pct:.1f}%, Sharpe delta: {delta:+.3f}", flush=True)

        portfolio_results[sname] = {
            'base': s_base_stats,
            'filtered': s_filt_stats,
            'keep_pct': round(keep_pct, 1),
            'sharpe_delta': round(delta, 3),
        }

        base_daily = trades_to_daily(s_trades)
        filt_daily = trades_to_daily(filt_s_trades)

        combined_base_daily = combined_base_daily.add(base_daily, fill_value=0)
        combined_filt_all_daily = combined_filt_all_daily.add(filt_daily, fill_value=0)

        if sname == 'SESS_BO':
            combined_filt_sess_only_daily = combined_filt_sess_only_daily.add(filt_daily, fill_value=0)
        else:
            combined_filt_sess_only_daily = combined_filt_sess_only_daily.add(base_daily, fill_value=0)

    port_base_sharpe = sharpe(combined_base_daily.values) if len(combined_base_daily) > 10 else 0.0
    port_filt_all_sharpe = sharpe(combined_filt_all_daily.values) if len(combined_filt_all_daily) > 10 else 0.0
    port_filt_sess_sharpe = sharpe(combined_filt_sess_only_daily.values) if len(combined_filt_sess_only_daily) > 10 else 0.0

    print(f"\n  Combined Portfolio Impact:", flush=True)
    print(f"    Base portfolio Sharpe:              {port_base_sharpe:.3f}", flush=True)
    print(f"    ML on SESS_BO only:                 {port_filt_sess_sharpe:.3f} (delta: {port_filt_sess_sharpe - port_base_sharpe:+.3f})", flush=True)
    print(f"    ML on ALL 3 strategies:             {port_filt_all_sharpe:.3f} (delta: {port_filt_all_sharpe - port_base_sharpe:+.3f})", flush=True)
    print(f"    Base portfolio PnL:                 ${combined_base_daily.sum():.2f}", flush=True)
    print(f"    ML-SESS_BO-only portfolio PnL:      ${combined_filt_sess_only_daily.sum():.2f}", flush=True)
    print(f"    ML-all portfolio PnL:               ${combined_filt_all_daily.sum():.2f}", flush=True)

    results['phase7'] = {
        'strategies': portfolio_results,
        'portfolio_base_sharpe': round(port_base_sharpe, 3),
        'portfolio_filt_sess_only_sharpe': round(port_filt_sess_sharpe, 3),
        'portfolio_filt_all_sharpe': round(port_filt_all_sharpe, 3),
        'portfolio_base_pnl': round(float(combined_base_daily.sum()), 2),
        'portfolio_filt_sess_only_pnl': round(float(combined_filt_sess_only_daily.sum()), 2),
        'portfolio_filt_all_pnl': round(float(combined_filt_all_daily.sum()), 2),
    }

    # ═══════════════════════════════════════════════════════════
    # Summary & Save
    # ═══════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    print("\n" + "=" * 70, flush=True)
    print("[Summary] R151 ML Entry Filter Validation", flush=True)
    print("=" * 70, flush=True)
    print(f"  Total runtime: {elapsed:.1f}s", flush=True)
    print(f"  Base SESS_BO: Sharpe={base_stats['sharpe']:.3f}, n={base_stats['n']}, PnL=${base_stats['pnl']:.2f}", flush=True)

    if wf_results:
        print(f"  Walk-Forward (OOS): mean Sharpe delta={results['phase3']['mean_sharpe_delta']:+.3f}, "
              f"mean AUC={results['phase3']['mean_auc']:.4f}, "
              f"positive windows={results['phase3']['positive_delta_count']}/{len(wf_results)}", flush=True)

    if kfold_results:
        print(f"  K-Fold (OOS): mean Sharpe delta={results['phase4']['mean_sharpe_delta']:+.3f}, "
              f"mean AUC={results['phase4']['mean_auc']:.4f}, "
              f"positive folds={results['phase4']['positive_delta_count']}/{len(kfold_results)}", flush=True)

    print(f"  Feature stability Jaccard: {results['phase6']['mean_jaccard']:.3f}", flush=True)
    print(f"  Portfolio (base): {port_base_sharpe:.3f} -> (SESS_BO ML only): {port_filt_sess_sharpe:.3f} -> (all ML): {port_filt_all_sharpe:.3f}", flush=True)

    results['summary'] = {
        'runtime_sec': round(elapsed, 1),
        'base_sharpe': base_stats['sharpe'],
        'wf_mean_sharpe_delta': results.get('phase3', {}).get('mean_sharpe_delta', 0),
        'kfold_mean_sharpe_delta': results.get('phase4', {}).get('mean_sharpe_delta', 0),
        'feature_jaccard': results['phase6']['mean_jaccard'],
        'conclusion': 'RESEARCH_ONLY — review results before any deployment decision',
    }

    out_path = OUTPUT_DIR / "r151_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}", flush=True)
    print(f"\nDone. ({elapsed:.1f}s)", flush=True)


if __name__ == "__main__":
    main()

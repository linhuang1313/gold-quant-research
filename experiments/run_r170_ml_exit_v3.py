#!/usr/bin/env python3
"""
R170 — ML Exit V3: Production Walk-Forward Pipeline
====================================================
Takes the R90-D ML Exit filter concept to production quality by running
rolling walk-forward validation across all 6 P6 strategies.

Phase 1: Rolling Walk-Forward ML Exit
  - For each of 6 strategies: run backtest, build features, train monthly
    XGBoost classifier with 12-month rolling window, predict OOS
  - Report AUC, precision, recall per month; compare filtered vs unfiltered

Phase 2: Feature Stability Analysis
  - Feature importance rank correlation (Kendall tau) across months
  - Flag strategies with unstable rankings (tau < 0.3)

Phase 3: Model Decay Analysis
  - Evaluate M+1, M+3, M+6, M+12 AUC after training on month M
  - Determine "shelf life" per model

Phase 4: Ensemble Comparison
  - XGBoost vs LightGBM vs Logistic Regression vs Ensemble
  - Per-strategy OOS metrics

Phase 5: Portfolio Impact
  - P6 portfolio Sharpe/MaxDD/PnL with vs without ML Exit
  - K-Fold 6-fold validation

Expected runtime: ~30-60 minutes on a 208-core server.
"""
import sys
import os
import io
import time
import json
import warnings
import glob
from pathlib import Path
from datetime import datetime
from copy import deepcopy

import numpy as np
import pandas as pd

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import kendalltau

OUTPUT_DIR = Path("results/r170_ml_exit_v3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100
SPREAD = 0.30
UNIT_LOT = 0.01
CAPITAL = 5000

P6_LOTS = {
    'L8_MAX': 0.01, 'PSAR': 0.03, 'TSMOM': 0.04,
    'SESS_BO': 0.04, 'DUAL_THRUST': 0.04, 'CHANDELIER': 0.08,
}
P6_CAPS = {
    'L8_MAX': 35, 'PSAR': 5, 'TSMOM': 0, 'SESS_BO': 35,
    'DUAL_THRUST': 35, 'CHANDELIER': 35,
}

STRAT_ORDER = ['L8_MAX', 'PSAR', 'TSMOM', 'SESS_BO', 'DUAL_THRUST', 'CHANDELIER']

TECH_FEATURES = [
    'atr_14', 'adx_14', 'rsi_14', 'rsi_2', 'kc_breakout_strength',
    'volume_ratio', 'atr_percentile', 'ema9_ema21_cross', 'close_ema100_dist',
]
MACRO_FEATURES = [
    'real_yield_change5', 'real_yield_change20', 'vix_zscore', 'vix_close',
    'dxy_mom5', 'dxy_mom20', 'credit_stress', 'yield_curve_10y2y',
    'copper_gold_ratio', 'crude_mom5', 'usdcnh_mom5', 'cot_mm_net_zscore',
    'fed_funds_dff', 'risk_appetite_z',
]
TIME_FEATURES = ['hour_of_day', 'day_of_week', 'direction']
ALL_FEATURES = TECH_FEATURES + MACRO_FEATURES + TIME_FEATURES

TRAIN_MONTHS = 12
ML_THRESHOLD = 0.50
XGB_PARAMS = {
    'max_depth': 5, 'learning_rate': 0.05, 'n_estimators': 300,
    'min_child_weight': 10, 'subsample': 0.8, 'colsample_bytree': 0.8,
    'reg_alpha': 0.1, 'reg_lambda': 1.0, 'random_state': 42,
    'eval_metric': 'logloss', 'verbosity': 0,
}

KFOLD_FOLDS = [
    ("Fold1", "2015-01-01", "2017-01-01"),
    ("Fold2", "2017-01-01", "2019-01-01"),
    ("Fold3", "2019-01-01", "2021-01-01"),
    ("Fold4", "2021-01-01", "2023-01-01"),
    ("Fold5", "2023-01-01", "2025-01-01"),
    ("Fold6", "2025-01-01", "2026-05-01"),
]


# ═══════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════

def fmt(v):
    return f"${v:>10,.0f}" if v >= 0 else f"-${abs(v):>9,.0f}"


def compute_atr(df, period=14):
    tr = pd.DataFrame({
        'hl': df['High'] - df['Low'],
        'hc': (df['High'] - df['Close'].shift(1)).abs(),
        'lc': (df['Low'] - df['Close'].shift(1)).abs(),
    }).max(axis=1)
    return tr.rolling(period).mean()


def compute_adx(df, period=14):
    high = df['High']; low = df['Low']; close = df['Close']
    plus_dm = high.diff(); minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr = pd.DataFrame({'hl': high - low, 'hc': (high - close.shift(1)).abs(),
                        'lc': (low - close.shift(1)).abs()}).max(axis=1)
    atr_s = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr_s)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr_s)
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    return dx.rolling(period).mean()


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


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
                    ep = df['High'].iloc[i]; af = min(af + af_start, af_max)
        else:
            psar[i] = prev + af * (ep - prev)
            psar[i] = max(psar[i], df['High'].iloc[i-1], df['High'].iloc[max(0, i-2)])
            if df['High'].iloc[i] > psar[i]:
                direction[i] = 1; psar[i] = ep; ep = df['High'].iloc[i]; af = af_start
            else:
                direction[i] = -1
                if df['Low'].iloc[i] < ep:
                    ep = df['Low'].iloc[i]; af = min(af + af_start, af_max)
    df['PSAR_dir'] = direction; df['ATR'] = compute_atr(df)
    return df


def _mk(pos, exit_p, exit_time, reason, bar_idx, pnl):
    return {'dir': pos['dir'], 'entry': pos['entry'], 'exit': exit_p,
            'entry_time': pos['time'], 'exit_time': exit_time,
            'entry_bar': pos['bar'],
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


def trades_to_daily_series(trades):
    if not trades:
        return pd.Series(dtype=float)
    daily = {}
    for t in trades:
        d = pd.Timestamp(t['exit_time']).date()
        daily[d] = daily.get(d, 0) + t['pnl']
    dates = sorted(daily.keys())
    return pd.Series([daily[d] for d in dates], index=pd.DatetimeIndex(dates))


def _sharpe(arr):
    if len(arr) < 10 or np.std(arr, ddof=1) == 0:
        return 0.0
    return float(np.mean(arr) / np.std(arr, ddof=1) * np.sqrt(252))


def _max_dd(arr):
    if len(arr) == 0:
        return 0.0
    eq = np.cumsum(arr)
    return float((np.maximum.accumulate(eq) - eq).max())


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
        'wr': round(sum(1 for p in pnls if p > 0) / n * 100, 1) if n > 0 else 0,
        'max_dd': round(_max_dd(daily), 2),
    }


# ═══════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════

def load_h1():
    candidates = sorted(glob.glob("data/download/xauusd-h1-bid-2015-*.csv"))
    if not candidates:
        raise FileNotFoundError("No xauusd H1 CSV found")
    csv_path = candidates[-1]
    print(f"  Loading H1 from: {csv_path}", flush=True)
    df = pd.read_csv(csv_path)
    df['time'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_localize(None)
    df = df.set_index('time')[['open', 'high', 'low', 'close', 'volume']]
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = df[df['Close'] > 100]
    return df


def load_macro():
    macro_path = Path("data/external/aligned_daily.csv")
    if not macro_path.exists():
        print("  [WARN] Macro data not found, macro features disabled", flush=True)
        return None
    df = pd.read_csv(macro_path, parse_dates=['Date'])
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    df = df.set_index('Date')
    col_map = {
        'REAL_YIELD_Change5': 'real_yield_change5',
        'REAL_YIELD_Change20': 'real_yield_change20',
        'VIX_Zscore': 'vix_zscore',
        'VIX_Close': 'vix_close',
        'DXY_Mom5': 'dxy_mom5',
        'DXY_Mom20': 'dxy_mom20',
        'CREDIT_STRESS': 'credit_stress',
        'YIELD_CURVE_10Y2Y': 'yield_curve_10y2y',
        'COPPER_GOLD_RATIO': 'copper_gold_ratio',
        'CRUDE_Mom5': 'crude_mom5',
        'USDCNH_Mom5': 'usdcnh_mom5',
        'COT_MM_Net_Zscore': 'cot_mm_net_zscore',
        'FED_FUNDS_DFF': 'fed_funds_dff',
        'RISK_APPETITE_Z': 'risk_appetite_z',
    }
    rename = {}
    for orig, target in col_map.items():
        if orig in df.columns:
            rename[orig] = target
    df = df.rename(columns=rename)
    print(f"  Macro data: {len(df)} rows ({df.index[0].date()} -> {df.index[-1].date()})", flush=True)
    return df


# ═══════════════════════════════════════════════════════════════
# Strategy Backtests
# ═══════════════════════════════════════════════════════════════

def bt_l8_max(h1_df, spread, lot, maxloss_cap=35,
              adx_th=14, ema_period=25, kc_mult=1.2,
              sl_atr=3.5, tp_atr=8.0,
              trail_act=0.14, trail_dist=0.025, max_hold=20):
    df = h1_df.copy()
    df['ATR'] = compute_atr(df)
    df['ADX'] = compute_adx(df)
    df['EMA100'] = df['Close'].ewm(span=100, adjust=False).mean()
    df['KC_mid'] = df['Close'].ewm(span=ema_period, adjust=False).mean()
    df['KC_upper'] = df['KC_mid'] + kc_mult * df['ATR']
    df['KC_lower'] = df['KC_mid'] - kc_mult * df['ATR']
    df = df.dropna(subset=['ATR', 'ADX', 'EMA100', 'KC_upper'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; adx = df['ADX'].values
    ema100 = df['EMA100'].values
    kc_u = df['KC_upper'].values; kc_l = df['KC_lower'].values
    times = df.index; n = len(df)
    trades = []; pos = None; last_exit = -999
    for i in range(1, n):
        if pos is not None:
            result = _run_exit_with_cap(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                                        sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if i - last_exit < 2:
            continue
        if np.isnan(atr[i]) or atr[i] < 0.1 or np.isnan(adx[i]):
            continue
        if adx[i] < adx_th:
            continue
        if c[i] > kc_u[i] and c[i] > ema100[i]:
            pos = {'dir': 'BUY', 'entry': c[i] + spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif c[i] < kc_l[i] and c[i] < ema100[i]:
            pos = {'dir': 'SELL', 'entry': c[i] - spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_psar(h1_df, spread, lot, maxloss_cap=5,
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
        if i - last_exit < 2:
            continue
        if np.isnan(atr[i]) or atr[i] < 0.1:
            continue
        if pdir[i-1] == -1 and pdir[i] == 1:
            pos = {'dir': 'BUY', 'entry': c[i] + spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif pdir[i-1] == 1 and pdir[i] == -1:
            pos = {'dir': 'SELL', 'entry': c[i] - spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
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
        if c[i - fast] > 0:
            s += 0.5 * np.sign(c[i] / c[i - fast] - 1.0)
        if c[i - slow] > 0:
            s += 0.5 * np.sign(c[i] / c[i - slow] - 1.0)
        score[i] = s
    trades = []; pos = None; last_exit = -999
    for i in range(max_lb + 1, n):
        if pos is not None:
            result = _run_exit_with_cap(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                                        sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            if pos['dir'] == 'BUY' and score[i] < 0:
                pnl = (c[i] - pos['entry'] - spread) * lot * PV
                trades.append(_mk(pos, c[i], times[i], "Reversal", i, pnl)); pos = None; last_exit = i; continue
            elif pos['dir'] == 'SELL' and score[i] > 0:
                pnl = (pos['entry'] - c[i] - spread) * lot * PV
                trades.append(_mk(pos, c[i], times[i], "Reversal", i, pnl)); pos = None; last_exit = i; continue
            continue
        if i - last_exit < 2:
            continue
        if np.isnan(atr[i]) or atr[i] < 0.1:
            continue
        if np.isnan(score[i]) or np.isnan(score[i-1]):
            continue
        if score[i] > 0 and score[i-1] <= 0:
            pos = {'dir': 'BUY', 'entry': c[i] + spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif score[i] < 0 and score[i-1] >= 0:
            pos = {'dir': 'SELL', 'entry': c[i] - spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_sess_bo(h1_df, spread, lot, maxloss_cap=35,
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
        if hours[i] != session_hour:
            continue
        if i - last_exit < 2:
            continue
        if np.isnan(atr[i]) or atr[i] < 0.1:
            continue
        hh = max(h[i - j] for j in range(1, lookback + 1))
        ll = min(lo[i - j] for j in range(1, lookback + 1))
        if c[i] > hh:
            pos = {'dir': 'BUY', 'entry': c[i] + spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif c[i] < ll:
            pos = {'dir': 'SELL', 'entry': c[i] - spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_dual_thrust(h1_df, spread, lot, maxloss_cap=35,
                   n_bars=6, k=0.5, sl_atr=4.5, tp_atr=8.0,
                   trail_act=0.14, trail_dist=0.025, max_hold=20):
    df = h1_df.copy(); df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    o = df['Open'].values; atr = df['ATR'].values
    times = df.index; n = len(df)
    trades = []; pos = None; last_exit = -999
    for i in range(n_bars, n):
        if pos is not None:
            result = _run_exit_with_cap(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                                        sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if i - last_exit < 2:
            continue
        if np.isnan(atr[i]) or atr[i] < 0.1:
            continue
        hh = np.max(h[i - n_bars:i])
        lc = np.min(c[i - n_bars:i])
        hc = np.max(c[i - n_bars:i])
        ll = np.min(lo[i - n_bars:i])
        rng = max(hh - lc, hc - ll)
        buy_line = o[i] + k * rng
        sell_line = o[i] - k * rng
        if c[i] > buy_line:
            pos = {'dir': 'BUY', 'entry': c[i] + spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif c[i] < sell_line:
            pos = {'dir': 'SELL', 'entry': c[i] - spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_chandelier(h1_df, spread, lot, maxloss_cap=35,
                  period=22, mult=3.0, ema_period=100,
                  sl_atr=4.5, tp_atr=8.0,
                  trail_act=0.14, trail_dist=0.025, max_hold=20):
    df = h1_df.copy()
    df['ATR'] = compute_atr(df, period=period)
    df['EMA'] = df['Close'].ewm(span=ema_period, adjust=False).mean()
    df = df.dropna(subset=['ATR', 'EMA'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; ema = df['EMA'].values
    times = df.index; n = len(df)
    chandelier_long = np.full(n, np.nan)
    chandelier_short = np.full(n, np.nan)
    for i in range(period, n):
        hh = np.max(h[i - period + 1:i + 1])
        ll = np.min(lo[i - period + 1:i + 1])
        chandelier_long[i] = hh - mult * atr[i]
        chandelier_short[i] = ll + mult * atr[i]
    direction = np.zeros(n)
    for i in range(period + 1, n):
        if np.isnan(chandelier_long[i]) or np.isnan(chandelier_short[i]):
            direction[i] = direction[i-1]; continue
        if c[i] > chandelier_short[i-1]:
            direction[i] = 1
        elif c[i] < chandelier_long[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
    trades = []; pos = None; last_exit = -999
    for i in range(period + 2, n):
        if pos is not None:
            result = _run_exit_with_cap(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                                        sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if i - last_exit < 2:
            continue
        if np.isnan(atr[i]) or atr[i] < 0.1:
            continue
        flipped_long = direction[i] == 1 and direction[i-1] != 1
        flipped_short = direction[i] == -1 and direction[i-1] != -1
        if flipped_long and c[i] > ema[i]:
            pos = {'dir': 'BUY', 'entry': c[i] + spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif flipped_short and c[i] < ema[i]:
            pos = {'dir': 'SELL', 'entry': c[i] - spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


STRATEGY_BT_MAP = {
    'L8_MAX':      (bt_l8_max,       {}),
    'PSAR':        (bt_psar,         {}),
    'TSMOM':       (bt_tsmom,        {}),
    'SESS_BO':     (bt_sess_bo,      {}),
    'DUAL_THRUST': (bt_dual_thrust,  {}),
    'CHANDELIER':  (bt_chandelier,   {}),
}


# ═══════════════════════════════════════════════════════════════
# Feature Engineering
# ═══════════════════════════════════════════════════════════════

def build_h1_indicators(h1_df):
    """Add all technical indicator columns needed for feature building."""
    df = h1_df.copy()
    df['atr_14'] = compute_atr(df, 14)
    df['adx_14'] = compute_adx(df, 14)
    df['rsi_14'] = compute_rsi(df['Close'], 14)
    df['rsi_2'] = compute_rsi(df['Close'], 2)
    df['EMA9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['EMA100'] = df['Close'].ewm(span=100, adjust=False).mean()
    kc_mid = df['Close'].ewm(span=25, adjust=False).mean()
    df['KC_upper'] = kc_mid + 1.2 * df['atr_14']
    df['KC_lower'] = kc_mid - 1.2 * df['atr_14']
    kc_bw = df['KC_upper'] - df['KC_lower']
    df['kc_breakout_strength'] = (df['Close'] - kc_mid) / kc_bw.replace(0, np.nan)
    vol_ma = df['Volume'].rolling(20).mean()
    df['volume_ratio'] = df['Volume'] / vol_ma.replace(0, np.nan)
    df['atr_percentile'] = df['atr_14'].rolling(500, min_periods=50).rank(pct=True)
    df['ema9_ema21_cross'] = (df['EMA9'] - df['EMA21']) / df['atr_14'].replace(0, np.nan)
    df['close_ema100_dist'] = (df['Close'] - df['EMA100']) / df['atr_14'].replace(0, np.nan)
    return df


def build_trade_features(trades, h1_ind, macro_df):
    """Build feature matrix for each trade using entry-time indicators."""
    records = []
    for t in trades:
        entry_time = pd.Timestamp(t['entry_time'])
        if entry_time.tzinfo is not None:
            entry_time = entry_time.tz_localize(None)

        h1_idx = h1_ind.index.searchsorted(entry_time, side='right') - 1
        if h1_idx < 0 or h1_idx >= len(h1_ind):
            continue

        row = h1_ind.iloc[h1_idx]
        feat = {}

        for f in TECH_FEATURES:
            feat[f] = float(row.get(f, np.nan)) if f in row.index else np.nan

        feat['hour_of_day'] = entry_time.hour
        feat['day_of_week'] = entry_time.weekday()
        feat['direction'] = 1.0 if t['dir'] == 'BUY' else 0.0

        if macro_df is not None:
            entry_date = entry_time.normalize()
            m_idx = macro_df.index.searchsorted(entry_date, side='right') - 1
            if 0 <= m_idx < len(macro_df):
                mrow = macro_df.iloc[m_idx]
                for f in MACRO_FEATURES:
                    feat[f] = float(mrow[f]) if f in mrow.index and not pd.isna(mrow.get(f, np.nan)) else np.nan
            else:
                for f in MACRO_FEATURES:
                    feat[f] = np.nan

        feat['label'] = 1 if t['pnl'] > 0 else 0
        feat['pnl'] = t['pnl']
        feat['entry_time'] = entry_time
        feat['exit_time'] = t['exit_time']
        records.append(feat)

    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df['entry_month'] = df['entry_time'].dt.to_period('M')
    return df


# ═══════════════════════════════════════════════════════════════
# ML Model Helpers
# ═══════════════════════════════════════════════════════════════

def train_and_predict(X_train, y_train, X_test, model_type='xgb'):
    """Train a classifier and return OOS probabilities."""
    if model_type == 'xgb' and HAS_XGB:
        model = XGBClassifier(**XGB_PARAMS, use_label_encoder=False)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        importances = model.feature_importances_
    elif model_type == 'lgb' and HAS_LGB:
        model = LGBMClassifier(
            max_depth=5, learning_rate=0.05, n_estimators=300,
            min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1,
        )
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        importances = model.feature_importances_
    elif model_type == 'lr':
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)
        X_te_s = scaler.transform(X_test)
        model = LogisticRegression(max_iter=1000, C=0.1, random_state=42)
        model.fit(X_tr_s, y_train)
        proba = model.predict_proba(X_te_s)[:, 1]
        importances = np.abs(model.coef_[0])
    else:
        return None, None
    return proba, importances


def evaluate_predictions(y_true, y_proba, threshold=0.5):
    """Calculate AUC, precision, recall from probabilities."""
    if len(y_true) < 5 or len(np.unique(y_true)) < 2:
        return {'auc': np.nan, 'precision': np.nan, 'recall': np.nan, 'n': len(y_true)}
    auc = roc_auc_score(y_true, y_proba)
    y_pred = (y_proba >= threshold).astype(int)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    return {'auc': round(auc, 4), 'precision': round(prec, 4), 'recall': round(rec, 4), 'n': len(y_true)}


# ═══════════════════════════════════════════════════════════════
# Phase 1: Rolling Walk-Forward ML Exit
# ═══════════════════════════════════════════════════════════════

def run_phase1(h1_df, macro_df, all_results):
    print(f"\n{'='*80}", flush=True)
    print(f"  Phase 1: Rolling Walk-Forward ML Exit", flush=True)
    print(f"{'='*80}", flush=True)

    if not HAS_XGB:
        print("  [SKIP] XGBoost not installed. Phase 1 requires xgboost.", flush=True)
        return {}

    t_phase = time.time()
    h1_ind = build_h1_indicators(h1_df)
    print(f"  H1 indicators built: {len(h1_ind)} bars", flush=True)

    phase1_results = {}

    for strat_name in STRAT_ORDER:
        print(f"\n  --- {strat_name} ---", flush=True)
        t_strat = time.time()

        bt_fn, bt_kw = STRATEGY_BT_MAP[strat_name]
        cap = P6_CAPS[strat_name]
        trades = bt_fn(h1_df, spread=SPREAD, lot=UNIT_LOT, maxloss_cap=cap, **bt_kw)
        unfiltered_stats = _compute_stats(trades)
        print(f"    Unfiltered: {unfiltered_stats['n']} trades, Sharpe={unfiltered_stats['sharpe']:.3f}, "
              f"PnL={fmt(unfiltered_stats['pnl'])}", flush=True)

        feat_df = build_trade_features(trades, h1_ind, macro_df)
        if len(feat_df) < 30:
            print(f"    [SKIP] Too few trades for ML ({len(feat_df)})", flush=True)
            phase1_results[strat_name] = {'status': 'skipped', 'reason': 'too_few_trades'}
            continue

        months = sorted(feat_df['entry_month'].unique())
        if len(months) < TRAIN_MONTHS + 2:
            print(f"    [SKIP] Not enough monthly data ({len(months)} months)", flush=True)
            phase1_results[strat_name] = {'status': 'skipped', 'reason': 'not_enough_months'}
            continue

        feature_cols = [c for c in ALL_FEATURES if c in feat_df.columns]
        monthly_metrics = []
        monthly_importances = []
        filtered_trades_all = []

        for m_idx in range(TRAIN_MONTHS, len(months)):
            test_month = months[m_idx]
            train_months_range = months[m_idx - TRAIN_MONTHS:m_idx]

            train_mask = feat_df['entry_month'].isin(train_months_range)
            test_mask = feat_df['entry_month'] == test_month

            train_df = feat_df[train_mask].copy()
            test_df = feat_df[test_mask].copy()

            if len(train_df) < 20 or len(test_df) < 3:
                continue

            X_train = train_df[feature_cols].fillna(0).values
            y_train = train_df['label'].values
            X_test = test_df[feature_cols].fillna(0).values
            y_test = test_df['label'].values

            if len(np.unique(y_train)) < 2:
                continue

            proba, importances = train_and_predict(X_train, y_train, X_test, model_type='xgb')
            if proba is None:
                continue

            metrics = evaluate_predictions(y_test, proba, threshold=ML_THRESHOLD)
            metrics['month'] = str(test_month)
            metrics['n_train'] = len(train_df)
            monthly_metrics.append(metrics)

            if importances is not None:
                imp_dict = {feature_cols[j]: float(importances[j]) for j in range(len(feature_cols))}
                imp_dict['month'] = str(test_month)
                monthly_importances.append(imp_dict)

            for idx_t, (_, row) in enumerate(test_df.iterrows()):
                if proba[idx_t] >= ML_THRESHOLD:
                    trade_idx = row.name
                    filtered_trades_all.append({
                        'pnl': row['pnl'],
                        'exit_time': row['exit_time'],
                        'proba': float(proba[idx_t]),
                    })

        if not monthly_metrics:
            print(f"    [SKIP] No valid monthly predictions", flush=True)
            phase1_results[strat_name] = {'status': 'skipped', 'reason': 'no_valid_months'}
            continue

        aucs = [m['auc'] for m in monthly_metrics if not np.isnan(m['auc'])]
        mean_auc = np.mean(aucs) if aucs else 0
        filtered_daily = _trades_to_daily(filtered_trades_all) if filtered_trades_all else np.array([0.0])
        filtered_stats = {
            'n': len(filtered_trades_all),
            'sharpe': round(_sharpe(filtered_daily), 3),
            'pnl': round(sum(t['pnl'] for t in filtered_trades_all), 2),
            'max_dd': round(_max_dd(filtered_daily), 2),
        }

        sharpe_delta = filtered_stats['sharpe'] - unfiltered_stats['sharpe']
        print(f"    OOS AUC: {mean_auc:.3f} (over {len(aucs)} months)", flush=True)
        print(f"    Filtered: {filtered_stats['n']} trades, Sharpe={filtered_stats['sharpe']:.3f}, "
              f"PnL={fmt(filtered_stats['pnl'])}", flush=True)
        print(f"    Sharpe delta: {sharpe_delta:+.3f} ({sharpe_delta/max(abs(unfiltered_stats['sharpe']),0.01)*100:+.1f}%)", flush=True)
        print(f"    Elapsed: {time.time()-t_strat:.1f}s", flush=True)

        phase1_results[strat_name] = {
            'status': 'ok',
            'unfiltered': unfiltered_stats,
            'filtered': filtered_stats,
            'mean_auc': round(mean_auc, 4),
            'sharpe_delta': round(sharpe_delta, 3),
            'monthly_metrics': monthly_metrics,
            'monthly_importances': monthly_importances,
            'n_oos_months': len(monthly_metrics),
        }

    elapsed = time.time() - t_phase
    print(f"\n  Phase 1 complete: {elapsed:.1f}s", flush=True)
    all_results['phase1'] = phase1_results
    _save_intermediate('phase1', phase1_results)
    return phase1_results


# ═══════════════════════════════════════════════════════════════
# Phase 2: Feature Stability Analysis
# ═══════════════════════════════════════════════════════════════

def run_phase2(phase1_results, all_results):
    print(f"\n{'='*80}", flush=True)
    print(f"  Phase 2: Feature Stability Analysis", flush=True)
    print(f"{'='*80}", flush=True)

    t_phase = time.time()
    phase2_results = {}

    for strat_name in STRAT_ORDER:
        p1 = phase1_results.get(strat_name, {})
        if p1.get('status') != 'ok':
            continue

        imps = p1.get('monthly_importances', [])
        if len(imps) < 3:
            print(f"  {strat_name}: Too few months for stability analysis ({len(imps)})", flush=True)
            continue

        feature_names = [k for k in imps[0].keys() if k != 'month']
        n_months = len(imps)

        rank_matrix = np.zeros((n_months, len(feature_names)))
        for m_i, imp in enumerate(imps):
            vals = [imp.get(f, 0) for f in feature_names]
            ranks = np.argsort(np.argsort(-np.array(vals))) + 1
            rank_matrix[m_i] = ranks

        taus = []
        for m_i in range(n_months - 1):
            tau, _ = kendalltau(rank_matrix[m_i], rank_matrix[m_i + 1])
            if not np.isnan(tau):
                taus.append(tau)
        mean_tau = np.mean(taus) if taus else 0

        avg_importance = {}
        for f_i, f_name in enumerate(feature_names):
            vals = [imp.get(f_name, 0) for imp in imps]
            avg_importance[f_name] = float(np.mean(vals))
        top_10 = sorted(avg_importance.items(), key=lambda x: -x[1])[:10]

        stable = mean_tau >= 0.3
        status = "STABLE" if stable else "UNSTABLE"
        print(f"\n  {strat_name}: Kendall tau = {mean_tau:.3f} [{status}]", flush=True)
        print(f"    Top-10 features:", flush=True)
        print(f"    {'Rank':<5} {'Feature':<25} {'Avg Importance':>15}", flush=True)
        print(f"    {'-'*5} {'-'*25} {'-'*15}", flush=True)
        for rank, (fname, fval) in enumerate(top_10, 1):
            print(f"    {rank:<5} {fname:<25} {fval:>15.4f}", flush=True)

        phase2_results[strat_name] = {
            'mean_kendall_tau': round(mean_tau, 4),
            'stable': stable,
            'top_10_features': [{k: v} for k, v in top_10],
            'pairwise_taus': [round(t, 4) for t in taus],
            'n_months': n_months,
        }

    elapsed = time.time() - t_phase
    print(f"\n  Phase 2 complete: {elapsed:.1f}s", flush=True)
    all_results['phase2'] = phase2_results
    _save_intermediate('phase2', phase2_results)
    return phase2_results


# ═══════════════════════════════════════════════════════════════
# Phase 3: Model Decay Analysis
# ═══════════════════════════════════════════════════════════════

def run_phase3(h1_df, macro_df, all_results):
    print(f"\n{'='*80}", flush=True)
    print(f"  Phase 3: Model Decay Analysis", flush=True)
    print(f"{'='*80}", flush=True)

    if not HAS_XGB:
        print("  [SKIP] XGBoost not installed.", flush=True)
        return {}

    t_phase = time.time()
    h1_ind = build_h1_indicators(h1_df)
    phase3_results = {}
    decay_horizons = [1, 3, 6, 12]

    for strat_name in STRAT_ORDER:
        print(f"\n  --- {strat_name} ---", flush=True)
        t_strat = time.time()

        bt_fn, bt_kw = STRATEGY_BT_MAP[strat_name]
        cap = P6_CAPS[strat_name]
        trades = bt_fn(h1_df, spread=SPREAD, lot=UNIT_LOT, maxloss_cap=cap, **bt_kw)
        feat_df = build_trade_features(trades, h1_ind, macro_df)

        if len(feat_df) < 30:
            print(f"    [SKIP] Too few trades", flush=True)
            continue

        months = sorted(feat_df['entry_month'].unique())
        feature_cols = [c for c in ALL_FEATURES if c in feat_df.columns]

        decay_curves = {h: [] for h in decay_horizons}
        n_models = 0

        for m_idx in range(TRAIN_MONTHS, len(months)):
            train_months_range = months[m_idx - TRAIN_MONTHS:m_idx]
            train_mask = feat_df['entry_month'].isin(train_months_range)
            train_df = feat_df[train_mask].copy()

            if len(train_df) < 20 or len(np.unique(train_df['label'])) < 2:
                continue

            X_train = train_df[feature_cols].fillna(0).values
            y_train = train_df['label'].values

            model = XGBClassifier(**XGB_PARAMS, use_label_encoder=False)
            model.fit(X_train, y_train)
            n_models += 1

            for horizon in decay_horizons:
                test_month_idx = m_idx + horizon - 1
                if test_month_idx >= len(months):
                    continue

                test_mask = feat_df['entry_month'] == months[test_month_idx]
                test_df = feat_df[test_mask]

                if len(test_df) < 3 or len(np.unique(test_df['label'])) < 2:
                    continue

                X_test = test_df[feature_cols].fillna(0).values
                y_test = test_df['label'].values
                proba = model.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, proba)
                decay_curves[horizon].append(round(auc, 4))

        shelf_life = {}
        print(f"    Decay analysis ({n_models} models trained):", flush=True)
        print(f"    {'Horizon':>8} {'Mean AUC':>10} {'Std':>8} {'N':>5}", flush=True)
        print(f"    {'-'*8} {'-'*10} {'-'*8} {'-'*5}", flush=True)
        for horizon in decay_horizons:
            vals = decay_curves[horizon]
            if vals:
                mean_auc = np.mean(vals)
                std_auc = np.std(vals)
                print(f"    M+{horizon:<5} {mean_auc:>10.4f} {std_auc:>8.4f} {len(vals):>5}", flush=True)
                shelf_life[f"M+{horizon}"] = round(mean_auc, 4)
            else:
                print(f"    M+{horizon:<5} {'N/A':>10}", flush=True)

        first_below = None
        for horizon in decay_horizons:
            vals = decay_curves[horizon]
            if vals and np.mean(vals) < 0.55:
                first_below = horizon
                break

        if first_below:
            print(f"    Shelf life: AUC drops below 0.55 at M+{first_below}", flush=True)
        else:
            print(f"    Shelf life: AUC stays above 0.55 through M+{decay_horizons[-1]}", flush=True)

        phase3_results[strat_name] = {
            'decay_curves': {f"M+{h}": v for h, v in decay_curves.items()},
            'mean_aucs': shelf_life,
            'shelf_life_months': first_below if first_below else f">{decay_horizons[-1]}",
            'n_models': n_models,
        }
        print(f"    Elapsed: {time.time()-t_strat:.1f}s", flush=True)

    elapsed = time.time() - t_phase
    print(f"\n  Phase 3 complete: {elapsed:.1f}s", flush=True)
    all_results['phase3'] = phase3_results
    _save_intermediate('phase3', phase3_results)
    return phase3_results


# ═══════════════════════════════════════════════════════════════
# Phase 4: Ensemble Comparison
# ═══════════════════════════════════════════════════════════════

def run_phase4(h1_df, macro_df, all_results):
    print(f"\n{'='*80}", flush=True)
    print(f"  Phase 4: Ensemble Comparison (XGB vs LGB vs LR vs Ensemble)", flush=True)
    print(f"{'='*80}", flush=True)

    t_phase = time.time()
    h1_ind = build_h1_indicators(h1_df)
    phase4_results = {}

    model_types = []
    if HAS_XGB:
        model_types.append('xgb')
    if HAS_LGB:
        model_types.append('lgb')
    model_types.append('lr')

    if len(model_types) < 2:
        print(f"  Only {len(model_types)} model type(s) available, need at least 2 for comparison", flush=True)

    for strat_name in STRAT_ORDER:
        print(f"\n  --- {strat_name} ---", flush=True)
        t_strat = time.time()

        bt_fn, bt_kw = STRATEGY_BT_MAP[strat_name]
        cap = P6_CAPS[strat_name]
        trades = bt_fn(h1_df, spread=SPREAD, lot=UNIT_LOT, maxloss_cap=cap, **bt_kw)
        feat_df = build_trade_features(trades, h1_ind, macro_df)

        if len(feat_df) < 30:
            print(f"    [SKIP] Too few trades", flush=True)
            continue

        months = sorted(feat_df['entry_month'].unique())
        feature_cols = [c for c in ALL_FEATURES if c in feat_df.columns]

        if len(months) < TRAIN_MONTHS + 2:
            print(f"    [SKIP] Not enough months", flush=True)
            continue

        model_monthly_aucs = {mt: [] for mt in model_types}
        ensemble_aucs = []

        for m_idx in range(TRAIN_MONTHS, len(months)):
            train_months_range = months[m_idx - TRAIN_MONTHS:m_idx]
            test_month = months[m_idx]

            train_mask = feat_df['entry_month'].isin(train_months_range)
            test_mask = feat_df['entry_month'] == test_month
            train_df = feat_df[train_mask]; test_df = feat_df[test_mask]

            if len(train_df) < 20 or len(test_df) < 3:
                continue
            X_train = train_df[feature_cols].fillna(0).values
            y_train = train_df['label'].values
            X_test = test_df[feature_cols].fillna(0).values
            y_test = test_df['label'].values

            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                continue

            all_probas = []
            for mt in model_types:
                proba, _ = train_and_predict(X_train, y_train, X_test, model_type=mt)
                if proba is not None:
                    auc = roc_auc_score(y_test, proba)
                    model_monthly_aucs[mt].append(round(auc, 4))
                    all_probas.append(proba)

            if len(all_probas) >= 2:
                ensemble_proba = np.mean(all_probas, axis=0)
                ens_auc = roc_auc_score(y_test, ensemble_proba)
                ensemble_aucs.append(round(ens_auc, 4))

        print(f"    {'Model':<12} {'Mean AUC':>10} {'Std':>8} {'N':>5}", flush=True)
        print(f"    {'-'*12} {'-'*10} {'-'*8} {'-'*5}", flush=True)

        strat_results = {}
        for mt in model_types:
            vals = model_monthly_aucs[mt]
            if vals:
                mean_a = np.mean(vals)
                std_a = np.std(vals)
                print(f"    {mt.upper():<12} {mean_a:>10.4f} {std_a:>8.4f} {len(vals):>5}", flush=True)
                strat_results[mt] = {'mean_auc': round(mean_a, 4), 'std_auc': round(std_a, 4), 'n': len(vals)}
        if ensemble_aucs:
            mean_e = np.mean(ensemble_aucs)
            std_e = np.std(ensemble_aucs)
            print(f"    {'ENSEMBLE':<12} {mean_e:>10.4f} {std_e:>8.4f} {len(ensemble_aucs):>5}", flush=True)
            strat_results['ensemble'] = {'mean_auc': round(mean_e, 4), 'std_auc': round(std_e, 4), 'n': len(ensemble_aucs)}

        best = max(strat_results.items(), key=lambda x: x[1].get('mean_auc', 0))
        print(f"    Best: {best[0].upper()} (AUC={best[1]['mean_auc']:.4f})", flush=True)
        strat_results['best'] = best[0]
        phase4_results[strat_name] = strat_results
        print(f"    Elapsed: {time.time()-t_strat:.1f}s", flush=True)

    elapsed = time.time() - t_phase
    print(f"\n  Phase 4 complete: {elapsed:.1f}s", flush=True)
    all_results['phase4'] = phase4_results
    _save_intermediate('phase4', phase4_results)
    return phase4_results


# ═══════════════════════════════════════════════════════════════
# Phase 5: Portfolio Impact
# ═══════════════════════════════════════════════════════════════

def run_phase5(h1_df, macro_df, phase1_results, all_results):
    print(f"\n{'='*80}", flush=True)
    print(f"  Phase 5: Portfolio Impact — P6 with vs without ML Exit", flush=True)
    print(f"{'='*80}", flush=True)

    if not HAS_XGB:
        print("  [SKIP] XGBoost not installed.", flush=True)
        return {}

    t_phase = time.time()
    h1_ind = build_h1_indicators(h1_df)

    unfiltered_dailies = {}
    filtered_dailies = {}

    for strat_name in STRAT_ORDER:
        print(f"\n  --- {strat_name} (lot={P6_LOTS[strat_name]}) ---", flush=True)
        lot = P6_LOTS[strat_name]
        cap = P6_CAPS[strat_name]

        bt_fn, bt_kw = STRATEGY_BT_MAP[strat_name]
        trades = bt_fn(h1_df, spread=SPREAD, lot=lot, maxloss_cap=cap, **bt_kw)

        unfiltered_dailies[strat_name] = trades_to_daily_series(trades)
        uf_stats = _compute_stats(trades)
        print(f"    Unfiltered: N={uf_stats['n']}, Sharpe={uf_stats['sharpe']:.3f}, "
              f"PnL={fmt(uf_stats['pnl'])}", flush=True)

        feat_df = build_trade_features(trades, h1_ind, macro_df)
        if len(feat_df) < 30:
            print(f"    [SKIP ML] Too few trades, using unfiltered", flush=True)
            filtered_dailies[strat_name] = unfiltered_dailies[strat_name]
            continue

        months = sorted(feat_df['entry_month'].unique())
        feature_cols = [c for c in ALL_FEATURES if c in feat_df.columns]

        if len(months) < TRAIN_MONTHS + 2:
            filtered_dailies[strat_name] = unfiltered_dailies[strat_name]
            continue

        filtered_trades = []
        for m_idx in range(TRAIN_MONTHS, len(months)):
            train_months_range = months[m_idx - TRAIN_MONTHS:m_idx]
            test_month = months[m_idx]

            train_mask = feat_df['entry_month'].isin(train_months_range)
            test_mask = feat_df['entry_month'] == test_month
            train_df = feat_df[train_mask]; test_df = feat_df[test_mask]

            if len(train_df) < 20 or len(test_df) < 2:
                for _, row in test_df.iterrows():
                    filtered_trades.append({'pnl': row['pnl'], 'exit_time': row['exit_time']})
                continue

            X_train = train_df[feature_cols].fillna(0).values
            y_train = train_df['label'].values
            X_test = test_df[feature_cols].fillna(0).values

            if len(np.unique(y_train)) < 2:
                for _, row in test_df.iterrows():
                    filtered_trades.append({'pnl': row['pnl'], 'exit_time': row['exit_time']})
                continue

            proba, _ = train_and_predict(X_train, y_train, X_test, model_type='xgb')
            if proba is None:
                for _, row in test_df.iterrows():
                    filtered_trades.append({'pnl': row['pnl'], 'exit_time': row['exit_time']})
                continue

            for idx_t, (_, row) in enumerate(test_df.iterrows()):
                if proba[idx_t] >= ML_THRESHOLD:
                    filtered_trades.append({'pnl': row['pnl'], 'exit_time': row['exit_time']})

        filtered_dailies[strat_name] = trades_to_daily_series(filtered_trades)
        f_stats = _compute_stats(filtered_trades)
        print(f"    Filtered:   N={f_stats['n']}, Sharpe={f_stats['sharpe']:.3f}, "
              f"PnL={fmt(f_stats['pnl'])}", flush=True)

    # Combine portfolios
    print(f"\n  --- Portfolio Comparison ---", flush=True)

    def combine_daily_series(dailies_dict):
        all_dates = set()
        for ds in dailies_dict.values():
            all_dates.update(ds.index)
        all_dates = sorted(all_dates)
        idx = pd.DatetimeIndex(all_dates)
        portfolio = np.zeros(len(idx))
        for name in STRAT_ORDER:
            if name in dailies_dict:
                aligned = dailies_dict[name].reindex(idx, fill_value=0.0).values
                portfolio += aligned
        return portfolio

    uf_port = combine_daily_series(unfiltered_dailies)
    f_port = combine_daily_series(filtered_dailies)

    uf_sharpe = _sharpe(uf_port)
    uf_pnl = float(np.sum(uf_port))
    uf_dd = _max_dd(uf_port)
    f_sharpe = _sharpe(f_port)
    f_pnl = float(np.sum(f_port))
    f_dd = _max_dd(f_port)

    print(f"\n    {'Metric':<20} {'Unfiltered':>15} {'ML Filtered':>15} {'Delta':>10}", flush=True)
    print(f"    {'-'*20} {'-'*15} {'-'*15} {'-'*10}", flush=True)
    print(f"    {'Sharpe':<20} {uf_sharpe:>15.3f} {f_sharpe:>15.3f} {f_sharpe-uf_sharpe:>+10.3f}", flush=True)
    print(f"    {'PnL ($)':<20} {fmt(uf_pnl):>15} {fmt(f_pnl):>15} {fmt(f_pnl-uf_pnl):>10}", flush=True)
    print(f"    {'MaxDD ($)':<20} {fmt(uf_dd):>15} {fmt(f_dd):>15} {fmt(f_dd-uf_dd):>10}", flush=True)

    # K-Fold validation
    print(f"\n  --- K-Fold Validation (6 folds) ---", flush=True)
    kfold_results = []

    for fold_name, f_start, f_end in KFOLD_FOLDS:
        ts_start = pd.Timestamp(f_start)
        ts_end = pd.Timestamp(f_end)
        fold_h1 = h1_df[(h1_df.index >= ts_start) & (h1_df.index < ts_end)]

        if len(fold_h1) < 500:
            print(f"    {fold_name}: [SKIP] too few bars ({len(fold_h1)})", flush=True)
            continue

        fold_uf_dailies = {}
        fold_f_dailies = {}

        for sn in STRAT_ORDER:
            bt_fn, bt_kw = STRATEGY_BT_MAP[sn]
            lot = P6_LOTS[sn]; cap = P6_CAPS[sn]
            trades = bt_fn(fold_h1, spread=SPREAD, lot=lot, maxloss_cap=cap, **bt_kw)
            fold_uf_dailies[sn] = trades_to_daily_series(trades)
            fold_f_dailies[sn] = fold_uf_dailies[sn]

        fold_uf_port = combine_daily_series(fold_uf_dailies)
        fold_f_port = combine_daily_series(fold_f_dailies)

        uf_s = _sharpe(fold_uf_port)
        f_s = _sharpe(fold_f_port)
        uf_p = float(np.sum(fold_uf_port))

        kfold_results.append({
            'fold': fold_name,
            'start': f_start,
            'end': f_end,
            'n_bars': len(fold_h1),
            'unfiltered_sharpe': round(uf_s, 3),
            'unfiltered_pnl': round(uf_p, 2),
        })
        print(f"    {fold_name} ({f_start} -> {f_end}): "
              f"Sharpe={uf_s:.3f}, PnL={fmt(uf_p)}", flush=True)

    phase5_results = {
        'unfiltered': {
            'sharpe': round(uf_sharpe, 3),
            'pnl': round(uf_pnl, 2),
            'max_dd': round(uf_dd, 2),
        },
        'filtered': {
            'sharpe': round(f_sharpe, 3),
            'pnl': round(f_pnl, 2),
            'max_dd': round(f_dd, 2),
        },
        'sharpe_improvement_pct': round((f_sharpe - uf_sharpe) / max(abs(uf_sharpe), 0.01) * 100, 1),
        'kfold': kfold_results,
    }

    elapsed = time.time() - t_phase
    print(f"\n  Phase 5 complete: {elapsed:.1f}s", flush=True)
    all_results['phase5'] = phase5_results
    _save_intermediate('phase5', phase5_results)
    return phase5_results


# ═══════════════════════════════════════════════════════════════
# Serialization & main
# ═══════════════════════════════════════════════════════════════

def _sanitize(obj):
    """Make any object JSON-serializable."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp, datetime)):
        return str(obj)
    if isinstance(obj, pd.Period):
        return str(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def _save_intermediate(phase_name, data):
    path = OUTPUT_DIR / f"{phase_name}.json"
    try:
        with open(path, 'w') as f:
            json.dump(_sanitize(data), f, indent=2, default=str)
        print(f"  [SAVED] {path}", flush=True)
    except Exception as e:
        print(f"  [WARN] Failed to save {path}: {e}", flush=True)


def main():
    t0 = time.time()
    print("=" * 80, flush=True)
    print("  R170 — ML Exit V3: Production Walk-Forward Pipeline", flush=True)
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"  Output: {OUTPUT_DIR}", flush=True)
    print(f"  XGBoost: {'YES' if HAS_XGB else 'NO'}", flush=True)
    print(f"  LightGBM: {'YES' if HAS_LGB else 'NO'}", flush=True)
    print(f"  Train window: {TRAIN_MONTHS} months rolling", flush=True)
    print(f"  ML threshold: {ML_THRESHOLD}", flush=True)
    print("=" * 80, flush=True)

    all_results = {
        'experiment': 'R170_ML_Exit_V3',
        'started': datetime.now().isoformat(),
        'config': {
            'train_months': TRAIN_MONTHS,
            'ml_threshold': ML_THRESHOLD,
            'xgb_params': XGB_PARAMS,
            'p6_lots': P6_LOTS,
            'p6_caps': P6_CAPS,
            'features': ALL_FEATURES,
            'has_xgb': HAS_XGB,
            'has_lgb': HAS_LGB,
        },
    }

    # Load data
    print(f"\n  Loading data...", flush=True)
    h1_df = load_h1()
    print(f"  H1: {len(h1_df)} bars ({h1_df.index[0]} ~ {h1_df.index[-1]})", flush=True)

    macro_df = load_macro()

    # Phase 1: Rolling Walk-Forward ML Exit
    try:
        phase1_results = run_phase1(h1_df, macro_df, all_results)
    except Exception as e:
        print(f"\n  [ERROR] Phase 1 failed: {e}", flush=True)
        import traceback; traceback.print_exc()
        phase1_results = {}
        all_results['phase1'] = {'error': str(e)}

    # Phase 2: Feature Stability Analysis
    try:
        run_phase2(phase1_results, all_results)
    except Exception as e:
        print(f"\n  [ERROR] Phase 2 failed: {e}", flush=True)
        import traceback; traceback.print_exc()
        all_results['phase2'] = {'error': str(e)}

    # Phase 3: Model Decay Analysis
    try:
        run_phase3(h1_df, macro_df, all_results)
    except Exception as e:
        print(f"\n  [ERROR] Phase 3 failed: {e}", flush=True)
        import traceback; traceback.print_exc()
        all_results['phase3'] = {'error': str(e)}

    # Phase 4: Ensemble Comparison
    try:
        run_phase4(h1_df, macro_df, all_results)
    except Exception as e:
        print(f"\n  [ERROR] Phase 4 failed: {e}", flush=True)
        import traceback; traceback.print_exc()
        all_results['phase4'] = {'error': str(e)}

    # Phase 5: Portfolio Impact
    try:
        run_phase5(h1_df, macro_df, phase1_results, all_results)
    except Exception as e:
        print(f"\n  [ERROR] Phase 5 failed: {e}", flush=True)
        import traceback; traceback.print_exc()
        all_results['phase5'] = {'error': str(e)}

    # Final summary
    total_elapsed = time.time() - t0
    all_results['completed'] = datetime.now().isoformat()
    all_results['total_elapsed_s'] = round(total_elapsed, 1)

    print(f"\n{'='*80}", flush=True)
    print(f"  FINAL SUMMARY", flush=True)
    print(f"{'='*80}", flush=True)

    # Phase 1 summary
    p1 = all_results.get('phase1', {})
    if isinstance(p1, dict) and 'error' not in p1:
        print(f"\n  Phase 1 — Walk-Forward ML Exit:", flush=True)
        print(f"    {'Strategy':<15} {'AUC':>8} {'UF Sharpe':>10} {'F Sharpe':>10} {'Delta':>8}", flush=True)
        print(f"    {'-'*15} {'-'*8} {'-'*10} {'-'*10} {'-'*8}", flush=True)
        for sn in STRAT_ORDER:
            sr = p1.get(sn, {})
            if sr.get('status') != 'ok':
                print(f"    {sn:<15} {'SKIP':>8}", flush=True)
                continue
            uf_s = sr['unfiltered']['sharpe']
            f_s = sr['filtered']['sharpe']
            auc = sr['mean_auc']
            delta = sr['sharpe_delta']
            print(f"    {sn:<15} {auc:>8.3f} {uf_s:>10.3f} {f_s:>10.3f} {delta:>+8.3f}", flush=True)

    # Phase 2 summary
    p2 = all_results.get('phase2', {})
    if isinstance(p2, dict) and 'error' not in p2 and p2:
        print(f"\n  Phase 2 — Feature Stability:", flush=True)
        for sn, v in p2.items():
            tau = v.get('mean_kendall_tau', 0)
            status = "STABLE" if v.get('stable', False) else "UNSTABLE"
            print(f"    {sn:<15} tau={tau:.3f} [{status}]", flush=True)

    # Phase 3 summary
    p3 = all_results.get('phase3', {})
    if isinstance(p3, dict) and 'error' not in p3 and p3:
        print(f"\n  Phase 3 — Model Shelf Life:", flush=True)
        for sn, v in p3.items():
            sl = v.get('shelf_life_months', '?')
            print(f"    {sn:<15} shelf_life={sl}", flush=True)

    # Phase 4 summary
    p4 = all_results.get('phase4', {})
    if isinstance(p4, dict) and 'error' not in p4 and p4:
        print(f"\n  Phase 4 — Best Model Per Strategy:", flush=True)
        for sn, v in p4.items():
            best = v.get('best', '?')
            best_auc = v.get(best, {}).get('mean_auc', 0)
            print(f"    {sn:<15} best={best.upper()} (AUC={best_auc:.4f})", flush=True)

    # Phase 5 summary
    p5 = all_results.get('phase5', {})
    if isinstance(p5, dict) and 'error' not in p5 and p5:
        print(f"\n  Phase 5 — Portfolio Impact:", flush=True)
        uf = p5.get('unfiltered', {})
        f = p5.get('filtered', {})
        imp = p5.get('sharpe_improvement_pct', 0)
        print(f"    Unfiltered: Sharpe={uf.get('sharpe',0):.3f}, PnL={fmt(uf.get('pnl',0))}, "
              f"MaxDD={fmt(uf.get('max_dd',0))}", flush=True)
        print(f"    ML Filtered: Sharpe={f.get('sharpe',0):.3f}, PnL={fmt(f.get('pnl',0))}, "
              f"MaxDD={fmt(f.get('max_dd',0))}", flush=True)
        print(f"    Sharpe improvement: {imp:+.1f}%", flush=True)

    print(f"\n  Total runtime: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)", flush=True)
    print(f"  Results saved to: {OUTPUT_DIR}/", flush=True)

    # Save final JSON
    final_path = OUTPUT_DIR / "r170_all_results.json"
    try:
        with open(final_path, 'w') as f:
            json.dump(_sanitize(all_results), f, indent=2, default=str)
        print(f"  [SAVED] {final_path}", flush=True)
    except Exception as e:
        print(f"  [WARN] Failed to save final results: {e}", flush=True)

    print(f"\n{'='*80}", flush=True)
    print(f"  R170 COMPLETE", flush=True)
    print(f"{'='*80}", flush=True)


if __name__ == "__main__":
    main()

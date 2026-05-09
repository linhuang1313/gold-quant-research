#!/usr/bin/env python3
"""
R172 — Regime-Conditional Strategy Weight Optimization
========================================================
Per-strategy lot weight optimization conditioned on macro regime state.
Uses Rule-Based 3-regime detector (from R90-A: ANOVA F=10.81, p=0.00002).

  Phase 1: Regime Detection & Labeling
  Phase 2: Per-Strategy Regime Performance (6x3 matrix)
  Phase 3: Weight Grid Search (smart pruning: 729 per regime)
  Phase 4: Full Portfolio Evaluation (static vs dynamic)
  Phase 5: K-Fold Validation (6-fold, pass >=4/6)
  Phase 6: Robustness Checks (regime boundary sensitivity)
"""
import sys, os, io, time, json, glob, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from itertools import product
from multiprocessing import Pool, cpu_count

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("results/r172_regime_weight_optimizer")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100
SPREAD = 0.30
UNIT_LOT = 0.01

P6_LOTS = {'L8_MAX': 0.01, 'PSAR': 0.03, 'TSMOM': 0.04, 'SESS_BO': 0.04, 'DUAL_THRUST': 0.04, 'CHANDELIER': 0.08}
P6_CAPS = {'L8_MAX': 35, 'PSAR': 5, 'TSMOM': 0, 'SESS_BO': 35, 'DUAL_THRUST': 35, 'CHANDELIER': 35}
STRAT_ORDER = ['L8_MAX', 'PSAR', 'TSMOM', 'SESS_BO', 'DUAL_THRUST', 'CHANDELIER']

WEIGHT_GRID = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
REGIMES = ['Bullish', 'Neutral', 'Bearish']

FOLDS = [
    ("Fold1", "2015-01-01", "2016-11-01"),
    ("Fold2", "2016-11-01", "2018-09-01"),
    ("Fold3", "2018-09-01", "2020-07-01"),
    ("Fold4", "2020-07-01", "2022-05-01"),
    ("Fold5", "2022-05-01", "2024-03-01"),
    ("Fold6", "2024-03-01", "2027-01-01"),
]

ROBUSTNESS_VIX_THRESHOLDS = [0.5, 0.75, 1.0, 1.25, 1.5]


def fmt(v):
    return f"${v:>10,.0f}" if v >= 0 else f"-${abs(v):>9,.0f}"


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
    path = "data/external/aligned_daily.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Macro data not found: {path}")
    df = pd.read_csv(path, parse_dates=['Date'], index_col='Date')
    print(f"  Macro data: {len(df)} days ({df.index[0].date()} ~ {df.index[-1].date()})", flush=True)
    return df


# ═══════════════════════════════════════════════════════════════
# Regime Detection (Rule-Based from R90-A)
# ═══════════════════════════════════════════════════════════════

def detect_regimes(macro_df, vix_bear_threshold=1.0):
    """
    Rule-Based 3-regime detector:
      Bullish: DXY_mom20 < 0 AND VIX_zscore < 0 AND yield_curve > 0
      Bearish: DXY_mom20 > 0 AND VIX_zscore > vix_bear_threshold AND credit_stress > 0
      Neutral: everything else
    """
    df = macro_df.copy()

    if 'DXY_Mom20' in df.columns:
        dxy_mom20 = df['DXY_Mom20']
    elif 'DXY_Close' in df.columns:
        dxy_mom20 = df['DXY_Close'].pct_change(20)
    else:
        dxy_mom20 = pd.Series(0, index=df.index)

    if 'VIX_Zscore' in df.columns:
        vix_z = df['VIX_Zscore']
    elif 'VIX_Close' in df.columns:
        rm = df['VIX_Close'].rolling(60).mean()
        rs = df['VIX_Close'].rolling(60).std()
        vix_z = (df['VIX_Close'] - rm) / rs.replace(0, np.nan)
    else:
        vix_z = pd.Series(0, index=df.index)

    if 'YIELD_CURVE_10Y2Y' in df.columns:
        yc = df['YIELD_CURVE_10Y2Y']
    elif 'US10Y_Close' in df.columns and 'US2Y_Close' in df.columns:
        yc = df['US10Y_Close'] - df['US2Y_Close']
    else:
        yc = pd.Series(0, index=df.index)

    if 'CREDIT_STRESS' in df.columns:
        cs = df['CREDIT_STRESS']
    elif 'HYG_Close' in df.columns:
        cs = -df['HYG_Close'].pct_change(5)
    else:
        cs = pd.Series(0, index=df.index)

    regime = pd.Series('Neutral', index=df.index)

    bull_mask = (dxy_mom20 < 0) & (vix_z < 0) & (yc > 0)
    bear_mask = (dxy_mom20 > 0) & (vix_z > vix_bear_threshold) & (cs > 0)

    regime[bull_mask] = 'Bullish'
    regime[bear_mask] = 'Bearish'

    return regime


def map_regime_to_h1(regime_daily, h1_index):
    """Map daily regime labels to H1 timestamps using forward-fill."""
    regime_expanded = regime_daily.reindex(
        pd.DatetimeIndex(h1_index.date).unique()
    ).ffill()
    h1_dates = pd.Series(h1_index.date, index=h1_index)
    h1_regime = h1_dates.map(lambda d: regime_expanded.get(d, 'Neutral'))
    h1_regime = h1_regime.fillna('Neutral')
    return h1_regime


# ═══════════════════════════════════════════════════════════════
# Strategy Backtests
# ═══════════════════════════════════════════════════════════════

def compute_atr(df, period=14):
    tr = pd.DataFrame({
        'hl': df['High'] - df['Low'],
        'hc': (df['High'] - df['Close'].shift(1)).abs(),
        'lc': (df['Low'] - df['Close'].shift(1)).abs(),
    }).max(axis=1)
    return tr.rolling(period).mean()


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


def bt_dual_thrust(h1_df, spread, lot, maxloss_cap=35,
                   n_bars=6, k=0.5, sl_atr=4.5, tp_atr=8.0,
                   trail_act=0.14, trail_dist=0.025, max_hold=20):
    df = h1_df.copy()
    df['ATR'] = compute_atr(df)
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
        if i - last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        hh = np.max(h[i-n_bars:i])
        lc = np.min(c[i-n_bars:i])
        hc = np.max(c[i-n_bars:i])
        ll = np.min(lo[i-n_bars:i])
        rng = max(hh - lc, hc - ll)
        buy_line = o[i] + k * rng
        sell_line = o[i] - k * rng
        if c[i] > buy_line:
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif c[i] < sell_line:
            pos = {'dir': 'SELL', 'entry': c[i]-spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
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
        hh = np.max(h[i-period+1:i+1])
        ll = np.min(lo[i-period+1:i+1])
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
        if i - last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        flipped_long = direction[i] == 1 and direction[i-1] != 1
        flipped_short = direction[i] == -1 and direction[i-1] != -1
        if flipped_long and c[i] > ema[i]:
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif flipped_short and c[i] < ema[i]:
            pos = {'dir': 'SELL', 'entry': c[i]-spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_l8_max(data_bundle, spread, lot, maxloss_cap=0):
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
            'pnl': t.pnl, 'reason': t.exit_reason, 'bars': 0,
        })
    return trades


# ═══════════════════════════════════════════════════════════════
# Daily PnL & Stats Helpers
# ═══════════════════════════════════════════════════════════════

def trades_to_daily_series(trades):
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


def split_trades_by_regime(trades, regime_daily):
    """Split trades into regime buckets based on entry_time date."""
    regime_trades = {r: [] for r in REGIMES}
    for t in trades:
        entry_date = pd.Timestamp(t['entry_time']).date()
        r = regime_daily.get(entry_date, 'Neutral')
        if r in regime_trades:
            regime_trades[r].append(t)
        else:
            regime_trades['Neutral'].append(t)
    return regime_trades


def calc_trade_stats(trades):
    if not trades:
        return {'sharpe': 0.0, 'pnl': 0.0, 'win_rate': 0.0, 'avg_pnl': 0.0, 'n_trades': 0}
    pnls = [t['pnl'] for t in trades]
    daily = trades_to_daily_series(trades)
    daily_arr = daily.values if len(daily) > 0 else np.array([0.0])
    return {
        'sharpe': round(sharpe(daily_arr), 3),
        'pnl': round(sum(pnls), 2),
        'win_rate': round(sum(1 for p in pnls if p > 0) / len(pnls) * 100, 1),
        'avg_pnl': round(np.mean(pnls), 2),
        'n_trades': len(trades),
    }


# ═══════════════════════════════════════════════════════════════
# Grid Search Worker (for multiprocessing)
# ═══════════════════════════════════════════════════════════════

def _eval_regime_combo(args):
    """Evaluate a single weight combination for a specific regime."""
    weights, regime_daily_arrays, regime_name = args
    port_daily = np.zeros(len(list(regime_daily_arrays.values())[0]))
    for idx, strat in enumerate(STRAT_ORDER):
        port_daily += regime_daily_arrays[strat] * weights[idx]
    sh = sharpe(port_daily)
    pnl = float(np.sum(port_daily))
    dd = max_dd(port_daily)
    return {
        'weights': {s: w for s, w in zip(STRAT_ORDER, weights)},
        'regime': regime_name,
        'sharpe': round(sh, 3),
        'pnl': round(pnl, 2),
        'max_dd': round(dd, 2),
    }


# ═══════════════════════════════════════════════════════════════
# Phase functions
# ═══════════════════════════════════════════════════════════════

def phase1_regime_detection(macro_df, h1_df):
    """Phase 1: Detect regimes and map to H1 timestamps."""
    print(f"\n{'='*80}", flush=True)
    print(f"  Phase 1: Regime Detection & Labeling", flush=True)
    print(f"{'='*80}\n", flush=True)

    regime_daily = detect_regimes(macro_df, vix_bear_threshold=1.0)
    regime_dict = regime_daily.to_dict()

    dist = regime_daily.value_counts()
    total = len(regime_daily.dropna())
    print(f"  Regime Distribution (Daily):", flush=True)
    for r in REGIMES:
        cnt = dist.get(r, 0)
        pct = cnt / total * 100 if total > 0 else 0
        print(f"    {r:>10}: {cnt:>5} days ({pct:>5.1f}%)", flush=True)

    h1_regime = map_regime_to_h1(regime_daily, h1_df.index)

    h1_dist = h1_regime.value_counts()
    h1_total = len(h1_regime)
    print(f"\n  Regime Distribution (H1 bars):", flush=True)
    for r in REGIMES:
        cnt = h1_dist.get(r, 0)
        pct = cnt / h1_total * 100 if h1_total > 0 else 0
        print(f"    {r:>10}: {cnt:>6} bars ({pct:>5.1f}%)", flush=True)

    phase1_out = {
        'daily_distribution': {r: int(dist.get(r, 0)) for r in REGIMES},
        'daily_total': total,
        'h1_distribution': {r: int(h1_dist.get(r, 0)) for r in REGIMES},
        'h1_total': h1_total,
    }
    with open(OUTPUT_DIR / "phase1_regimes.json", 'w') as f:
        json.dump(phase1_out, f, indent=2)
    print(f"\n  Phase 1 complete. Saved phase1_regimes.json", flush=True)

    return regime_dict, h1_regime


def phase2_strategy_regime_performance(h1_df, l8_bundle, regime_dict):
    """Phase 2: Run each strategy and compute per-regime metrics."""
    print(f"\n{'='*80}", flush=True)
    print(f"  Phase 2: Per-Strategy Regime Performance", flush=True)
    print(f"{'='*80}\n", flush=True)

    all_trades = {}
    h1_strats = {
        'PSAR':        (bt_psar,         {'maxloss_cap': P6_CAPS['PSAR']}),
        'TSMOM':       (bt_tsmom,        {'maxloss_cap': P6_CAPS['TSMOM']}),
        'SESS_BO':     (bt_sess_bo,      {'maxloss_cap': P6_CAPS['SESS_BO']}),
        'DUAL_THRUST': (bt_dual_thrust,  {'maxloss_cap': P6_CAPS['DUAL_THRUST']}),
        'CHANDELIER':  (bt_chandelier,   {'maxloss_cap': P6_CAPS['CHANDELIER']}),
    }

    print("  Running H1 strategy backtests...", flush=True)
    for name, (fn, kw) in h1_strats.items():
        trades = fn(h1_df, spread=SPREAD, lot=UNIT_LOT, **kw)
        all_trades[name] = trades
        print(f"    {name:>12}: {len(trades)} trades", flush=True)

    print("  Running L8_MAX via engine...", flush=True)
    trades = bt_l8_max(l8_bundle, spread=SPREAD, lot=UNIT_LOT, maxloss_cap=P6_CAPS['L8_MAX'])
    all_trades['L8_MAX'] = trades
    print(f"    {'L8_MAX':>12}: {len(trades)} trades", flush=True)

    # Split by regime and compute stats
    matrix = {}
    print(f"\n  {'Strategy':<12}", end="", flush=True)
    for r in REGIMES:
        print(f" | {'Sharpe':>7} {'PnL':>8} {'WR%':>5} {'AvgPnL':>7} {'N':>5}", end="", flush=True)
    print("", flush=True)
    print(f"  {'':12}", end="", flush=True)
    for r in REGIMES:
        print(f" | {r:^37}", end="", flush=True)
    print("", flush=True)
    print(f"  {'-'*12}", end="", flush=True)
    for _ in REGIMES:
        print(f"-+-{'-'*37}", end="", flush=True)
    print("", flush=True)

    for name in STRAT_ORDER:
        regime_trades = split_trades_by_regime(all_trades[name], regime_dict)
        matrix[name] = {}
        print(f"  {name:<12}", end="", flush=True)
        for r in REGIMES:
            stats = calc_trade_stats(regime_trades[r])
            matrix[name][r] = stats
            print(f" | {stats['sharpe']:>7.2f} {stats['pnl']:>8.0f} {stats['win_rate']:>5.1f} "
                  f"{stats['avg_pnl']:>7.2f} {stats['n_trades']:>5}", end="", flush=True)
        print("", flush=True)

    phase2_out = {'matrix': matrix}
    with open(OUTPUT_DIR / "phase2_regime_performance.json", 'w') as f:
        json.dump(phase2_out, f, indent=2)
    print(f"\n  Phase 2 complete. Saved phase2_regime_performance.json", flush=True)

    return all_trades, matrix


def phase3_weight_grid_search(all_trades, regime_dict):
    """Phase 3: Smart pruned weight grid search per regime."""
    print(f"\n{'='*80}", flush=True)
    print(f"  Phase 3: Weight Grid Search (Smart Pruning)", flush=True)
    print(f"  Grid per strategy: {WEIGHT_GRID}", flush=True)
    print(f"  Pruning: 6 strats x 6 weights = 36 per regime -> top-3 -> 3^6 = 729/regime", flush=True)
    print(f"{'='*80}\n", flush=True)

    # Build regime-specific daily PnL arrays at unit lot
    regime_unit_dailies = {r: {} for r in REGIMES}
    for name in STRAT_ORDER:
        regime_trades = split_trades_by_regime(all_trades[name], regime_dict)
        for r in REGIMES:
            regime_unit_dailies[r][name] = trades_to_daily_series(regime_trades[r])

    # Align all regime dailies to common date index per regime
    regime_aligned = {}
    for r in REGIMES:
        all_dates = set()
        for ds in regime_unit_dailies[r].values():
            all_dates.update(ds.index)
        all_dates = sorted(all_dates)
        idx = pd.DatetimeIndex(all_dates)
        regime_aligned[r] = {}
        for name in STRAT_ORDER:
            ds = regime_unit_dailies[r][name]
            regime_aligned[r][name] = ds.reindex(idx, fill_value=0.0).values
        regime_aligned[r]['_idx'] = idx

    # Stage A: Per-strategy independent screening (pick top-3 weights per strat per regime)
    print("  Stage A: Per-strategy independent screening...", flush=True)
    top3_per_strat = {r: {} for r in REGIMES}

    for r in REGIMES:
        idx = regime_aligned[r]['_idx']
        n_days = len(idx)
        if n_days < 10:
            print(f"    [WARN] Regime {r} has only {n_days} days, skipping", flush=True)
            for name in STRAT_ORDER:
                top3_per_strat[r][name] = [1.0, 1.0, 1.0]
            continue

        for name in STRAT_ORDER:
            base_arr = regime_aligned[r][name]
            weight_scores = []
            for w in WEIGHT_GRID:
                scaled = base_arr * w
                sh = sharpe(scaled)
                weight_scores.append((sh, w))
            weight_scores.sort(reverse=True)
            top3_per_strat[r][name] = [ws[1] for ws in weight_scores[:3]]

        print(f"    {r:>8}: ", end="", flush=True)
        for name in STRAT_ORDER:
            print(f"{name}={top3_per_strat[r][name]}  ", end="", flush=True)
        print("", flush=True)

    # Stage B: Full combinatorial search with pruned grid (3^6 = 729 per regime)
    print(f"\n  Stage B: Combinatorial search (729 combos per regime)...", flush=True)
    top_configs = {r: [] for r in REGIMES}
    total_evaluated = 0

    for r in REGIMES:
        idx = regime_aligned[r]['_idx']
        n_days = len(idx)
        if n_days < 10:
            continue

        strat_grids = [top3_per_strat[r][name] for name in STRAT_ORDER]
        combos = list(product(*strat_grids))
        print(f"    {r:>8}: evaluating {len(combos)} combos over {n_days} days...", flush=True)

        results = []
        for combo in combos:
            port_daily = np.zeros(n_days)
            for s_idx, name in enumerate(STRAT_ORDER):
                port_daily += regime_aligned[r][name] * combo[s_idx]
            sh = sharpe(port_daily)
            pnl = float(np.sum(port_daily))
            dd = max_dd(port_daily)
            results.append({
                'weights': {name: combo[s_idx] for s_idx, name in enumerate(STRAT_ORDER)},
                'sharpe': round(sh, 3),
                'pnl': round(pnl, 2),
                'max_dd': round(dd, 2),
            })
            total_evaluated += 1

        results.sort(key=lambda x: x['sharpe'], reverse=True)
        top_configs[r] = results[:5]

        print(f"             Top-5 by Sharpe:", flush=True)
        for i, res in enumerate(results[:5]):
            w = res['weights']
            print(f"             #{i+1} Sh={res['sharpe']:.3f} PnL={res['pnl']:.0f} DD={res['max_dd']:.0f}  "
                  f"W=[{w['L8_MAX']:.2f},{w['PSAR']:.2f},{w['TSMOM']:.2f},"
                  f"{w['SESS_BO']:.2f},{w['DUAL_THRUST']:.2f},{w['CHANDELIER']:.2f}]", flush=True)

    print(f"\n  Total combos evaluated: {total_evaluated}", flush=True)

    phase3_out = {
        'top3_weights_per_strat': {r: {s: top3_per_strat[r][s] for s in STRAT_ORDER} for r in REGIMES},
        'top5_configs': top_configs,
        'total_evaluated': total_evaluated,
    }
    with open(OUTPUT_DIR / "phase3_grid_search.json", 'w') as f:
        json.dump(phase3_out, f, indent=2)
    print(f"  Phase 3 complete. Saved phase3_grid_search.json", flush=True)

    return top_configs


def phase4_portfolio_evaluation(all_trades, regime_dict, top_configs):
    """Phase 4: Compare static P6 vs dynamic regime-weighted portfolio."""
    print(f"\n{'='*80}", flush=True)
    print(f"  Phase 4: Full Portfolio Evaluation (Static vs Dynamic)", flush=True)
    print(f"{'='*80}\n", flush=True)

    # Build static portfolio daily PnL (P6 config)
    all_daily = {}
    for name in STRAT_ORDER:
        all_daily[name] = trades_to_daily_series(all_trades[name])

    all_dates = set()
    for ds in all_daily.values():
        all_dates.update(ds.index)
    all_dates = sorted(all_dates)
    port_idx = pd.DatetimeIndex(all_dates)

    static_daily = np.zeros(len(port_idx))
    for name in STRAT_ORDER:
        mult = P6_LOTS[name] / UNIT_LOT
        aligned = all_daily[name].reindex(port_idx, fill_value=0.0).values * mult
        static_daily += aligned

    static_sharpe = sharpe(static_daily)
    static_pnl = float(np.sum(static_daily))
    static_dd = max_dd(static_daily)

    print(f"  Static P6 Portfolio:", flush=True)
    print(f"    Sharpe: {static_sharpe:.3f}  PnL: {fmt(static_pnl)}  MaxDD: {fmt(static_dd)}", flush=True)
    for name in STRAT_ORDER:
        print(f"      {name:>12}: {P6_LOTS[name]:.2f} lot", flush=True)

    # Build dynamic portfolio: apply regime-specific weight multipliers
    best_regime_weights = {}
    for r in REGIMES:
        if top_configs[r]:
            best_regime_weights[r] = top_configs[r][0]['weights']
        else:
            best_regime_weights[r] = {s: 1.0 for s in STRAT_ORDER}

    # Build trade-level dynamic PnL: adjust each trade's pnl by regime weight at entry
    dynamic_configs = []
    for combo_idx in range(min(3, max(len(top_configs[r]) for r in REGIMES))):
        regime_w = {}
        for r in REGIMES:
            if combo_idx < len(top_configs[r]):
                regime_w[r] = top_configs[r][combo_idx]['weights']
            else:
                regime_w[r] = {s: 1.0 for s in STRAT_ORDER}

        dynamic_daily_pnl = np.zeros(len(port_idx))
        date_to_idx = {d: i for i, d in enumerate(port_idx)}

        for name in STRAT_ORDER:
            base_lot = P6_LOTS[name]
            for t in all_trades[name]:
                entry_date = pd.Timestamp(t['entry_time']).date()
                r = regime_dict.get(entry_date, 'Neutral')
                if r not in REGIMES:
                    r = 'Neutral'
                weight_mult = regime_w[r].get(name, 1.0)
                adjusted_pnl = t['pnl'] * (base_lot / UNIT_LOT) * weight_mult
                exit_date = pd.Timestamp(t['exit_time']).date()
                exit_ts = pd.Timestamp(exit_date)
                if exit_ts in date_to_idx:
                    dynamic_daily_pnl[date_to_idx[exit_ts]] += adjusted_pnl

        dyn_sharpe = sharpe(dynamic_daily_pnl)
        dyn_pnl = float(np.sum(dynamic_daily_pnl))
        dyn_dd = max_dd(dynamic_daily_pnl)

        dynamic_configs.append({
            'combo_idx': combo_idx,
            'regime_weights': regime_w,
            'sharpe': round(dyn_sharpe, 3),
            'pnl': round(dyn_pnl, 2),
            'max_dd': round(dyn_dd, 2),
            'sharpe_delta': round(dyn_sharpe - static_sharpe, 3),
        })

        label = f"Dynamic#{combo_idx+1}"
        print(f"\n  {label}:", flush=True)
        print(f"    Sharpe: {dyn_sharpe:.3f}  PnL: {fmt(dyn_pnl)}  MaxDD: {fmt(dyn_dd)}", flush=True)
        print(f"    Delta vs Static: Sharpe {dyn_sharpe-static_sharpe:+.3f}", flush=True)
        for r in REGIMES:
            w = regime_w[r]
            print(f"      {r:>8}: [{w['L8_MAX']:.2f}, {w['PSAR']:.2f}, {w['TSMOM']:.2f}, "
                  f"{w['SESS_BO']:.2f}, {w['DUAL_THRUST']:.2f}, {w['CHANDELIER']:.2f}]", flush=True)

    phase4_out = {
        'static': {
            'lots': P6_LOTS,
            'sharpe': round(static_sharpe, 3),
            'pnl': round(static_pnl, 2),
            'max_dd': round(static_dd, 2),
        },
        'dynamic_configs': dynamic_configs,
    }
    with open(OUTPUT_DIR / "phase4_portfolio.json", 'w') as f:
        json.dump(phase4_out, f, indent=2)
    print(f"\n  Phase 4 complete. Saved phase4_portfolio.json", flush=True)

    return static_sharpe, dynamic_configs


def phase5_kfold_validation(h1_df, l8_bundle, macro_df, dynamic_configs, static_sharpe):
    """Phase 5: 6-fold K-Fold validation for top-3 dynamic configs."""
    print(f"\n{'='*80}", flush=True)
    print(f"  Phase 5: K-Fold Validation (6-fold, pass >= 4/6)", flush=True)
    print(f"{'='*80}\n", flush=True)

    h1_strats = {
        'PSAR':        (bt_psar,         {'maxloss_cap': P6_CAPS['PSAR']}),
        'TSMOM':       (bt_tsmom,        {'maxloss_cap': P6_CAPS['TSMOM']}),
        'SESS_BO':     (bt_sess_bo,      {'maxloss_cap': P6_CAPS['SESS_BO']}),
        'DUAL_THRUST': (bt_dual_thrust,  {'maxloss_cap': P6_CAPS['DUAL_THRUST']}),
        'CHANDELIER':  (bt_chandelier,   {'maxloss_cap': P6_CAPS['CHANDELIER']}),
    }

    kfold_results = {}

    for cfg_idx, cfg in enumerate(dynamic_configs[:3]):
        regime_w = cfg['regime_weights']
        print(f"  Config #{cfg_idx+1} (full-sample Sharpe={cfg['sharpe']:.3f}):", flush=True)

        fold_sharpe_static = []
        fold_sharpe_dynamic = []

        for fold_name, start, end in FOLDS:
            h1_fold = h1_df[start:end]
            if len(h1_fold) < 200:
                print(f"    {fold_name}: skipped (too few bars: {len(h1_fold)})", flush=True)
                fold_sharpe_static.append(0.0)
                fold_sharpe_dynamic.append(0.0)
                continue

            macro_fold = macro_df[start:end]
            regime_fold = detect_regimes(macro_fold, vix_bear_threshold=1.0)
            regime_fold_dict = regime_fold.to_dict()

            fold_trades = {}
            for name, (fn, kw) in h1_strats.items():
                fold_trades[name] = fn(h1_fold, spread=SPREAD, lot=UNIT_LOT, **kw)

            try:
                l8_fold = l8_bundle.slice(start, end)
                fold_trades['L8_MAX'] = bt_l8_max(l8_fold, spread=SPREAD, lot=UNIT_LOT,
                                                   maxloss_cap=P6_CAPS['L8_MAX'])
            except Exception:
                fold_trades['L8_MAX'] = []

            # Static portfolio for this fold
            fold_daily = {}
            for name in STRAT_ORDER:
                fold_daily[name] = trades_to_daily_series(fold_trades[name])

            fold_dates = set()
            for ds in fold_daily.values():
                fold_dates.update(ds.index)
            fold_dates = sorted(fold_dates)
            fold_idx = pd.DatetimeIndex(fold_dates)

            static_arr = np.zeros(len(fold_idx))
            for name in STRAT_ORDER:
                mult = P6_LOTS[name] / UNIT_LOT
                static_arr += fold_daily[name].reindex(fold_idx, fill_value=0.0).values * mult
            s_static = sharpe(static_arr)
            fold_sharpe_static.append(s_static)

            # Dynamic portfolio for this fold
            dynamic_arr = np.zeros(len(fold_idx))
            date_to_idx_f = {d: i for i, d in enumerate(fold_idx)}
            for name in STRAT_ORDER:
                base_lot = P6_LOTS[name]
                for t in fold_trades[name]:
                    entry_date = pd.Timestamp(t['entry_time']).date()
                    r = regime_fold_dict.get(entry_date, 'Neutral')
                    if r not in REGIMES:
                        r = 'Neutral'
                    weight_mult = regime_w[r].get(name, 1.0)
                    adjusted_pnl = t['pnl'] * (base_lot / UNIT_LOT) * weight_mult
                    exit_date = pd.Timestamp(t['exit_time']).date()
                    exit_ts = pd.Timestamp(exit_date)
                    if exit_ts in date_to_idx_f:
                        dynamic_arr[date_to_idx_f[exit_ts]] += adjusted_pnl

            s_dynamic = sharpe(dynamic_arr)
            fold_sharpe_dynamic.append(s_dynamic)

            delta = s_dynamic - s_static
            marker = "+" if delta > 0 else "-"
            print(f"    {fold_name} ({start}~{end}): Static={s_static:.3f} Dynamic={s_dynamic:.3f} "
                  f"Delta={delta:+.3f} [{marker}]", flush=True)

        wins = sum(1 for sd, ss in zip(fold_sharpe_dynamic, fold_sharpe_static) if sd > ss)
        passed = wins >= 4
        status = "PASS" if passed else "FAIL"
        print(f"    Result: {wins}/6 folds dynamic > static => [{status}]", flush=True)

        kfold_results[f"config_{cfg_idx+1}"] = {
            'regime_weights': regime_w,
            'fold_sharpe_static': [round(s, 3) for s in fold_sharpe_static],
            'fold_sharpe_dynamic': [round(s, 3) for s in fold_sharpe_dynamic],
            'fold_deltas': [round(d-s, 3) for d, s in zip(fold_sharpe_dynamic, fold_sharpe_static)],
            'wins': wins,
            'passed': passed,
        }
        print("", flush=True)

    phase5_out = {'kfold_results': kfold_results, 'pass_criterion': '>=4/6 folds dynamic > static'}
    with open(OUTPUT_DIR / "phase5_kfold.json", 'w') as f:
        json.dump(phase5_out, f, indent=2)
    print(f"  Phase 5 complete. Saved phase5_kfold.json", flush=True)

    return kfold_results


def phase6_robustness_checks(h1_df, all_trades, macro_df, top_configs):
    """Phase 6: Sensitivity of regime labels to boundary thresholds."""
    print(f"\n{'='*80}", flush=True)
    print(f"  Phase 6: Robustness Checks (Regime Boundary Sensitivity)", flush=True)
    print(f"  VIX threshold grid: {ROBUSTNESS_VIX_THRESHOLDS}", flush=True)
    print(f"{'='*80}\n", flush=True)

    sensitivity_results = {}
    reference_weights = {}
    for r in REGIMES:
        if top_configs[r]:
            reference_weights[r] = top_configs[r][0]['weights']
        else:
            reference_weights[r] = {s: 1.0 for s in STRAT_ORDER}

    for vix_thresh in ROBUSTNESS_VIX_THRESHOLDS:
        regime_daily = detect_regimes(macro_df, vix_bear_threshold=vix_thresh)
        regime_dict_alt = regime_daily.to_dict()

        dist = regime_daily.value_counts()
        total = len(regime_daily.dropna())

        # Re-run grid search for this threshold (quick version: just evaluate reference weights)
        regime_sharpes = {}
        for r in REGIMES:
            regime_trades_r = {}
            for name in STRAT_ORDER:
                rt = split_trades_by_regime(all_trades[name], regime_dict_alt)
                regime_trades_r[name] = rt[r]

            # Evaluate reference weights in this regime
            all_dates_r = set()
            for name in STRAT_ORDER:
                ds = trades_to_daily_series(regime_trades_r[name])
                all_dates_r.update(ds.index)
            all_dates_r = sorted(all_dates_r)
            if len(all_dates_r) < 10:
                regime_sharpes[r] = 0.0
                continue

            idx_r = pd.DatetimeIndex(all_dates_r)
            port_daily = np.zeros(len(idx_r))
            for name in STRAT_ORDER:
                ds = trades_to_daily_series(regime_trades_r[name])
                w = reference_weights[r].get(name, 1.0)
                port_daily += ds.reindex(idx_r, fill_value=0.0).values * w
            regime_sharpes[r] = round(sharpe(port_daily), 3)

        # Also find optimal weights for this threshold
        optimal_weights_alt = {}
        for r in REGIMES:
            regime_trades_r = {}
            for name in STRAT_ORDER:
                rt = split_trades_by_regime(all_trades[name], regime_dict_alt)
                regime_trades_r[name] = rt[r]

            all_dates_r = set()
            for name in STRAT_ORDER:
                ds = trades_to_daily_series(regime_trades_r[name])
                all_dates_r.update(ds.index)
            all_dates_r = sorted(all_dates_r)
            if len(all_dates_r) < 10:
                optimal_weights_alt[r] = {s: 1.0 for s in STRAT_ORDER}
                continue

            idx_r = pd.DatetimeIndex(all_dates_r)
            aligned_r = {}
            for name in STRAT_ORDER:
                ds = trades_to_daily_series(regime_trades_r[name])
                aligned_r[name] = ds.reindex(idx_r, fill_value=0.0).values

            best_sh = -999
            best_w = {s: 1.0 for s in STRAT_ORDER}
            # Quick search: independent per-strategy
            for name in STRAT_ORDER:
                best_w_s = 1.0
                best_sh_s = -999
                for w in WEIGHT_GRID:
                    sh = sharpe(aligned_r[name] * w)
                    if sh > best_sh_s:
                        best_sh_s = sh
                        best_w_s = w
                best_w[name] = best_w_s
            optimal_weights_alt[r] = best_w

        # Measure weight stability vs reference
        weight_diffs = {}
        for r in REGIMES:
            diffs = []
            for name in STRAT_ORDER:
                ref = reference_weights[r].get(name, 1.0)
                alt = optimal_weights_alt[r].get(name, 1.0)
                diffs.append(abs(ref - alt))
            weight_diffs[r] = round(np.mean(diffs), 3)

        sensitivity_results[f"vix_{vix_thresh}"] = {
            'vix_threshold': vix_thresh,
            'distribution': {r: int(dist.get(r, 0)) for r in REGIMES},
            'regime_sharpes_with_ref_weights': regime_sharpes,
            'optimal_weights': optimal_weights_alt,
            'mean_weight_diff_vs_reference': weight_diffs,
        }

        print(f"  VIX threshold = {vix_thresh}:", flush=True)
        print(f"    Distribution: Bull={dist.get('Bullish',0)} Neut={dist.get('Neutral',0)} Bear={dist.get('Bearish',0)}", flush=True)
        print(f"    Sharpe w/ ref weights: Bull={regime_sharpes.get('Bullish',0):.3f} "
              f"Neut={regime_sharpes.get('Neutral',0):.3f} Bear={regime_sharpes.get('Bearish',0):.3f}", flush=True)
        print(f"    Mean weight drift: Bull={weight_diffs.get('Bullish',0):.3f} "
              f"Neut={weight_diffs.get('Neutral',0):.3f} Bear={weight_diffs.get('Bearish',0):.3f}", flush=True)

    # Assess overall robustness
    all_diffs = []
    for key, res in sensitivity_results.items():
        for r in REGIMES:
            all_diffs.append(res['mean_weight_diff_vs_reference'].get(r, 0))
    mean_instability = np.mean(all_diffs) if all_diffs else 0
    is_robust = mean_instability < 0.3

    print(f"\n  Overall weight instability: {mean_instability:.3f}", flush=True)
    if is_robust:
        print(f"  CONCLUSION: Weights are reasonably STABLE across regime definitions.", flush=True)
    else:
        print(f"  CONCLUSION: Weights are BRITTLE - dynamic allocation may NOT be robust.", flush=True)

    phase6_out = {
        'sensitivity_results': sensitivity_results,
        'mean_instability': round(float(mean_instability), 4),
        'is_robust': bool(is_robust),
        'reference_weights': reference_weights,
    }
    with open(OUTPUT_DIR / "phase6_robustness.json", 'w') as f:
        json.dump(phase6_out, f, indent=2, default=str)
    print(f"  Phase 6 complete. Saved phase6_robustness.json", flush=True)

    return phase6_out


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 80, flush=True)
    print("  R172 — Regime-Conditional Strategy Weight Optimization", flush=True)
    print(f"  Strategies: {STRAT_ORDER}", flush=True)
    print(f"  Regimes: {REGIMES}", flush=True)
    print(f"  Weight grid: {WEIGHT_GRID}", flush=True)
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("=" * 80, flush=True)

    from backtest.runner import DataBundle

    # Load data
    print("\n  Loading data...", flush=True)
    h1_df = load_h1()
    print(f"  H1: {len(h1_df)} bars ({h1_df.index[0]} ~ {h1_df.index[-1]})", flush=True)

    macro_df = load_macro()

    print("  Preparing L8_MAX DataBundle...", flush=True)
    l8_bundle = DataBundle.load_custom()
    print("  L8 bundle ready.", flush=True)

    # ═══════════ Phase 1 ═══════════
    regime_dict, h1_regime = phase1_regime_detection(macro_df, h1_df)

    # ═══════════ Phase 2 ═══════════
    all_trades, perf_matrix = phase2_strategy_regime_performance(h1_df, l8_bundle, regime_dict)

    # ═══════════ Phase 3 ═══════════
    top_configs = phase3_weight_grid_search(all_trades, regime_dict)

    # ═══════════ Phase 4 ═══════════
    static_sharpe, dynamic_configs = phase4_portfolio_evaluation(all_trades, regime_dict, top_configs)

    # ═══════════ Phase 5 ═══════════
    kfold_results = phase5_kfold_validation(h1_df, l8_bundle, macro_df, dynamic_configs, static_sharpe)

    # ═══════════ Phase 6 ═══════════
    robustness = phase6_robustness_checks(h1_df, all_trades, macro_df, top_configs)

    # ═══════════ Final Summary ═══════════
    elapsed = time.time() - t0
    print(f"\n{'='*80}", flush=True)
    print(f"  R172 — FINAL SUMMARY", flush=True)
    print(f"{'='*80}\n", flush=True)

    # K-Fold pass count
    n_passed = sum(1 for k, v in kfold_results.items() if v.get('passed', False))
    n_configs = len(kfold_results)
    print(f"  K-Fold: {n_passed}/{n_configs} configs passed (>=4/6 folds)", flush=True)
    print(f"  Robustness: {'STABLE' if robustness['is_robust'] else 'BRITTLE'} "
          f"(instability={robustness['mean_instability']:.3f})", flush=True)

    # Decision
    if n_passed > 0 and robustness['is_robust']:
        print(f"\n  RECOMMENDATION: Deploy regime-conditional weights.", flush=True)
        print(f"  Use the following weight map:", flush=True)
        for r in REGIMES:
            if top_configs[r]:
                w = top_configs[r][0]['weights']
                print(f"    {r:>8}: L8={w['L8_MAX']:.2f} PSAR={w['PSAR']:.2f} TSMOM={w['TSMOM']:.2f} "
                      f"SESS={w['SESS_BO']:.2f} DT={w['DUAL_THRUST']:.2f} CH={w['CHANDELIER']:.2f}", flush=True)
    elif n_passed > 0:
        print(f"\n  RECOMMENDATION: Regime weights show in-sample benefit but are NOT robust.", flush=True)
        print(f"  Consider using a simplified version (e.g., only scale down in Bearish regime).", flush=True)
    else:
        print(f"\n  RECOMMENDATION: Static P6 weights remain optimal.", flush=True)
        print(f"  Dynamic regime weighting does NOT pass K-Fold validation.", flush=True)

    print(f"\n  P6 Static Config (baseline):", flush=True)
    for name in STRAT_ORDER:
        print(f"    {name:>12}: {P6_LOTS[name]:.2f} lot", flush=True)

    # Save master output
    master_output = {
        'config': {
            'strat_order': STRAT_ORDER,
            'regimes': REGIMES,
            'weight_grid': WEIGHT_GRID,
            'p6_lots': P6_LOTS,
            'p6_caps': P6_CAPS,
            'folds': FOLDS,
            'vix_thresholds': ROBUSTNESS_VIX_THRESHOLDS,
        },
        'kfold_summary': {
            'n_passed': n_passed,
            'n_configs': n_configs,
        },
        'robustness_summary': {
            'is_robust': robustness['is_robust'],
            'mean_instability': robustness['mean_instability'],
        },
        'best_regime_weights': {
            r: top_configs[r][0]['weights'] if top_configs[r] else {s: 1.0 for s in STRAT_ORDER}
            for r in REGIMES
        },
        'elapsed_s': round(elapsed, 1),
        'elapsed_min': round(elapsed / 60, 1),
    }
    out_path = OUTPUT_DIR / "r172_master_results.json"
    with open(out_path, 'w') as f:
        json.dump(master_output, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}", flush=True)

    print(f"\n{'='*80}", flush=True)
    print(f"  R172 COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)
    print(f"{'='*80}", flush=True)


if __name__ == "__main__":
    main()

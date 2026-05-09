#!/usr/bin/env python3
"""
R152 — Silver (XAGUSD) Strategy Migration & Validation
========================================================
Migrates the 5 H1 strategies (PSAR, TSMOM, SESS_BO, DUAL_THRUST, CHANDELIER)
from gold to silver. L8_MAX is excluded because it depends on the gold-specific
DataBundle/engine pipeline.

Key differences for XAGUSD:
  - PV = 5000 (standard lot = 5000 oz, vs gold's 100 oz)
  - Spread ~ 0.02 USD (vs gold's ~0.30 USD)
  - Price range: ~$15-76 (much lower than gold $1000-3000)
  - ATR much smaller in absolute terms -> ATR minimum check adjusted
  - MaxLossCap scaled proportionally

Phases:
  1. Load silver H1 data & basic statistics
  2. Run all 5 strategies at unit lot (0.01) — gold params first
  3. Parameter sensitivity: sweep key params to find silver-optimal
  4. K-Fold 5-fold validation on best config per strategy
  5. Walk-Forward validation (6 windows, 2yr train / 1yr test)
  6. Gold vs Silver correlation (using gold_silver_ratio.csv if available)
  7. Combined gold+silver portfolio analysis
"""
import sys, os, time, json, glob, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from itertools import product

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("results/r152_silver_expansion")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# XAGUSD contract specs
PV_SILVER = 5000       # 1 standard lot = 5000 oz
SPREAD_SILVER = 0.020  # typical XAGUSD spread ~$0.02
UNIT_LOT = 0.01
CAPITAL = 5000
ATR_MIN_SILVER = 0.005  # much lower than gold's 0.1

# Gold specs for comparison
PV_GOLD = 100
SPREAD_GOLD = 0.30

# Cap values scaled for silver (proportional to typical trade size)
CAPS_SILVER = {
    'PSAR':         5,
    'TSMOM':        0,
    'SESS_BO':     35,
    'DUAL_THRUST': 35,
    'CHANDELIER':  35,
}

STRAT_ORDER = ['PSAR', 'TSMOM', 'SESS_BO', 'DUAL_THRUST', 'CHANDELIER']

FOLDS = [
    ("Fold1", "2015-01-01", "2017-03-01"),
    ("Fold2", "2017-03-01", "2019-06-01"),
    ("Fold3", "2019-06-01", "2021-09-01"),
    ("Fold4", "2021-09-01", "2023-12-01"),
    ("Fold5", "2023-12-01", "2027-01-01"),
]

WF_WINDOWS = [
    ("WF1", "2015-01-01", "2017-01-01", "2017-01-01", "2018-01-01"),
    ("WF2", "2017-01-01", "2019-01-01", "2019-01-01", "2020-01-01"),
    ("WF3", "2019-01-01", "2021-01-01", "2021-01-01", "2022-01-01"),
    ("WF4", "2021-01-01", "2023-01-01", "2023-01-01", "2024-01-01"),
    ("WF5", "2023-01-01", "2025-01-01", "2025-01-01", "2026-05-01"),
    ("WF6", "2020-01-01", "2022-01-01", "2022-01-01", "2023-01-01"),
]

t0 = time.time()


def fmt(v):
    return f"${v:>10,.0f}" if v >= 0 else f"-${abs(v):>9,.0f}"


# ═══════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════

def load_h1_silver():
    candidates = sorted(glob.glob("data/download/xagusd-h1-bid-2015-*.csv"))
    if not candidates:
        raise FileNotFoundError("No xagusd H1 CSV found in data/download/")
    csv_path = candidates[-1]
    print(f"  Loading silver H1 from: {csv_path}", flush=True)
    df = pd.read_csv(csv_path)
    df['time'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_localize(None)
    df = df.set_index('time')[['open', 'high', 'low', 'close', 'volume']]
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = df[df['Close'] > 1]
    return df


def load_h1_gold():
    candidates = sorted(glob.glob("data/download/xauusd-h1-bid-2015-*.csv"))
    if not candidates:
        return None
    csv_path = candidates[-1]
    df = pd.read_csv(csv_path)
    df['time'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_localize(None)
    df = df.set_index('time')[['open', 'high', 'low', 'close', 'volume']]
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = df[df['Close'] > 100]
    return df


# ═══════════════════════════════════════════════════════════════
# Shared helpers
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


# ═══════════════════════════════════════════════════════════════
# Strategy backtests — parameterized pv/spread/atr_min
# ═══════════════════════════════════════════════════════════════

def bt_psar(h1_df, spread, lot, pv, atr_min=0.1, maxloss_cap=0,
            sl_atr=4.5, tp_atr=16.0, trail_act=0.20, trail_dist=0.04, max_hold=20):
    df = add_psar(h1_df).dropna(subset=['PSAR_dir', 'ATR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    pdir = df['PSAR_dir'].values; atr = df['ATR'].values
    times = df.index; n = len(df)
    trades = []; pos = None; last_exit = -999
    for i in range(1, n):
        if pos is not None:
            result = _run_exit_with_cap(pos, i, h[i], lo[i], c[i], spread, lot, pv, times,
                                        sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if i-last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < atr_min: continue
        if pdir[i-1] == -1 and pdir[i] == 1:
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif pdir[i-1] == 1 and pdir[i] == -1:
            pos = {'dir': 'SELL', 'entry': c[i]-spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_tsmom(h1_df, spread, lot, pv, atr_min=0.1, maxloss_cap=0,
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
            result = _run_exit_with_cap(pos, i, h[i], lo[i], c[i], spread, lot, pv, times,
                                        sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            if pos['dir'] == 'BUY' and score[i] < 0:
                pnl = (c[i]-pos['entry']-spread)*lot*pv
                trades.append(_mk(pos, c[i], times[i], "Reversal", i, pnl)); pos = None; last_exit = i; continue
            elif pos['dir'] == 'SELL' and score[i] > 0:
                pnl = (pos['entry']-c[i]-spread)*lot*pv
                trades.append(_mk(pos, c[i], times[i], "Reversal", i, pnl)); pos = None; last_exit = i; continue
            continue
        if i-last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < atr_min: continue
        if np.isnan(score[i]) or np.isnan(score[i-1]): continue
        if score[i] > 0 and score[i-1] <= 0:
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif score[i] < 0 and score[i-1] >= 0:
            pos = {'dir': 'SELL', 'entry': c[i]-spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_sess_bo(h1_df, spread, lot, pv, atr_min=0.1, maxloss_cap=0,
               session_hour=12, lookback=4, sl_atr=4.5, tp_atr=4.0,
               trail_act=0.14, trail_dist=0.025, max_hold=20):
    df = h1_df.copy(); df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; hours = df.index.hour; times = df.index; n = len(df)
    trades = []; pos = None; last_exit = -999
    for i in range(lookback, n):
        if pos is not None:
            result = _run_exit_with_cap(pos, i, h[i], lo[i], c[i], spread, lot, pv, times,
                                        sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if hours[i] != session_hour: continue
        if i-last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < atr_min: continue
        hh = max(h[i-j] for j in range(1, lookback+1))
        ll = min(lo[i-j] for j in range(1, lookback+1))
        if c[i] > hh:
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif c[i] < ll:
            pos = {'dir': 'SELL', 'entry': c[i]-spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_dual_thrust(h1_df, spread, lot, pv, atr_min=0.1, maxloss_cap=35,
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
            result = _run_exit_with_cap(pos, i, h[i], lo[i], c[i], spread, lot, pv, times,
                                        sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if i - last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < atr_min: continue
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


def bt_chandelier(h1_df, spread, lot, pv, atr_min=0.1, maxloss_cap=35,
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
            result = _run_exit_with_cap(pos, i, h[i], lo[i], c[i], spread, lot, pv, times,
                                        sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if i - last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < atr_min: continue
        if direction[i] == 1 and direction[i-1] != 1 and c[i] > ema[i]:
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif direction[i] == -1 and direction[i-1] != -1 and c[i] < ema[i]:
            pos = {'dir': 'SELL', 'entry': c[i]-spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


# ═══════════════════════════════════════════════════════════════
# Stats helpers
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
    if len(arr) < 10: return 0.0
    s = np.std(arr, ddof=1)
    if s == 0: return 0.0
    return float(np.mean(arr) / s * np.sqrt(252))


def max_dd(arr):
    if len(arr) == 0: return 0.0
    eq = np.cumsum(arr)
    return float((np.maximum.accumulate(eq) - eq).max())


def cvar99(arr):
    if len(arr) < 20: return 0.0
    v = np.percentile(arr, 1)
    tail = arr[arr <= v]
    return float(tail.mean()) if len(tail) > 0 else float(v)


def compute_stats(trades, label=""):
    if not trades:
        return {'n_trades': 0, 'pnl': 0, 'sharpe': 0, 'max_dd': 0, 'wr': 0, 'avg_pnl': 0}
    pnls = [t['pnl'] for t in trades]
    ds = trades_to_daily_series(trades)
    daily_arr = ds.values
    n = len(trades)
    return {
        'n_trades': n,
        'pnl': round(sum(pnls), 2),
        'sharpe': round(sharpe(daily_arr), 2),
        'max_dd': round(max_dd(daily_arr), 2),
        'wr': round(sum(1 for p in pnls if p > 0) / n * 100, 1),
        'avg_pnl': round(sum(pnls) / n, 2),
        'cvar99': round(cvar99(daily_arr), 2) if len(daily_arr) >= 20 else 0,
    }


# ═══════════════════════════════════════════════════════════════
# Strategy runner helper
# ═══════════════════════════════════════════════════════════════

ALL_STRATS = {
    'PSAR':         (bt_psar,         {'sl_atr': 4.5, 'tp_atr': 16.0, 'trail_act': 0.20, 'trail_dist': 0.04, 'max_hold': 20}),
    'TSMOM':        (bt_tsmom,        {'fast': 480, 'slow': 720, 'sl_atr': 4.5, 'tp_atr': 6.0, 'trail_act': 0.14, 'trail_dist': 0.025, 'max_hold': 20}),
    'SESS_BO':      (bt_sess_bo,      {'session_hour': 12, 'lookback': 4, 'sl_atr': 4.5, 'tp_atr': 4.0, 'trail_act': 0.14, 'trail_dist': 0.025, 'max_hold': 20}),
    'DUAL_THRUST':  (bt_dual_thrust,  {'n_bars': 6, 'k': 0.5, 'sl_atr': 4.5, 'tp_atr': 8.0, 'trail_act': 0.14, 'trail_dist': 0.025, 'max_hold': 20}),
    'CHANDELIER':   (bt_chandelier,   {'period': 22, 'mult': 3.0, 'ema_period': 100, 'sl_atr': 4.5, 'tp_atr': 8.0, 'trail_act': 0.14, 'trail_dist': 0.025, 'max_hold': 20}),
}


def run_strat(name, h1_df, spread, lot, pv, atr_min, cap=0, **overrides):
    fn, defaults = ALL_STRATS[name]
    params = dict(defaults)
    params.update(overrides)
    params['maxloss_cap'] = cap
    return fn(h1_df, spread, lot, pv, atr_min, **params)


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    results = {}

    print("=" * 80, flush=True)
    print("  R152 — Silver (XAGUSD) Strategy Migration & Validation", flush=True)
    print(f"  PV_silver={PV_SILVER}  Spread={SPREAD_SILVER}  ATR_min={ATR_MIN_SILVER}", flush=True)
    print(f"  Started: {datetime.now()}", flush=True)
    print("=" * 80, flush=True)

    # ── Phase 1: Load data ──
    print(f"\n{'='*80}", flush=True)
    print("  Phase 1: Load Silver H1 Data & Statistics", flush=True)
    print(f"{'='*80}\n", flush=True)

    h1_ag = load_h1_silver()
    print(f"  Silver H1: {len(h1_ag)} bars ({h1_ag.index[0]} ~ {h1_ag.index[-1]})", flush=True)

    atr_ag = compute_atr(h1_ag)
    print(f"  Price range: ${h1_ag['Close'].min():.2f} ~ ${h1_ag['Close'].max():.2f}", flush=True)
    print(f"  Mean price: ${h1_ag['Close'].mean():.2f}", flush=True)
    print(f"  ATR14 range: {atr_ag.min():.4f} ~ {atr_ag.max():.4f} (mean={atr_ag.mean():.4f})", flush=True)
    print(f"  Spread as % of ATR: {SPREAD_SILVER / atr_ag.mean() * 100:.1f}%", flush=True)

    results['data'] = {
        'bars': len(h1_ag),
        'date_range': f"{h1_ag.index[0]} ~ {h1_ag.index[-1]}",
        'price_range': [round(h1_ag['Close'].min(), 2), round(h1_ag['Close'].max(), 2)],
        'mean_price': round(h1_ag['Close'].mean(), 2),
        'mean_atr': round(float(atr_ag.mean()), 4),
    }

    # ── Phase 2: Baseline — gold params on silver ──
    print(f"\n{'='*80}", flush=True)
    print("  Phase 2: Run All 5 Strategies with Gold Params on Silver", flush=True)
    print(f"{'='*80}\n", flush=True)

    baseline = {}
    for name in STRAT_ORDER:
        cap = CAPS_SILVER[name]
        trades = run_strat(name, h1_ag, SPREAD_SILVER, UNIT_LOT, PV_SILVER, ATR_MIN_SILVER, cap=cap)
        stats = compute_stats(trades, name)
        baseline[name] = stats
        print(f"    {name:>14}: {stats['n_trades']:>5} trades, Sharpe={stats['sharpe']:>6.2f}, "
              f"PnL={fmt(stats['pnl'])}, MaxDD={fmt(stats['max_dd'])}, WR={stats['wr']:.1f}%", flush=True)

    results['phase2_baseline'] = baseline

    # ── Phase 3: Parameter sensitivity ──
    print(f"\n{'='*80}", flush=True)
    print("  Phase 3: Parameter Sensitivity for Silver", flush=True)
    print(f"{'='*80}\n", flush=True)

    sensitivity = {}

    # 3a. SL/TP ATR multiplier sweep (most impactful params)
    sl_grid = [3.0, 4.5, 6.0, 8.0]
    tp_grid = [4.0, 6.0, 8.0, 12.0, 16.0]

    for name in STRAT_ORDER:
        print(f"  {name} — SL/TP sweep:", flush=True)
        cap = CAPS_SILVER[name]
        best_sharpe = -999
        best_params = {}
        strat_results = []

        for sl_atr, tp_atr in product(sl_grid, tp_grid):
            if tp_atr <= sl_atr:
                continue
            trades = run_strat(name, h1_ag, SPREAD_SILVER, UNIT_LOT, PV_SILVER, ATR_MIN_SILVER,
                               cap=cap, sl_atr=sl_atr, tp_atr=tp_atr)
            stats = compute_stats(trades)
            row = {'sl_atr': sl_atr, 'tp_atr': tp_atr, **stats}
            strat_results.append(row)
            if stats['sharpe'] > best_sharpe and stats['n_trades'] >= 50:
                best_sharpe = stats['sharpe']
                best_params = {'sl_atr': sl_atr, 'tp_atr': tp_atr}

        top3 = sorted(strat_results, key=lambda x: x['sharpe'], reverse=True)[:3]
        for r in top3:
            print(f"    SL={r['sl_atr']:.1f} TP={r['tp_atr']:.1f}: "
                  f"n={r['n_trades']}, Sharpe={r['sharpe']:.2f}, PnL={fmt(r['pnl'])}", flush=True)

        sensitivity[name] = {
            'best_params': best_params,
            'best_sharpe': round(best_sharpe, 2),
            'all_results': strat_results[:10],
        }

    results['phase3_sensitivity'] = sensitivity

    # 3b. Use best params per strategy for remaining phases
    best_overrides = {}
    for name in STRAT_ORDER:
        best_overrides[name] = sensitivity[name].get('best_params', {})

    print(f"\n  Best SL/TP per strategy:", flush=True)
    for name in STRAT_ORDER:
        bp = best_overrides[name]
        print(f"    {name:>14}: SL={bp.get('sl_atr', '?')}, TP={bp.get('tp_atr', '?')}", flush=True)

    # Run with best params
    print(f"\n  Performance with optimized SL/TP:", flush=True)
    optimized = {}
    for name in STRAT_ORDER:
        cap = CAPS_SILVER[name]
        trades = run_strat(name, h1_ag, SPREAD_SILVER, UNIT_LOT, PV_SILVER, ATR_MIN_SILVER,
                           cap=cap, **best_overrides[name])
        stats = compute_stats(trades)
        optimized[name] = stats
        delta = stats['sharpe'] - baseline[name]['sharpe']
        print(f"    {name:>14}: Sharpe={stats['sharpe']:>6.2f} (delta={delta:+.2f}), "
              f"n={stats['n_trades']}, PnL={fmt(stats['pnl'])}", flush=True)

    results['phase3_optimized'] = optimized

    # ── Phase 4: K-Fold validation ──
    print(f"\n{'='*80}", flush=True)
    print("  Phase 4: K-Fold 5-Fold Validation (Optimized Params)", flush=True)
    print(f"{'='*80}\n", flush=True)

    kfold = {}
    for name in STRAT_ORDER:
        cap = CAPS_SILVER[name]
        fold_sharpes = []
        for fold_name, start, end in FOLDS:
            h1_fold = h1_ag[start:end]
            if len(h1_fold) < 100:
                fold_sharpes.append(0.0)
                continue
            trades = run_strat(name, h1_fold, SPREAD_SILVER, UNIT_LOT, PV_SILVER, ATR_MIN_SILVER,
                               cap=cap, **best_overrides[name])
            stats = compute_stats(trades)
            fold_sharpes.append(stats['sharpe'])

        positive = sum(1 for s in fold_sharpes if s > 0)
        mean_sh = float(np.mean(fold_sharpes))
        passed = positive >= 4
        status = "PASS" if passed else "FAIL"
        kfold[name] = {
            'fold_sharpes': [round(s, 2) for s in fold_sharpes],
            'positive_folds': positive,
            'mean_sharpe': round(mean_sh, 2),
            'pass': passed,
        }
        print(f"  {name:>14}: {positive}/5 positive, mean={mean_sh:.2f}  [{status}]", flush=True)
        print(f"    folds={[round(s, 1) for s in fold_sharpes]}", flush=True)

    results['phase4_kfold'] = kfold

    # ── Phase 5: Walk-Forward validation ──
    print(f"\n{'='*80}", flush=True)
    print("  Phase 5: Walk-Forward Validation (6 windows)", flush=True)
    print(f"{'='*80}\n", flush=True)

    wf_results = {}
    for name in STRAT_ORDER:
        cap = CAPS_SILVER[name]
        wf_sharpes = []
        for wf_name, tr_start, tr_end, te_start, te_end in WF_WINDOWS:
            h1_test = h1_ag[te_start:te_end]
            if len(h1_test) < 100:
                wf_sharpes.append(0.0)
                continue
            trades = run_strat(name, h1_test, SPREAD_SILVER, UNIT_LOT, PV_SILVER, ATR_MIN_SILVER,
                               cap=cap, **best_overrides[name])
            stats = compute_stats(trades)
            wf_sharpes.append(stats['sharpe'])

        positive = sum(1 for s in wf_sharpes if s > 0)
        mean_sh = float(np.mean(wf_sharpes))
        wf_results[name] = {
            'window_sharpes': [round(s, 2) for s in wf_sharpes],
            'positive_windows': positive,
            'mean_sharpe': round(mean_sh, 2),
        }
        print(f"  {name:>14}: {positive}/6 positive OOS, mean Sharpe={mean_sh:.2f}", flush=True)
        print(f"    windows={[round(s, 1) for s in wf_sharpes]}", flush=True)

    results['phase5_walk_forward'] = wf_results

    # ── Phase 6: Gold vs Silver correlation ──
    print(f"\n{'='*80}", flush=True)
    print("  Phase 6: Gold vs Silver Daily PnL Correlation", flush=True)
    print(f"{'='*80}\n", flush=True)

    h1_au = load_h1_gold()
    if h1_au is not None:
        print(f"  Gold H1: {len(h1_au)} bars loaded for comparison", flush=True)
        cross_corr = {}

        for name in STRAT_ORDER:
            cap_ag = CAPS_SILVER[name]
            trades_ag = run_strat(name, h1_ag, SPREAD_SILVER, UNIT_LOT, PV_SILVER, ATR_MIN_SILVER,
                                  cap=cap_ag, **best_overrides[name])
            ds_ag = trades_to_daily_series(trades_ag)

            trades_au = run_strat(name, h1_au, SPREAD_GOLD, UNIT_LOT, PV_GOLD, 0.1, cap=cap_ag)
            ds_au = trades_to_daily_series(trades_au)

            if len(ds_ag) > 10 and len(ds_au) > 10:
                all_dates = sorted(set(ds_ag.index) | set(ds_au.index))
                idx = pd.DatetimeIndex(all_dates)
                a_ag = ds_ag.reindex(idx, fill_value=0.0)
                a_au = ds_au.reindex(idx, fill_value=0.0)
                corr = float(a_ag.corr(a_au))
            else:
                corr = 0.0

            cross_corr[name] = round(corr, 3)
            print(f"    {name:>14}: gold-silver corr = {corr:.3f}", flush=True)

        results['phase6_cross_corr'] = cross_corr

        # Combined portfolio
        print(f"\n  Combined Gold+Silver portfolio impact:", flush=True)
        gold_daily = {}
        silver_daily = {}
        for name in STRAT_ORDER:
            cap = CAPS_SILVER[name]
            t_ag = run_strat(name, h1_ag, SPREAD_SILVER, UNIT_LOT, PV_SILVER, ATR_MIN_SILVER,
                             cap=cap, **best_overrides[name])
            t_au = run_strat(name, h1_au, SPREAD_GOLD, UNIT_LOT, PV_GOLD, 0.1, cap=cap)
            silver_daily[name] = trades_to_daily_series(t_ag)
            gold_daily[name] = trades_to_daily_series(t_au)

        all_dates = set()
        for ds in list(gold_daily.values()) + list(silver_daily.values()):
            all_dates.update(ds.index)
        all_dates = sorted(all_dates)
        idx = pd.DatetimeIndex(all_dates)

        gold_port = np.zeros(len(idx))
        silver_port = np.zeros(len(idx))
        for name in STRAT_ORDER:
            gold_port += gold_daily[name].reindex(idx, fill_value=0.0).values
            silver_port += silver_daily[name].reindex(idx, fill_value=0.0).values
        combined = gold_port + silver_port

        g_sh = sharpe(gold_port); s_sh = sharpe(silver_port); c_sh = sharpe(combined)
        g_dd = max_dd(gold_port); s_dd = max_dd(silver_port); c_dd = max_dd(combined)
        g_pnl = float(np.sum(gold_port)); s_pnl = float(np.sum(silver_port)); c_pnl = float(np.sum(combined))
        port_corr = float(pd.Series(gold_port).corr(pd.Series(silver_port)))

        print(f"    Gold-only portfolio:    Sharpe={g_sh:.2f}, PnL={fmt(g_pnl)}, MaxDD={fmt(g_dd)}", flush=True)
        print(f"    Silver-only portfolio:  Sharpe={s_sh:.2f}, PnL={fmt(s_pnl)}, MaxDD={fmt(s_dd)}", flush=True)
        print(f"    Combined Gold+Silver:   Sharpe={c_sh:.2f}, PnL={fmt(c_pnl)}, MaxDD={fmt(c_dd)}", flush=True)
        print(f"    Gold/Silver portfolio corr: {port_corr:.3f}", flush=True)
        print(f"    Diversification benefit: Sharpe delta = {c_sh - g_sh:+.2f}", flush=True)

        results['phase6_portfolio'] = {
            'gold_sharpe': round(g_sh, 2), 'silver_sharpe': round(s_sh, 2), 'combined_sharpe': round(c_sh, 2),
            'gold_pnl': round(g_pnl, 2), 'silver_pnl': round(s_pnl, 2), 'combined_pnl': round(c_pnl, 2),
            'gold_dd': round(g_dd, 2), 'silver_dd': round(s_dd, 2), 'combined_dd': round(c_dd, 2),
            'portfolio_corr': round(port_corr, 3),
        }
    else:
        print("  Gold H1 data not available — skipping cross-asset analysis.", flush=True)

    # ── Phase 7: Summary ──
    print(f"\n{'='*80}", flush=True)
    print("  FINAL SUMMARY", flush=True)
    print(f"{'='*80}\n", flush=True)

    viable = []
    for name in STRAT_ORDER:
        kf = kfold.get(name, {})
        wf = wf_results.get(name, {})
        opt = optimized.get(name, {})
        kf_pass = kf.get('pass', False)
        wf_pos = wf.get('positive_windows', 0)
        sh = opt.get('sharpe', 0)

        status = "VIABLE" if kf_pass and wf_pos >= 4 and sh > 1.0 else \
                 "MARGINAL" if kf_pass or (wf_pos >= 3 and sh > 0.5) else "REJECT"

        print(f"  {name:>14}: Sharpe={sh:.2f}, K-Fold={kf.get('positive_folds',0)}/5, "
              f"WF={wf_pos}/6  -> [{status}]", flush=True)

        if status == "VIABLE":
            viable.append(name)

    results['viable_strategies'] = viable
    print(f"\n  Viable for silver deployment: {viable if viable else 'NONE'}", flush=True)

    elapsed = time.time() - t0
    print(f"\n{'='*80}", flush=True)
    print(f"  R152 COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)
    print(f"{'='*80}", flush=True)

    results['elapsed_s'] = round(elapsed, 1)

    with open(OUTPUT_DIR / "r152_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved: {OUTPUT_DIR}/r152_results.json", flush=True)


if __name__ == "__main__":
    main()

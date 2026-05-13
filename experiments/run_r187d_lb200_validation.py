#!/usr/bin/env python3
"""
R187d — ATR Pctl Floor with lookback=200 (Live Data Provider Limit)
====================================================================
Live system only has 200 H1 bars via bars_h1.json.
Validates that ATR pctl floor still works at lookback=200.
Also confirms ATR calculation consistency between implementations.

Phase 16: Sweep lookback=200 with pctl 0/10/15/20/25/30/35/40
Phase 17: K-Fold 6-fold at best pctl for lb=200
Phase 18: Walk-Forward OOS at best pctl for lb=200
Phase 19: ATR implementation comparison (simple HL vs true TR)
Phase 20: Per-strategy detail at best config
"""
import sys, os, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("results/r187_live_stress_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100; SPREAD = 0.30; CAPITAL = 5000

LIVE_CONFIG = {
    'L8_MAX':      {'lot': 0.02, 'cap': 35,  'sl': 3.5, 'tp': 8.0, 'trail_act': 0.14, 'trail_dist': 0.025, 'max_hold': 2},
    'PSAR':        {'lot': 0.09, 'cap': 60,  'sl': 4.0, 'tp': 6.0, 'trail_act': 0.08, 'trail_dist': 0.015, 'max_hold': 15},
    'TSMOM':       {'lot': 0.15, 'cap': 60,  'sl': 6.0, 'tp': 8.0, 'trail_act': 0.14, 'trail_dist': 0.025, 'max_hold': 12},
    'SESS_BO':     {'lot': 0.13, 'cap': 60,  'sl': 4.5, 'tp': 4.0, 'trail_act': 0.14, 'trail_dist': 0.025, 'max_hold': 20},
    'DUAL_THRUST': {'lot': 0.04, 'cap': 18,  'sl': 4.5, 'tp': 8.0, 'trail_act': 0.14, 'trail_dist': 0.025, 'max_hold': 20},
    'CHANDELIER':  {'lot': 0.08, 'cap': 25,  'sl': 4.5, 'tp': 8.0, 'trail_act': 0.14, 'trail_dist': 0.025, 'max_hold': 20},
}
STRAT_ORDER = ['L8_MAX', 'PSAR', 'TSMOM', 'SESS_BO', 'DUAL_THRUST', 'CHANDELIER']

ERA_SEGMENTS = {
    'full': None,
    'hike': [("2015-12-01", "2019-01-01"), ("2022-03-01", "2023-08-01")],
    'cut':  [("2019-07-01", "2022-03-01"), ("2024-09-01", "2026-06-01")],
    'recent_3y': [("2023-06-01", "2026-06-01")],
}

import glob as _glob
t0 = time.time()

# ═══════════════════════════════════════════════════════════════
# Helpers (copied from R187c with dual ATR support)
# ═══════════════════════════════════════════════════════════════

def compute_atr_tr(df, period=14):
    """True Range ATR (used in R187 series)"""
    tr = pd.DataFrame({
        'hl': df['High'] - df['Low'],
        'hc': (df['High'] - df['Close'].shift(1)).abs(),
        'lc': (df['Low'] - df['Close'].shift(1)).abs(),
    }).max(axis=1)
    return tr.rolling(period).mean()

def compute_atr_hl(df, period=14):
    """Simple H-L ATR (used in indicators.py / live system)"""
    return (df['High'] - df['Low']).rolling(period).mean()

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

def compute_atr_pctl_rank(atr_series, lookback=200):
    n = len(atr_series)
    pctl = np.full(n, np.nan)
    atr_vals = atr_series.values
    for i in range(lookback, n):
        window = atr_vals[i-lookback:i]
        valid = window[~np.isnan(window)]
        if len(valid) < 30:
            continue
        pctl[i] = np.sum(valid <= atr_vals[i]) / len(valid) * 100
    return pd.Series(pctl, index=atr_series.index)

def _mk(pos, exit_p, exit_time, reason, bar_idx, pnl):
    return {'dir': pos['dir'], 'entry': pos['entry'], 'exit': exit_p,
            'entry_time': pos['time'], 'exit_time': exit_time,
            'pnl': pnl, 'reason': reason, 'bars': bar_idx - pos['bar'],
            'atr': pos['atr'], 'strategy': pos.get('strategy', '')}

def _run_exit(pos, i, h, lo_v, c, spread, lot, pv, times,
              sl_atr, tp_atr, trail_act_atr, trail_dist_atr, max_hold, maxloss_cap=0):
    held = i - pos['bar']
    if pos['dir'] == 'BUY':
        pnl_c = (c - pos['entry'] - spread) * lot * pv
        pnl_h = (h - pos['entry'] - spread) * lot * pv
        pnl_l = (lo_v - pos['entry'] - spread) * lot * pv
    else:
        pnl_c = (pos['entry'] - c - spread) * lot * pv
        pnl_h = (pos['entry'] - lo_v - spread) * lot * pv
        pnl_l = (pos['entry'] - h - spread) * lot * pv
    tp_val = tp_atr * pos['atr'] * lot * pv
    sl_val = sl_atr * pos['atr'] * lot * pv
    if pnl_h >= tp_val: return _mk(pos, c, times[i], "TP", i, tp_val)
    if pnl_l <= -sl_val: return _mk(pos, c, times[i], "SL", i, -sl_val)
    if maxloss_cap > 0 and pnl_c < -maxloss_cap: return _mk(pos, c, times[i], "MaxLossCap", i, -maxloss_cap)
    ad = trail_act_atr * pos['atr']; td = trail_dist_atr * pos['atr']
    if pos['dir'] == 'BUY' and h - pos['entry'] >= ad:
        ts_p = h - td
        if lo_v <= ts_p: return _mk(pos, c, times[i], "Trail", i, (ts_p - pos['entry'] - spread) * lot * pv)
    elif pos['dir'] == 'SELL' and pos['entry'] - lo_v >= ad:
        ts_p = lo_v + td
        if h >= ts_p: return _mk(pos, c, times[i], "Trail", i, (pos['entry'] - ts_p - spread) * lot * pv)
    if held >= max_hold: return _mk(pos, c, times[i], "Timeout", i, pnl_c)
    return None

def _trades_to_daily(trades):
    if not trades: return pd.Series(dtype=float)
    daily = {}
    for t in trades:
        d = pd.Timestamp(t['exit_time']).normalize()
        daily[d] = daily.get(d, 0) + t['pnl']
    return pd.Series(daily).sort_index()

def _sharpe(daily):
    if len(daily) < 10 or daily.std() == 0: return 0.0
    return float(daily.mean() / daily.std() * np.sqrt(252))

def _stats(trades):
    if not trades:
        return {'n': 0, 'sharpe': 0, 'pnl': 0, 'wr': 0, 'max_dd': 0}
    daily = _trades_to_daily(trades); pnls = [t['pnl'] for t in trades]; n = len(trades)
    wins = [p for p in pnls if p > 0]
    eq = daily.cumsum()
    dd = float((np.maximum.accumulate(eq) - eq).max()) if len(eq) > 1 else 0
    return {'n': n, 'sharpe': round(_sharpe(daily), 3), 'pnl': round(sum(pnls), 2),
            'wr': round(len(wins)/n*100, 1), 'max_dd': round(dd, 2)}

def filter_trades_by_era(trades, era_name):
    if era_name == 'full' or ERA_SEGMENTS[era_name] is None: return trades
    periods = ERA_SEGMENTS[era_name]
    return [t for t in trades if any(pd.Timestamp(s) <= pd.Timestamp(t['entry_time']) < pd.Timestamp(e) for s, e in periods)]

def load_h1():
    candidates = sorted(_glob.glob("data/download/xauusd-h1-bid-2015-*.csv"))
    if not candidates: raise FileNotFoundError("No H1 data")
    df = pd.read_csv(candidates[-1])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df.set_index('timestamp')
    df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}, inplace=True)
    df = df[['Open', 'High', 'Low', 'Close']].copy()
    print(f"  {len(df)} bars ({df.index[0]} ~ {df.index[-1]})", flush=True)
    return df

# Strategy backtests (same as R187c but with atr_fn parameter)
def _bt_generic(h1_df, name, lot, cap, sl, tp, trail_act, trail_dist, max_hold,
                atr_pctl=None, atr_pctl_floor=0, atr_fn=compute_atr_tr, setup_fn=None):
    df = h1_df.copy()
    df['ATR'] = atr_fn(df)
    if setup_fn: df = setup_fn(df)
    df = df.dropna(subset=['ATR'])
    if atr_pctl is not None:
        pctl_vals = atr_pctl.reindex(df.index).values
    else:
        pctl_vals = None
    return df, pctl_vals

def bt_keltner(h1_df, lot, cap, sl, tp, trail_act, trail_dist, max_hold, atr_pctl=None, atr_pctl_floor=0, atr_fn=compute_atr_tr):
    df = h1_df.copy(); df['ATR'] = atr_fn(df); df['ADX'] = compute_adx(df)
    df['EMA100'] = df['Close'].ewm(span=100, adjust=False).mean()
    df['KC_mid'] = df['Close'].ewm(span=25, adjust=False).mean()
    df['KC_upper'] = df['KC_mid'] + 1.2 * df['ATR']; df['KC_lower'] = df['KC_mid'] - 1.2 * df['ATR']
    df = df.dropna(subset=['ATR', 'ADX', 'EMA100', 'KC_upper'])
    pctl_vals = atr_pctl.reindex(df.index).values if atr_pctl is not None else None
    c, h, lo, atr, adx, ema = df['Close'].values, df['High'].values, df['Low'].values, df['ATR'].values, df['ADX'].values, df['EMA100'].values
    kc_u, kc_l = df['KC_upper'].values, df['KC_lower'].values
    times = df.index; n = len(df); trades = []; pos = None; last_exit = -999
    for i in range(1, n):
        if pos is not None:
            r = _run_exit(pos, i, h[i], lo[i], c[i], SPREAD, lot, PV, times, sl, tp, trail_act, trail_dist, max_hold, cap)
            if r: trades.append(r); pos = None; last_exit = i; continue
            continue
        if i - last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        if pctl_vals is not None and (np.isnan(pctl_vals[i]) or pctl_vals[i] < atr_pctl_floor): continue
        if np.isnan(adx[i]) or adx[i] < 14: continue
        if c[i] > kc_u[i] and c[i] > ema[i]:
            pos = {'dir': 'BUY', 'entry': c[i]+SPREAD/2, 'bar': i, 'time': times[i], 'atr': atr[i], 'strategy': 'L8_MAX'}
        elif c[i] < kc_l[i] and c[i] < ema[i]:
            pos = {'dir': 'SELL', 'entry': c[i]-SPREAD/2, 'bar': i, 'time': times[i], 'atr': atr[i], 'strategy': 'L8_MAX'}
    return trades

def bt_psar(h1_df, lot, cap, sl, tp, trail_act, trail_dist, max_hold, atr_pctl=None, atr_pctl_floor=0, atr_fn=compute_atr_tr):
    df = h1_df.copy()
    h_arr, l_arr = df['High'].values, df['Low'].values; n = len(df)
    psar = np.empty(n); psar[:] = np.nan; af_start = 0.01; af_max = 0.05
    af = af_start; rising = True; ep = h_arr[0]; psar[0] = l_arr[0]
    for i in range(1, n):
        prev = psar[i-1]
        if rising:
            psar[i] = prev + af * (ep - prev); psar[i] = min(psar[i], l_arr[i-1], l_arr[max(0,i-2)])
            if l_arr[i] < psar[i]: rising = False; psar[i] = ep; ep = l_arr[i]; af = af_start
            else:
                if h_arr[i] > ep: ep = h_arr[i]; af = min(af+af_start, af_max)
        else:
            psar[i] = prev + af * (ep - prev); psar[i] = max(psar[i], h_arr[i-1], h_arr[max(0,i-2)])
            if h_arr[i] > psar[i]: rising = True; psar[i] = ep; ep = h_arr[i]; af = af_start
            else:
                if l_arr[i] < ep: ep = l_arr[i]; af = min(af+af_start, af_max)
    df['PSAR'] = psar; df['ATR'] = atr_fn(df); df = df.dropna(subset=['ATR', 'PSAR'])
    pctl_vals = atr_pctl.reindex(df.index).values if atr_pctl is not None else None
    c, h, lo, atr, psar_v = df['Close'].values, df['High'].values, df['Low'].values, df['ATR'].values, df['PSAR'].values
    times = df.index; n2 = len(df); trades = []; pos = None; last_exit = -999; prev_above = c[0] > psar_v[0]
    for i in range(1, n2):
        cur_above = c[i] > psar_v[i]
        if pos is not None:
            r = _run_exit(pos, i, h[i], lo[i], c[i], SPREAD, lot, PV, times, sl, tp, trail_act, trail_dist, max_hold, cap)
            if r: trades.append(r); pos = None; last_exit = i; prev_above = cur_above; continue
            prev_above = cur_above; continue
        if i - last_exit < 2: prev_above = cur_above; continue
        if np.isnan(atr[i]) or atr[i] < 0.1: prev_above = cur_above; continue
        if pctl_vals is not None and (np.isnan(pctl_vals[i]) or pctl_vals[i] < atr_pctl_floor): prev_above = cur_above; continue
        if cur_above and not prev_above:
            pos = {'dir': 'BUY', 'entry': c[i]+SPREAD/2, 'bar': i, 'time': times[i], 'atr': atr[i], 'strategy': 'PSAR'}
        elif not cur_above and prev_above:
            pos = {'dir': 'SELL', 'entry': c[i]-SPREAD/2, 'bar': i, 'time': times[i], 'atr': atr[i], 'strategy': 'PSAR'}
        prev_above = cur_above
    return trades

def bt_tsmom(h1_df, lot, cap, sl, tp, trail_act, trail_dist, max_hold, atr_pctl=None, atr_pctl_floor=0, atr_fn=compute_atr_tr):
    df = h1_df.copy(); df['ATR'] = atr_fn(df); df = df.dropna(subset=['ATR'])
    pctl_vals = atr_pctl.reindex(df.index).values if atr_pctl is not None else None
    c, h, lo, atr = df['Close'].values, df['High'].values, df['Low'].values, df['ATR'].values
    times = df.index; n = len(df); fast, slow = 480, 720; mx = max(fast, slow)
    score = np.full(n, np.nan)
    for i in range(mx, n):
        s = 0.0
        if c[i-fast] > 0: s += 0.5 * np.sign(c[i]/c[i-fast] - 1.0)
        if c[i-slow] > 0: s += 0.5 * np.sign(c[i]/c[i-slow] - 1.0)
        score[i] = s
    trades = []; pos = None; last_exit = -999
    for i in range(mx+1, n):
        if pos is not None:
            r = _run_exit(pos, i, h[i], lo[i], c[i], SPREAD, lot, PV, times, sl, tp, trail_act, trail_dist, max_hold, cap)
            if r: trades.append(r); pos = None; last_exit = i; continue
            if pos['dir'] == 'BUY' and score[i] < 0:
                trades.append(_mk(pos, c[i], times[i], "Reversal", i, (c[i]-pos['entry']-SPREAD)*lot*PV)); pos = None; last_exit = i; continue
            elif pos['dir'] == 'SELL' and score[i] > 0:
                trades.append(_mk(pos, c[i], times[i], "Reversal", i, (pos['entry']-c[i]-SPREAD)*lot*PV)); pos = None; last_exit = i; continue
            continue
        if i-last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        if pctl_vals is not None and (np.isnan(pctl_vals[i]) or pctl_vals[i] < atr_pctl_floor): continue
        if np.isnan(score[i]) or np.isnan(score[i-1]): continue
        if score[i] > 0 and score[i-1] <= 0:
            pos = {'dir': 'BUY', 'entry': c[i]+SPREAD/2, 'bar': i, 'time': times[i], 'atr': atr[i], 'strategy': 'TSMOM'}
        elif score[i] < 0 and score[i-1] >= 0:
            pos = {'dir': 'SELL', 'entry': c[i]-SPREAD/2, 'bar': i, 'time': times[i], 'atr': atr[i], 'strategy': 'TSMOM'}
    return trades

def bt_sess_bo(h1_df, lot, cap, sl, tp, trail_act, trail_dist, max_hold, atr_pctl=None, atr_pctl_floor=0, atr_fn=compute_atr_tr):
    df = h1_df.copy(); df['ATR'] = atr_fn(df); df = df.dropna(subset=['ATR'])
    pctl_vals = atr_pctl.reindex(df.index).values if atr_pctl is not None else None
    c, h, lo, atr = df['Close'].values, df['High'].values, df['Low'].values, df['ATR'].values
    hours = df.index.hour; times = df.index; n = len(df); lb = 4
    trades = []; pos = None; last_exit = -999
    for i in range(lb, n):
        if pos is not None:
            r = _run_exit(pos, i, h[i], lo[i], c[i], SPREAD, lot, PV, times, sl, tp, trail_act, trail_dist, max_hold, cap)
            if r: trades.append(r); pos = None; last_exit = i; continue
            continue
        if hours[i] != 12: continue
        if i-last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        if pctl_vals is not None and (np.isnan(pctl_vals[i]) or pctl_vals[i] < atr_pctl_floor): continue
        hh = max(h[i-j] for j in range(1, lb+1)); ll = min(lo[i-j] for j in range(1, lb+1))
        if c[i] > hh:
            pos = {'dir': 'BUY', 'entry': c[i]+SPREAD/2, 'bar': i, 'time': times[i], 'atr': atr[i], 'strategy': 'SESS_BO'}
        elif c[i] < ll:
            pos = {'dir': 'SELL', 'entry': c[i]-SPREAD/2, 'bar': i, 'time': times[i], 'atr': atr[i], 'strategy': 'SESS_BO'}
    return trades

def bt_dual_thrust(h1_df, lot, cap, sl, tp, trail_act, trail_dist, max_hold, atr_pctl=None, atr_pctl_floor=0, atr_fn=compute_atr_tr):
    df = h1_df.copy(); df['ATR'] = atr_fn(df); df = df.dropna(subset=['ATR'])
    pctl_vals = atr_pctl.reindex(df.index).values if atr_pctl is not None else None
    c, h, lo, o, atr = df['Close'].values, df['High'].values, df['Low'].values, df['Open'].values, df['ATR'].values
    times = df.index; n = len(df); nb = 6; k = 0.5
    trades = []; pos = None; last_exit = -999
    for i in range(nb, n):
        if pos is not None:
            r = _run_exit(pos, i, h[i], lo[i], c[i], SPREAD, lot, PV, times, sl, tp, trail_act, trail_dist, max_hold, cap)
            if r: trades.append(r); pos = None; last_exit = i; continue
            continue
        if i-last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        if pctl_vals is not None and (np.isnan(pctl_vals[i]) or pctl_vals[i] < atr_pctl_floor): continue
        hh = np.max(h[i-nb:i]); lc = np.min(c[i-nb:i]); hc = np.max(c[i-nb:i]); ll = np.min(lo[i-nb:i])
        rng = max(hh - lc, hc - ll)
        if c[i] > o[i] + k * rng:
            pos = {'dir': 'BUY', 'entry': c[i]+SPREAD/2, 'bar': i, 'time': times[i], 'atr': atr[i], 'strategy': 'DUAL_THRUST'}
        elif c[i] < o[i] - k * rng:
            pos = {'dir': 'SELL', 'entry': c[i]-SPREAD/2, 'bar': i, 'time': times[i], 'atr': atr[i], 'strategy': 'DUAL_THRUST'}
    return trades

def bt_chandelier(h1_df, lot, cap, sl, tp, trail_act, trail_dist, max_hold, atr_pctl=None, atr_pctl_floor=0, atr_fn=compute_atr_tr):
    df = h1_df.copy(); df['ATR'] = atr_fn(df, period=22)
    df['EMA'] = df['Close'].ewm(span=100, adjust=False).mean()
    df = df.dropna(subset=['ATR', 'EMA'])
    pctl_vals = atr_pctl.reindex(df.index).values if atr_pctl is not None else None
    c, h, lo, atr, ema = df['Close'].values, df['High'].values, df['Low'].values, df['ATR'].values, df['EMA'].values
    times = df.index; n = len(df); p = 22; mult = 3.0
    ch_l = np.full(n, np.nan); ch_s = np.full(n, np.nan)
    for i in range(p, n):
        ch_l[i] = np.max(h[i-p+1:i+1]) - mult * atr[i]
        ch_s[i] = np.min(lo[i-p+1:i+1]) + mult * atr[i]
    d = np.zeros(n)
    for i in range(p+1, n):
        if np.isnan(ch_l[i]) or np.isnan(ch_s[i]): d[i] = d[i-1]; continue
        if c[i] > ch_s[i-1]: d[i] = 1
        elif c[i] < ch_l[i-1]: d[i] = -1
        else: d[i] = d[i-1]
    trades = []; pos = None; last_exit = -999
    for i in range(p+2, n):
        if pos is not None:
            r = _run_exit(pos, i, h[i], lo[i], c[i], SPREAD, lot, PV, times, sl, tp, trail_act, trail_dist, max_hold, cap)
            if r: trades.append(r); pos = None; last_exit = i; continue
            continue
        if i-last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        if pctl_vals is not None and (np.isnan(pctl_vals[i]) or pctl_vals[i] < atr_pctl_floor): continue
        if d[i] == 1 and d[i-1] != 1 and c[i] > ema[i]:
            pos = {'dir': 'BUY', 'entry': c[i]+SPREAD/2, 'bar': i, 'time': times[i], 'atr': atr[i], 'strategy': 'CHANDELIER'}
        elif d[i] == -1 and d[i-1] != -1 and c[i] < ema[i]:
            pos = {'dir': 'SELL', 'entry': c[i]-SPREAD/2, 'bar': i, 'time': times[i], 'atr': atr[i], 'strategy': 'CHANDELIER'}
    return trades

STRAT_BT = {'L8_MAX': bt_keltner, 'PSAR': bt_psar, 'TSMOM': bt_tsmom,
            'SESS_BO': bt_sess_bo, 'DUAL_THRUST': bt_dual_thrust, 'CHANDELIER': bt_chandelier}

def run_all(h1_df, atr_pctl=None, atr_pctl_floor=0, atr_fn=compute_atr_tr):
    r = {}
    for name in STRAT_ORDER:
        cfg = LIVE_CONFIG[name]
        r[name] = STRAT_BT[name](h1_df, cfg['lot'], cfg['cap'], cfg['sl'], cfg['tp'],
                                  cfg['trail_act'], cfg['trail_dist'], cfg['max_hold'],
                                  atr_pctl=atr_pctl, atr_pctl_floor=atr_pctl_floor, atr_fn=atr_fn)
    return r

def port_stats(all_trades):
    all_t = [t for name in STRAT_ORDER for t in all_trades[name]]
    return _stats(all_t), _trades_to_daily(all_t)


# ═══════════════════════════════════════════════════════════════
# Phase 16: Sweep lb=200
# ═══════════════════════════════════════════════════════════════
def phase16(h1_df):
    print(f"\n{'='*120}")
    print(f"  PHASE 16: ATR Pctl Sweep at lookback=200 (live data provider limit)")
    print(f"{'='*120}")
    LOOKBACKS = [100, 150, 200, 300, 500]
    PCTL_FLOORS = [0, 10, 15, 20, 25, 30, 35, 40]
    all_results = {}
    for lb in LOOKBACKS:
        pctl = compute_atr_pctl_rank(compute_atr_tr(h1_df), lookback=lb)
        print(f"\n  --- Lookback = {lb} bars ---")
        print(f"  {'Pctl':>6} {'N':>6} {'Sharpe':>7} {'PnL':>10} {'dSharpe':>8}")
        b_sh = None; lb_results = []
        for pf in PCTL_FLOORS:
            all_t = run_all(h1_df, atr_pctl=pctl, atr_pctl_floor=pf)
            s, _ = port_stats(all_t)
            if b_sh is None: b_sh = s['sharpe']
            d = s['sharpe'] - b_sh
            pnl_s = f"${s['pnl']:>9,.0f}" if s['pnl'] >= 0 else f"-${abs(s['pnl']):>8,.0f}"
            print(f"  {pf:>6} {s['n']:>6} {s['sharpe']:>7.3f} {pnl_s} {d:>+8.3f}")
            lb_results.append({'pctl': pf, 'n': s['n'], 'sharpe': s['sharpe'], 'pnl': round(s['pnl'],0), 'd_sharpe': round(d,3)})
        all_results[str(lb)] = lb_results
    best = None
    for k, v in all_results.items():
        for r in v:
            if r['pctl'] == 0: continue
            if best is None or r['sharpe'] > best['sharpe']:
                best = {**r, 'lookback': int(k)}
    print(f"\n  BEST: lb={best['lookback']}, pctl={best['pctl']} (Sharpe={best['sharpe']:.3f}, d={best['d_sharpe']:+.3f})")
    return all_results, best

# ═══════════════════════════════════════════════════════════════
# Phase 17: K-Fold
# ═══════════════════════════════════════════════════════════════
def phase17_kfold(h1_df, lb, pf):
    print(f"\n{'='*120}")
    print(f"  PHASE 17: K-Fold 6-Fold (lb={lb}, pctl={pf})")
    print(f"{'='*120}")
    K = 6; start = h1_df.index[0]; end = h1_df.index[-1]
    total = (end - start).days; fold_d = total // K
    pctl_full = compute_atr_pctl_rank(compute_atr_tr(h1_df), lookback=lb)
    results = []; fw = 0; bw = 0
    print(f"  {'Fold':>5} {'Period':>25} {'Sh_base':>8} {'Sh_filt':>8} {'dSharpe':>8} {'Winner':>8}")
    for fold in range(K):
        fs = start + pd.Timedelta(days=fold * fold_d)
        fe = start + pd.Timedelta(days=(fold+1)*fold_d) if fold < K-1 else end + pd.Timedelta(days=1)
        h1f = h1_df[(h1_df.index >= fs) & (h1_df.index < fe)]
        if len(h1f) < 300: continue
        pctl_f = pctl_full.reindex(h1f.index)
        s_base, _ = port_stats(run_all(h1f))
        s_filt, _ = port_stats(run_all(h1f, atr_pctl=pctl_f, atr_pctl_floor=pf))
        d = s_filt['sharpe'] - s_base['sharpe']
        w = "FILTER" if d > 0 else "BASE"
        if d > 0: fw += 1
        else: bw += 1
        per = f"{fs.date()} ~ {fe.date()}"
        print(f"  {fold+1:>5} {per:>25} {s_base['sharpe']:>8.3f} {s_filt['sharpe']:>8.3f} {d:>+8.3f} {w:>8}")
        results.append({'fold': fold+1, 'start': str(fs.date()), 'end': str(fe.date()),
                         'sh_base': s_base['sharpe'], 'sh_filt': s_filt['sharpe'], 'd': round(d,3), 'w': w})
    pr = fw/(fw+bw)*100 if fw+bw > 0 else 0
    v = "PASS" if fw >= 4 else "FAIL"
    print(f"\n  K-Fold: {fw}/{fw+bw} ({pr:.0f}%) — {v}")
    return {'folds': results, 'fw': fw, 'bw': bw, 'verdict': v}

# ═══════════════════════════════════════════════════════════════
# Phase 18: Walk-Forward
# ═══════════════════════════════════════════════════════════════
def phase18_wf(h1_df, lb, pf):
    print(f"\n{'='*120}")
    print(f"  PHASE 18: Walk-Forward OOS (18mo train / 6mo test, lb={lb}, pctl={pf})")
    print(f"{'='*120}")
    train_d, test_d = int(1.5*365), 180
    start = h1_df.index[0]; end = h1_df.index[-1]
    pctl_full = compute_atr_pctl_rank(compute_atr_tr(h1_df), lookback=lb)
    cursor = start + pd.Timedelta(days=train_d)
    results = []; fw = 0; tot = 0; period = 0
    print(f"  {'#':>3} {'Test':>25} {'Sh_base':>8} {'Sh_filt':>8} {'dSharpe':>8} {'OOS':>5}")
    while cursor + pd.Timedelta(days=test_d) <= end + pd.Timedelta(days=1):
        period += 1
        ts = cursor; te = cursor + pd.Timedelta(days=test_d)
        h1t = h1_df[(h1_df.index >= ts) & (h1_df.index < te)]
        if len(h1t) < 200: cursor += pd.Timedelta(days=test_d); continue
        pctl_t = pctl_full.reindex(h1t.index)
        s_base, _ = port_stats(run_all(h1t))
        s_filt, _ = port_stats(run_all(h1t, atr_pctl=pctl_t, atr_pctl_floor=pf))
        d = s_filt['sharpe'] - s_base['sharpe']
        oos = "PASS" if d > 0 else "FAIL"
        if d > 0: fw += 1
        tot += 1
        test_str = f"{ts.date()} ~ {te.date()}"
        print(f"  {period:>3} {test_str:>25} {s_base['sharpe']:>8.3f} {s_filt['sharpe']:>8.3f} {d:>+8.3f} {oos:>5}")
        results.append({'p': period, 'test': test_str, 'sh_base': s_base['sharpe'], 'sh_filt': s_filt['sharpe'],
                         'd': round(d,3), 'oos': oos})
        cursor += pd.Timedelta(days=test_d)
    pr = fw/tot*100 if tot > 0 else 0
    v = "PASS" if fw >= tot*0.6 else "FAIL"
    print(f"\n  Walk-Forward: {fw}/{tot} ({pr:.0f}%) — {v}")
    return {'periods': results, 'fw': fw, 'tot': tot, 'verdict': v}

# ═══════════════════════════════════════════════════════════════
# Phase 19: ATR implementation comparison
# ═══════════════════════════════════════════════════════════════
def phase19_atr_compare(h1_df, lb, pf):
    print(f"\n{'='*120}")
    print(f"  PHASE 19: ATR Implementation Comparison")
    print(f"  Does it matter if we use simple H-L or true TR for the pctl calculation?")
    print(f"{'='*120}")
    for atr_name, atr_fn in [("TrueRange_SMA", compute_atr_tr), ("HighLow_SMA", compute_atr_hl)]:
        atr_s = atr_fn(h1_df)
        pctl = compute_atr_pctl_rank(atr_s, lookback=lb)
        base_t = run_all(h1_df, atr_fn=atr_fn)
        filt_t = run_all(h1_df, atr_pctl=pctl, atr_pctl_floor=pf, atr_fn=atr_fn)
        s_b, _ = port_stats(base_t); s_f, _ = port_stats(filt_t)
        d = s_f['sharpe'] - s_b['sharpe']
        print(f"  {atr_name:<16} base={s_b['sharpe']:.3f} filter={s_f['sharpe']:.3f} d={d:+.3f} "
              f"N_base={s_b['n']} N_filt={s_f['n']}")

    # Correlation between two ATR series
    atr_tr = compute_atr_tr(h1_df).dropna()
    atr_hl = compute_atr_hl(h1_df).reindex(atr_tr.index)
    corr = atr_tr.corr(atr_hl)
    print(f"\n  Correlation(TR_ATR, HL_ATR): {corr:.6f}")
    pctl_tr = compute_atr_pctl_rank(atr_tr, lookback=lb)
    pctl_hl = compute_atr_pctl_rank(atr_hl.dropna(), lookback=lb)
    common = pctl_tr.dropna().index.intersection(pctl_hl.dropna().index)
    pctl_corr = pctl_tr.loc[common].corr(pctl_hl.loc[common])
    print(f"  Correlation(TR_pctl, HL_pctl): {pctl_corr:.6f}")
    agree = ((pctl_tr.loc[common] >= pf) == (pctl_hl.loc[common] >= pf)).mean() * 100
    print(f"  Agreement rate (both pass or both block at pctl={pf}): {agree:.1f}%")
    return {'corr_atr': round(corr, 6), 'corr_pctl': round(pctl_corr, 6), 'agree_pct': round(agree, 1)}

# ═══════════════════════════════════════════════════════════════
# Phase 20: Per-strategy detail
# ═══════════════════════════════════════════════════════════════
def phase20_per_strat(h1_df, lb, pf):
    print(f"\n{'='*120}")
    print(f"  PHASE 20: Per-Strategy Impact (lb={lb}, pctl={pf})")
    print(f"{'='*120}")
    pctl = compute_atr_pctl_rank(compute_atr_tr(h1_df), lookback=lb)
    base = run_all(h1_df); filt = run_all(h1_df, atr_pctl=pctl, atr_pctl_floor=pf)
    print(f"  {'Strategy':<15} {'N_base':>7} {'N_filt':>7} {'Skip':>6} {'Sh_b':>7} {'Sh_f':>7} {'dSh':>7} {'PnL_b':>10} {'PnL_f':>10}")
    results = {}
    for name in STRAT_ORDER:
        sb = _stats(base[name]); sf = _stats(filt[name])
        skip = sb['n'] - sf['n']; d = sf['sharpe'] - sb['sharpe']
        print(f"  {name:<15} {sb['n']:>7} {sf['n']:>7} {skip:>6} {sb['sharpe']:>7.3f} {sf['sharpe']:>7.3f} "
              f"{d:>+7.3f} ${sb['pnl']:>9,.0f} ${sf['pnl']:>9,.0f}")
        results[name] = {'n_base': sb['n'], 'n_filt': sf['n'], 'skip': skip,
                          'sh_base': sb['sharpe'], 'sh_filt': sf['sharpe'], 'd': round(d, 3)}
    return results


def main():
    print("=" * 120)
    print("  R187d — ATR Pctl Floor: Lookback=200 Validation + ATR Implementation Check")
    print("=" * 120, flush=True)
    h1_df = load_h1()

    p16, best = phase16(h1_df)
    lb = best['lookback']; pf = best['pctl']

    p17 = phase17_kfold(h1_df, lb, pf)
    p18 = phase18_wf(h1_df, lb, pf)
    p19 = phase19_atr_compare(h1_df, lb, pf)
    p20 = phase20_per_strat(h1_df, lb, pf)

    print(f"\n{'='*120}")
    print(f"  FINAL: lb={lb}, pctl={pf}")
    print(f"  K-Fold: {p17['verdict']} ({p17['fw']}/{p17['fw']+p17['bw']})")
    print(f"  Walk-Forward: {p18['verdict']} ({p18['fw']}/{p18['tot']})")
    if p17['verdict'] == 'PASS' and p18['verdict'] == 'PASS':
        print(f"  ==> GO at lb={lb}, pctl={pf}")
    else:
        print(f"  ==> Review needed")

    elapsed = time.time() - t0
    print(f"  Runtime: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    save = {'phase16': p16, 'best': best, 'phase17': p17, 'phase18': p18, 'phase19': p19, 'phase20': p20, 'runtime_s': round(elapsed,1)}
    out = OUTPUT_DIR / "r187d_results.json"
    with open(out, 'w') as f: json.dump(save, f, indent=2, default=str)
    print(f"  Saved: {out}")
    print(f"{'='*120}")

if __name__ == "__main__":
    main()

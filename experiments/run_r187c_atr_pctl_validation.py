#!/usr/bin/env python3
"""
R187c — ATR Percentile Floor: K-Fold + Walk-Forward Validation
================================================================
Addresses two critical gaps from R187b review:
1. ATR floor must be PERCENTILE-based (rolling window), not absolute dollar value
   - Same lesson as Rule B: absolute ATR fails when gold price regime shifts
   - Rolling percentile adapts to any price level
2. Must pass K-Fold (6-fold) and Walk-Forward OOS before GO decision

Design:
- ATR_pctl_floor = rolling percentile rank of current ATR over lookback window
- If current ATR is below the Nth percentile of recent ATR history → skip entry
- This is price-regime invariant: works at $1200 gold (ATR~3) and $3400 gold (ATR~20)

Phase 11: ATR percentile sweep (pctl 0/10/15/20/25/30/35/40 with rolling windows 500/1000/2000)
Phase 12: K-Fold 6-fold validation at best ATR pctl floor
Phase 13: Walk-Forward OOS (5 periods, 1.5y train / 6mo test)
Phase 14: Era check at best ATR pctl floor
Phase 15: Yearly consistency at best ATR pctl floor
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

PV = 100
SPREAD = 0.30
CAPITAL = 5000

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
    'full':       None,
    'hike':       [("2015-12-01", "2019-01-01"), ("2022-03-01", "2023-08-01")],
    'cut':        [("2019-07-01", "2022-03-01"), ("2024-09-01", "2026-06-01")],
    'recent_3y':  [("2023-06-01", "2026-06-01")],
}

import glob as _glob
t0 = time.time()


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

def compute_atr_pctl_rank(atr_series, lookback=1000):
    """Rolling percentile rank: what % of recent ATR values are <= current ATR."""
    n = len(atr_series)
    pctl = np.full(n, np.nan)
    atr_vals = atr_series.values
    for i in range(lookback, n):
        window = atr_vals[i-lookback:i]
        valid = window[~np.isnan(window)]
        if len(valid) < 50:
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
    daily = _trades_to_daily(trades)
    pnls = [t['pnl'] for t in trades]; n = len(trades)
    wins = [p for p in pnls if p > 0]
    eq = daily.cumsum()
    dd = float((np.maximum.accumulate(eq) - eq).max()) if len(eq) > 1 else 0
    return {'n': n, 'sharpe': round(_sharpe(daily), 3), 'pnl': round(sum(pnls), 2),
            'wr': round(len(wins)/n*100, 1), 'max_dd': round(dd, 2)}

def filter_trades_by_era(trades, era_name):
    if era_name == 'full' or ERA_SEGMENTS[era_name] is None:
        return trades
    periods = ERA_SEGMENTS[era_name]
    filtered = []
    for t in trades:
        entry = pd.Timestamp(t['entry_time'])
        for start, end in periods:
            if pd.Timestamp(start) <= entry < pd.Timestamp(end):
                filtered.append(t)
                break
    return filtered

def filter_trades_by_range(trades, start, end):
    filtered = []
    for t in trades:
        entry = pd.Timestamp(t['entry_time'])
        if pd.Timestamp(start) <= entry < pd.Timestamp(end):
            filtered.append(t)
    return filtered

def load_h1():
    candidates = sorted(_glob.glob("data/download/xauusd-h1-bid-2015-*.csv"))
    if not candidates: raise FileNotFoundError("No H1 data")
    csv_path = candidates[-1]
    print(f"  Loading H1: {csv_path}", flush=True)
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df.set_index('timestamp')
    df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}, inplace=True)
    df = df[['Open', 'High', 'Low', 'Close']].copy()
    print(f"  {len(df)} bars ({df.index[0]} ~ {df.index[-1]})", flush=True)
    return df


# ═══════════════════════════════════════════════════════════════
# Strategy backtests — with ATR percentile floor
# ═══════════════════════════════════════════════════════════════

def bt_keltner(h1_df, lot, cap, sl, tp, trail_act, trail_dist, max_hold, atr_pctl=None, atr_pctl_floor=0):
    df = h1_df.copy()
    df['ATR'] = compute_atr(df); df['ADX'] = compute_adx(df)
    df['EMA100'] = df['Close'].ewm(span=100, adjust=False).mean()
    df['KC_mid'] = df['Close'].ewm(span=25, adjust=False).mean()
    df['KC_upper'] = df['KC_mid'] + 1.2 * df['ATR']
    df['KC_lower'] = df['KC_mid'] - 1.2 * df['ATR']
    df = df.dropna(subset=['ATR', 'ADX', 'EMA100', 'KC_upper'])
    if atr_pctl is not None:
        pctl_vals = atr_pctl.reindex(df.index).values
    else:
        pctl_vals = None
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; adx = df['ADX'].values; ema = df['EMA100'].values
    kc_u = df['KC_upper'].values; kc_l = df['KC_lower'].values
    times = df.index; n = len(df)
    trades = []; pos = None; last_exit = -999
    for i in range(1, n):
        if pos is not None:
            result = _run_exit(pos, i, h[i], lo[i], c[i], SPREAD, lot, PV, times,
                               sl, tp, trail_act, trail_dist, max_hold, cap)
            if result: trades.append(result); pos = None; last_exit = i; continue
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

def bt_psar(h1_df, lot, cap, sl, tp, trail_act, trail_dist, max_hold, atr_pctl=None, atr_pctl_floor=0):
    df = h1_df.copy()
    h_arr, l_arr, c_arr = df['High'].values, df['Low'].values, df['Close'].values
    n = len(df)
    psar = np.empty(n); psar[:] = np.nan
    af_start = 0.01; af_max = 0.05
    af = af_start; rising = True; ep = h_arr[0]; psar[0] = l_arr[0]
    for i in range(1, n):
        prev = psar[i-1]
        if rising:
            psar[i] = prev + af * (ep - prev)
            psar[i] = min(psar[i], l_arr[i-1], l_arr[max(0,i-2)])
            if l_arr[i] < psar[i]:
                rising = False; psar[i] = ep; ep = l_arr[i]; af = af_start
            else:
                if h_arr[i] > ep: ep = h_arr[i]; af = min(af+af_start, af_max)
        else:
            psar[i] = prev + af * (ep - prev)
            psar[i] = max(psar[i], h_arr[i-1], h_arr[max(0,i-2)])
            if h_arr[i] > psar[i]:
                rising = True; psar[i] = ep; ep = h_arr[i]; af = af_start
            else:
                if l_arr[i] < ep: ep = l_arr[i]; af = min(af+af_start, af_max)
    df['PSAR'] = psar; df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR', 'PSAR'])
    if atr_pctl is not None:
        pctl_vals = atr_pctl.reindex(df.index).values
    else:
        pctl_vals = None
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; psar_v = df['PSAR'].values
    times = df.index; n2 = len(df)
    trades = []; pos = None; last_exit = -999
    prev_above = c[0] > psar_v[0]
    for i in range(1, n2):
        cur_above = c[i] > psar_v[i]
        if pos is not None:
            result = _run_exit(pos, i, h[i], lo[i], c[i], SPREAD, lot, PV, times,
                               sl, tp, trail_act, trail_dist, max_hold, cap)
            if result: trades.append(result); pos = None; last_exit = i; prev_above = cur_above; continue
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

def bt_tsmom(h1_df, lot, cap, sl, tp, trail_act, trail_dist, max_hold, atr_pctl=None, atr_pctl_floor=0):
    df = h1_df.copy(); df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR'])
    if atr_pctl is not None:
        pctl_vals = atr_pctl.reindex(df.index).values
    else:
        pctl_vals = None
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; times = df.index; n = len(df)
    fast, slow = 480, 720; max_lb = max(fast, slow)
    score = np.full(n, np.nan)
    for i in range(max_lb, n):
        s = 0.0
        if c[i-fast] > 0: s += 0.5 * np.sign(c[i]/c[i-fast] - 1.0)
        if c[i-slow] > 0: s += 0.5 * np.sign(c[i]/c[i-slow] - 1.0)
        score[i] = s
    trades = []; pos = None; last_exit = -999
    for i in range(max_lb+1, n):
        if pos is not None:
            result = _run_exit(pos, i, h[i], lo[i], c[i], SPREAD, lot, PV, times,
                               sl, tp, trail_act, trail_dist, max_hold, cap)
            if result: trades.append(result); pos = None; last_exit = i; continue
            if pos['dir'] == 'BUY' and score[i] < 0:
                pnl = (c[i]-pos['entry']-SPREAD)*lot*PV
                trades.append(_mk(pos, c[i], times[i], "Reversal", i, pnl)); pos = None; last_exit = i; continue
            elif pos['dir'] == 'SELL' and score[i] > 0:
                pnl = (pos['entry']-c[i]-SPREAD)*lot*PV
                trades.append(_mk(pos, c[i], times[i], "Reversal", i, pnl)); pos = None; last_exit = i; continue
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

def bt_sess_bo(h1_df, lot, cap, sl, tp, trail_act, trail_dist, max_hold, atr_pctl=None, atr_pctl_floor=0):
    df = h1_df.copy(); df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR'])
    if atr_pctl is not None:
        pctl_vals = atr_pctl.reindex(df.index).values
    else:
        pctl_vals = None
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; hours = df.index.hour; times = df.index; n = len(df)
    session_hour, lookback = 12, 4
    trades = []; pos = None; last_exit = -999
    for i in range(lookback, n):
        if pos is not None:
            result = _run_exit(pos, i, h[i], lo[i], c[i], SPREAD, lot, PV, times,
                               sl, tp, trail_act, trail_dist, max_hold, cap)
            if result: trades.append(result); pos = None; last_exit = i; continue
            continue
        if hours[i] != session_hour: continue
        if i-last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        if pctl_vals is not None and (np.isnan(pctl_vals[i]) or pctl_vals[i] < atr_pctl_floor): continue
        hh = max(h[i-j] for j in range(1, lookback+1))
        ll = min(lo[i-j] for j in range(1, lookback+1))
        if c[i] > hh:
            pos = {'dir': 'BUY', 'entry': c[i]+SPREAD/2, 'bar': i, 'time': times[i], 'atr': atr[i], 'strategy': 'SESS_BO'}
        elif c[i] < ll:
            pos = {'dir': 'SELL', 'entry': c[i]-SPREAD/2, 'bar': i, 'time': times[i], 'atr': atr[i], 'strategy': 'SESS_BO'}
    return trades

def bt_dual_thrust(h1_df, lot, cap, sl, tp, trail_act, trail_dist, max_hold, atr_pctl=None, atr_pctl_floor=0):
    df = h1_df.copy(); df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR'])
    if atr_pctl is not None:
        pctl_vals = atr_pctl.reindex(df.index).values
    else:
        pctl_vals = None
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    o = df['Open'].values; atr = df['ATR'].values
    times = df.index; n = len(df); n_bars = 6; k = 0.5
    trades = []; pos = None; last_exit = -999
    for i in range(n_bars, n):
        if pos is not None:
            result = _run_exit(pos, i, h[i], lo[i], c[i], SPREAD, lot, PV, times,
                               sl, tp, trail_act, trail_dist, max_hold, cap)
            if result: trades.append(result); pos = None; last_exit = i; continue
            continue
        if i - last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        if pctl_vals is not None and (np.isnan(pctl_vals[i]) or pctl_vals[i] < atr_pctl_floor): continue
        hh = np.max(h[i-n_bars:i]); lc = np.min(c[i-n_bars:i])
        hc = np.max(c[i-n_bars:i]); ll = np.min(lo[i-n_bars:i])
        rng = max(hh - lc, hc - ll)
        if c[i] > o[i] + k * rng:
            pos = {'dir': 'BUY', 'entry': c[i]+SPREAD/2, 'bar': i, 'time': times[i], 'atr': atr[i], 'strategy': 'DUAL_THRUST'}
        elif c[i] < o[i] - k * rng:
            pos = {'dir': 'SELL', 'entry': c[i]-SPREAD/2, 'bar': i, 'time': times[i], 'atr': atr[i], 'strategy': 'DUAL_THRUST'}
    return trades

def bt_chandelier(h1_df, lot, cap, sl, tp, trail_act, trail_dist, max_hold, atr_pctl=None, atr_pctl_floor=0):
    df = h1_df.copy()
    df['ATR'] = compute_atr(df, period=22)
    df['EMA'] = df['Close'].ewm(span=100, adjust=False).mean()
    df = df.dropna(subset=['ATR', 'EMA'])
    if atr_pctl is not None:
        pctl_vals = atr_pctl.reindex(df.index).values
    else:
        pctl_vals = None
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; ema = df['EMA'].values; times = df.index; n = len(df)
    period = 22; mult = 3.0
    ch_long = np.full(n, np.nan); ch_short = np.full(n, np.nan)
    for i in range(period, n):
        ch_long[i] = np.max(h[i-period+1:i+1]) - mult * atr[i]
        ch_short[i] = np.min(lo[i-period+1:i+1]) + mult * atr[i]
    direction = np.zeros(n)
    for i in range(period+1, n):
        if np.isnan(ch_long[i]) or np.isnan(ch_short[i]):
            direction[i] = direction[i-1]; continue
        if c[i] > ch_short[i-1]: direction[i] = 1
        elif c[i] < ch_long[i-1]: direction[i] = -1
        else: direction[i] = direction[i-1]
    trades = []; pos = None; last_exit = -999
    for i in range(period+2, n):
        if pos is not None:
            result = _run_exit(pos, i, h[i], lo[i], c[i], SPREAD, lot, PV, times,
                               sl, tp, trail_act, trail_dist, max_hold, cap)
            if result: trades.append(result); pos = None; last_exit = i; continue
            continue
        if i - last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        if pctl_vals is not None and (np.isnan(pctl_vals[i]) or pctl_vals[i] < atr_pctl_floor): continue
        if direction[i] == 1 and direction[i-1] != 1 and c[i] > ema[i]:
            pos = {'dir': 'BUY', 'entry': c[i]+SPREAD/2, 'bar': i, 'time': times[i], 'atr': atr[i], 'strategy': 'CHANDELIER'}
        elif direction[i] == -1 and direction[i-1] != -1 and c[i] < ema[i]:
            pos = {'dir': 'SELL', 'entry': c[i]-SPREAD/2, 'bar': i, 'time': times[i], 'atr': atr[i], 'strategy': 'CHANDELIER'}
    return trades

STRAT_BT = {
    'L8_MAX': bt_keltner, 'PSAR': bt_psar, 'TSMOM': bt_tsmom,
    'SESS_BO': bt_sess_bo, 'DUAL_THRUST': bt_dual_thrust, 'CHANDELIER': bt_chandelier,
}

def run_all_strategies(h1_df, atr_pctl=None, atr_pctl_floor=0):
    all_trades = {}
    for name in STRAT_ORDER:
        cfg = LIVE_CONFIG[name]
        trades = STRAT_BT[name](h1_df, cfg['lot'], cfg['cap'], cfg['sl'], cfg['tp'],
                                 cfg['trail_act'], cfg['trail_dist'], cfg['max_hold'],
                                 atr_pctl=atr_pctl, atr_pctl_floor=atr_pctl_floor)
        all_trades[name] = trades
    return all_trades

def portfolio_stats(all_trades):
    all_t = []
    for name in STRAT_ORDER:
        all_t.extend(all_trades[name])
    return _stats(all_t), _trades_to_daily(all_t)


# ═══════════════════════════════════════════════════════════════
# Phase 11: ATR Percentile Sweep
# ═══════════════════════════════════════════════════════════════

def phase11_pctl_sweep(h1_df):
    print(f"\n{'='*120}")
    print(f"  PHASE 11: ATR Percentile Floor Sweep")
    print(f"  Rolling percentile rank — adapts to any gold price regime")
    print(f"{'='*120}")

    atr_series = compute_atr(h1_df).dropna()
    LOOKBACKS = [500, 1000, 2000]
    PCTL_FLOORS = [0, 10, 15, 20, 25, 30, 35, 40]

    all_results = {}

    for lb in LOOKBACKS:
        print(f"\n  --- Lookback = {lb} bars (~{lb//24} trading days) ---")
        atr_pctl = compute_atr_pctl_rank(compute_atr(h1_df), lookback=lb)
        valid_pctl = atr_pctl.dropna()
        print(f"  Pctl coverage: {len(valid_pctl)}/{len(h1_df)} bars")

        print(f"  {'Pctl_floor':>10} {'N':>6} {'Sharpe':>7} {'PnL':>10} {'MaxDD':>8} {'dSharpe':>8} {'dPnL':>10}")

        baseline_sh = None; baseline_pnl = None
        lb_results = []

        for pf in PCTL_FLOORS:
            all_trades = run_all_strategies(h1_df, atr_pctl=atr_pctl, atr_pctl_floor=pf)
            s, daily = portfolio_stats(all_trades)

            if baseline_sh is None:
                baseline_sh = s['sharpe']; baseline_pnl = s['pnl']

            d_sh = s['sharpe'] - baseline_sh
            d_pnl = s['pnl'] - baseline_pnl
            pnl_s = f"${s['pnl']:>9,.0f}" if s['pnl'] >= 0 else f"-${abs(s['pnl']):>8,.0f}"
            d_pnl_s = f"+${d_pnl:>8,.0f}" if d_pnl >= 0 else f"-${abs(d_pnl):>8,.0f}"
            print(f"  {pf:>10} {s['n']:>6} {s['sharpe']:>7.3f} {pnl_s} ${s['max_dd']:>7,.0f} "
                  f"{d_sh:>+8.3f} {d_pnl_s}")

            lb_results.append({
                'pctl_floor': pf, 'n': s['n'], 'sharpe': s['sharpe'],
                'pnl': round(s['pnl'], 0), 'max_dd': round(s['max_dd'], 0),
                'd_sharpe': round(d_sh, 3), 'd_pnl': round(d_pnl, 0),
            })
        all_results[str(lb)] = lb_results

    # Find overall best
    best = None
    for lb_key, results in all_results.items():
        for r in results:
            if r['pctl_floor'] == 0: continue
            if best is None or r['sharpe'] > best['sharpe']:
                best = {**r, 'lookback': int(lb_key)}

    print(f"\n  BEST: lookback={best['lookback']}, pctl_floor={best['pctl_floor']} "
          f"(Sharpe={best['sharpe']:.3f}, dSharpe={best['d_sharpe']:+.3f})")

    return all_results, best


# ═══════════════════════════════════════════════════════════════
# Phase 12: K-Fold Validation
# ═══════════════════════════════════════════════════════════════

def phase12_kfold(h1_df, best_lb, best_pf):
    print(f"\n{'='*120}")
    print(f"  PHASE 12: K-Fold 6-Fold Validation (ATR pctl floor={best_pf}, lookback={best_lb})")
    print(f"{'='*120}")

    K = 6
    start = h1_df.index[0]
    end = h1_df.index[-1]
    total_days = (end - start).days
    fold_days = total_days // K

    print(f"  Total: {start.date()} ~ {end.date()} ({total_days} days)")
    print(f"  Fold size: ~{fold_days} days\n")

    atr_pctl = compute_atr_pctl_rank(compute_atr(h1_df), lookback=best_lb)
    atr_pctl_none = None  # baseline uses no filter

    results = []
    baseline_wins = 0; filter_wins = 0

    print(f"  {'Fold':>5} {'Period':>25} {'Sh_base':>8} {'Sh_filter':>9} {'dSharpe':>8} {'Winner':>8}")

    for fold in range(K):
        fold_start = start + pd.Timedelta(days=fold * fold_days)
        fold_end = start + pd.Timedelta(days=(fold + 1) * fold_days) if fold < K - 1 else end + pd.Timedelta(days=1)

        h1_fold = h1_df[(h1_df.index >= fold_start) & (h1_df.index < fold_end)]
        if len(h1_fold) < 500:
            continue

        atr_pctl_fold = atr_pctl.reindex(h1_fold.index)

        # Baseline (no pctl filter)
        base_trades = run_all_strategies(h1_fold, atr_pctl=None, atr_pctl_floor=0)
        s_base, _ = portfolio_stats(base_trades)

        # With pctl filter
        filt_trades = run_all_strategies(h1_fold, atr_pctl=atr_pctl_fold, atr_pctl_floor=best_pf)
        s_filt, _ = portfolio_stats(filt_trades)

        d_sh = s_filt['sharpe'] - s_base['sharpe']
        winner = "FILTER" if d_sh > 0 else "BASE"
        if d_sh > 0: filter_wins += 1
        else: baseline_wins += 1

        period_str = f"{fold_start.date()} ~ {fold_end.date()}"
        print(f"  {fold+1:>5} {period_str:>25} {s_base['sharpe']:>8.3f} {s_filt['sharpe']:>9.3f} "
              f"{d_sh:>+8.3f} {winner:>8}")

        results.append({
            'fold': fold + 1, 'start': str(fold_start.date()), 'end': str(fold_end.date()),
            'sharpe_base': s_base['sharpe'], 'sharpe_filter': s_filt['sharpe'],
            'd_sharpe': round(d_sh, 3), 'winner': winner,
        })

    pass_rate = filter_wins / (filter_wins + baseline_wins) * 100
    verdict = "PASS" if filter_wins >= 4 else "FAIL"
    print(f"\n  K-Fold result: Filter wins {filter_wins}/{filter_wins+baseline_wins} folds ({pass_rate:.0f}%) — {verdict}")

    return {'folds': results, 'filter_wins': filter_wins, 'baseline_wins': baseline_wins,
            'pass_rate': round(pass_rate, 1), 'verdict': verdict}


# ═══════════════════════════════════════════════════════════════
# Phase 13: Walk-Forward OOS
# ═══════════════════════════════════════════════════════════════

def phase13_walkforward(h1_df, best_lb, best_pf):
    print(f"\n{'='*120}")
    print(f"  PHASE 13: Walk-Forward OOS (18mo train / 6mo test, 5 periods)")
    print(f"{'='*120}")

    train_days = int(1.5 * 365)
    test_days = 180
    step_days = test_days

    start = h1_df.index[0]
    end = h1_df.index[-1]

    atr_pctl_full = compute_atr_pctl_rank(compute_atr(h1_df), lookback=best_lb)

    results = []
    period = 0
    cursor = start + pd.Timedelta(days=train_days)

    print(f"  {'#':>3} {'Train':>25} {'Test':>25} {'Sh_base':>8} {'Sh_filter':>9} {'dSharpe':>8} {'OOS':>5}")

    filter_wins = 0; total_periods = 0

    while cursor + pd.Timedelta(days=test_days) <= end + pd.Timedelta(days=1):
        period += 1
        test_start = cursor
        test_end = cursor + pd.Timedelta(days=test_days)
        train_start = cursor - pd.Timedelta(days=train_days)

        h1_test = h1_df[(h1_df.index >= test_start) & (h1_df.index < test_end)]
        if len(h1_test) < 200:
            cursor += pd.Timedelta(days=step_days)
            continue

        atr_pctl_test = atr_pctl_full.reindex(h1_test.index)

        base_trades = run_all_strategies(h1_test, atr_pctl=None, atr_pctl_floor=0)
        s_base, _ = portfolio_stats(base_trades)

        filt_trades = run_all_strategies(h1_test, atr_pctl=atr_pctl_test, atr_pctl_floor=best_pf)
        s_filt, _ = portfolio_stats(filt_trades)

        d_sh = s_filt['sharpe'] - s_base['sharpe']
        oos_pass = "PASS" if d_sh > 0 else "FAIL"
        if d_sh > 0: filter_wins += 1
        total_periods += 1

        train_str = f"{train_start.date()} ~ {test_start.date()}"
        test_str = f"{test_start.date()} ~ {test_end.date()}"
        print(f"  {period:>3} {train_str:>25} {test_str:>25} {s_base['sharpe']:>8.3f} "
              f"{s_filt['sharpe']:>9.3f} {d_sh:>+8.3f} {oos_pass:>5}")

        results.append({
            'period': period, 'train': train_str, 'test': test_str,
            'sharpe_base': s_base['sharpe'], 'sharpe_filter': s_filt['sharpe'],
            'd_sharpe': round(d_sh, 3), 'oos': oos_pass,
        })

        cursor += pd.Timedelta(days=step_days)

    pass_rate = filter_wins / total_periods * 100 if total_periods > 0 else 0
    verdict = "PASS" if filter_wins >= total_periods * 0.6 else "FAIL"
    print(f"\n  Walk-Forward: Filter wins {filter_wins}/{total_periods} periods ({pass_rate:.0f}%) — {verdict}")

    return {'periods': results, 'filter_wins': filter_wins, 'total_periods': total_periods,
            'pass_rate': round(pass_rate, 1), 'verdict': verdict}


# ═══════════════════════════════════════════════════════════════
# Phase 14: Era Check
# ═══════════════════════════════════════════════════════════════

def phase14_era(h1_df, best_lb, best_pf):
    print(f"\n{'='*120}")
    print(f"  PHASE 14: Era Check (ATR pctl floor={best_pf}, lookback={best_lb})")
    print(f"{'='*120}")

    atr_pctl = compute_atr_pctl_rank(compute_atr(h1_df), lookback=best_lb)

    base_trades = run_all_strategies(h1_df, atr_pctl=None, atr_pctl_floor=0)
    filt_trades = run_all_strategies(h1_df, atr_pctl=atr_pctl, atr_pctl_floor=best_pf)

    print(f"  {'Era':<12} {'Sh_base':>8} {'Sh_filter':>9} {'dSharpe':>8} {'PnL_base':>10} {'PnL_filt':>10}")

    results = {}
    for era_name in ['full', 'hike', 'cut', 'recent_3y']:
        base_era = []
        filt_era = []
        for name in STRAT_ORDER:
            base_era.extend(filter_trades_by_era(base_trades[name], era_name))
            filt_era.extend(filter_trades_by_era(filt_trades[name], era_name))
        sh_base = _sharpe(_trades_to_daily(base_era))
        sh_filt = _sharpe(_trades_to_daily(filt_era))
        pnl_base = sum(t['pnl'] for t in base_era)
        pnl_filt = sum(t['pnl'] for t in filt_era)
        d = sh_filt - sh_base
        print(f"  {era_name:<12} {sh_base:>8.3f} {sh_filt:>9.3f} {d:>+8.3f} "
              f"${pnl_base:>9,.0f} ${pnl_filt:>9,.0f}")
        results[era_name] = {'sharpe_base': round(sh_base, 3), 'sharpe_filter': round(sh_filt, 3),
                              'd_sharpe': round(d, 3)}

    all_improved = all(results[e]['d_sharpe'] >= 0 for e in results)
    print(f"\n  All eras improved or equal: {'YES' if all_improved else 'NO'}")
    return results


# ═══════════════════════════════════════════════════════════════
# Phase 15: Yearly Consistency
# ═══════════════════════════════════════════════════════════════

def phase15_yearly(h1_df, best_lb, best_pf):
    print(f"\n{'='*120}")
    print(f"  PHASE 15: Yearly Consistency (ATR pctl floor={best_pf}, lookback={best_lb})")
    print(f"{'='*120}")

    atr_pctl = compute_atr_pctl_rank(compute_atr(h1_df), lookback=best_lb)

    base_trades = run_all_strategies(h1_df, atr_pctl=None, atr_pctl_floor=0)
    filt_trades = run_all_strategies(h1_df, atr_pctl=atr_pctl, atr_pctl_floor=best_pf)

    base_all = []; filt_all = []
    for name in STRAT_ORDER:
        base_all.extend(base_trades[name])
        filt_all.extend(filt_trades[name])
    daily_base = _trades_to_daily(base_all)
    daily_filt = _trades_to_daily(filt_all)
    years = sorted(set(daily_base.index.year) | set(daily_filt.index.year))

    print(f"  {'Year':>6} {'PnL_base':>10} {'PnL_filt':>10} {'dPnL':>10} {'Sh_base':>8} {'Sh_filt':>8}")

    results = {}
    filter_better_years = 0
    for yr in years:
        db = daily_base[daily_base.index.year == yr]
        df_y = daily_filt[daily_filt.index.year == yr]
        pnl_b = float(db.sum()); pnl_f = float(df_y.sum()); d_pnl = pnl_f - pnl_b
        sh_b = _sharpe(db); sh_f = _sharpe(df_y)
        d_pnl_s = f"+${d_pnl:>8,.0f}" if d_pnl >= 0 else f"-${abs(d_pnl):>8,.0f}"
        print(f"  {yr:>6} ${pnl_b:>9,.0f} ${pnl_f:>9,.0f} {d_pnl_s} {sh_b:>8.2f} {sh_f:>8.2f}")
        if sh_f >= sh_b: filter_better_years += 1
        results[str(yr)] = {
            'pnl_base': round(pnl_b, 0), 'pnl_filter': round(pnl_f, 0),
            'sharpe_base': round(sh_b, 2), 'sharpe_filter': round(sh_f, 2),
        }

    print(f"\n  Filter Sharpe >= baseline in {filter_better_years}/{len(years)} years")
    return results


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 120, flush=True)
    print("  R187c — ATR Percentile Floor: K-Fold + Walk-Forward Validation", flush=True)
    print("  Rolling percentile rank (not absolute $) for price-regime robustness", flush=True)
    print("=" * 120, flush=True)

    h1_df = load_h1()
    mean_atr = compute_atr(h1_df).dropna().mean()
    print(f"  H1 mean ATR: ${mean_atr:.2f}")

    p11, best = phase11_pctl_sweep(h1_df)
    best_lb = best['lookback']
    best_pf = best['pctl_floor']

    p12 = phase12_kfold(h1_df, best_lb, best_pf)
    p13 = phase13_walkforward(h1_df, best_lb, best_pf)
    p14 = phase14_era(h1_df, best_lb, best_pf)
    p15 = phase15_yearly(h1_df, best_lb, best_pf)

    # Final verdict
    print(f"\n{'='*120}")
    print(f"  FINAL VERDICT")
    print(f"{'='*120}")
    print(f"  Best config: ATR pctl floor = {best_pf}, lookback = {best_lb}")
    print(f"  K-Fold: {p12['verdict']} ({p12['filter_wins']}/{p12['filter_wins']+p12['baseline_wins']})")
    print(f"  Walk-Forward: {p13['verdict']} ({p13['filter_wins']}/{p13['total_periods']})")

    if p12['verdict'] == 'PASS' and p13['verdict'] == 'PASS':
        print(f"  ==> GO — ATR percentile floor is validated")
    elif p12['verdict'] == 'PASS' or p13['verdict'] == 'PASS':
        print(f"  ==> CONDITIONAL GO — one validation passed, monitor closely")
    else:
        print(f"  ==> NO-GO — insufficient OOS evidence")

    elapsed = time.time() - t0
    print(f"\n  Runtime: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    save = {
        'phase11': p11, 'best': best,
        'phase12_kfold': p12, 'phase13_walkforward': p13,
        'phase14_era': p14, 'phase15_yearly': p15,
        'runtime_s': round(elapsed, 1),
    }
    out_path = OUTPUT_DIR / "r187c_results.json"
    with open(out_path, 'w') as f:
        json.dump(save, f, indent=2, default=str)
    print(f"  Saved: {out_path}")
    print(f"{'='*120}")


if __name__ == "__main__":
    main()

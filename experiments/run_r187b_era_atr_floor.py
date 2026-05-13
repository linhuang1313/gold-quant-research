#!/usr/bin/env python3
"""
R187b — Era Segmented Analysis + ATR Floor Filter Sweep
========================================================
Extension of R187: adds mandatory 4-era breakdown and tests ATR floor
filter to address the structural low-ATR loss problem discovered in R187 Phase 3.

Phase 9: Era segmented analysis (Full / Hike / Cut / Recent 3Y) per strategy + portfolio
Phase 10: ATR floor filter sweep — test min ATR thresholds to skip low-volatility entries
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
# Shared helpers (same as R187)
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
    s = pd.Series(daily)
    return s.sort_index()

def _sharpe(daily):
    if len(daily) < 10 or daily.std() == 0: return 0.0
    return float(daily.mean() / daily.std() * np.sqrt(252))

def _stats(trades):
    if not trades:
        return {'n': 0, 'sharpe': 0, 'pnl': 0, 'wr': 0, 'max_dd': 0, 'r_mult': 0}
    daily = _trades_to_daily(trades)
    pnls = [t['pnl'] for t in trades]; n = len(trades)
    wins = [p for p in pnls if p > 0]; losses = [p for p in pnls if p <= 0]
    avg_w = np.mean(wins) if wins else 0; avg_l = abs(np.mean(losses)) if losses else 0
    r = avg_w / avg_l if avg_l > 0 else 0
    eq = daily.cumsum()
    dd = float((np.maximum.accumulate(eq) - eq).max()) if len(eq) > 1 else 0
    return {'n': n, 'sharpe': round(_sharpe(daily), 3), 'pnl': round(sum(pnls), 2),
            'wr': round(len(wins)/n*100, 1), 'max_dd': round(dd, 2), 'r_mult': round(r, 3)}

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
# Strategy backtests — with atr_floor parameter
# ═══════════════════════════════════════════════════════════════

def bt_keltner(h1_df, lot, cap, sl, tp, trail_act, trail_dist, max_hold, atr_floor=0.1):
    df = h1_df.copy()
    df['ATR'] = compute_atr(df); df['ADX'] = compute_adx(df)
    df['EMA100'] = df['Close'].ewm(span=100, adjust=False).mean()
    df['KC_mid'] = df['Close'].ewm(span=25, adjust=False).mean()
    df['KC_upper'] = df['KC_mid'] + 1.2 * df['ATR']
    df['KC_lower'] = df['KC_mid'] - 1.2 * df['ATR']
    df = df.dropna(subset=['ATR', 'ADX', 'EMA100', 'KC_upper'])
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
        if np.isnan(atr[i]) or atr[i] < atr_floor: continue
        if np.isnan(adx[i]) or adx[i] < 14: continue
        if c[i] > kc_u[i] and c[i] > ema[i]:
            pos = {'dir': 'BUY', 'entry': c[i]+SPREAD/2, 'bar': i, 'time': times[i], 'atr': atr[i], 'strategy': 'L8_MAX'}
        elif c[i] < kc_l[i] and c[i] < ema[i]:
            pos = {'dir': 'SELL', 'entry': c[i]-SPREAD/2, 'bar': i, 'time': times[i], 'atr': atr[i], 'strategy': 'L8_MAX'}
    return trades

def bt_psar(h1_df, lot, cap, sl, tp, trail_act, trail_dist, max_hold, atr_floor=0.1):
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
        if np.isnan(atr[i]) or atr[i] < atr_floor: prev_above = cur_above; continue
        if cur_above and not prev_above:
            pos = {'dir': 'BUY', 'entry': c[i]+SPREAD/2, 'bar': i, 'time': times[i], 'atr': atr[i], 'strategy': 'PSAR'}
        elif not cur_above and prev_above:
            pos = {'dir': 'SELL', 'entry': c[i]-SPREAD/2, 'bar': i, 'time': times[i], 'atr': atr[i], 'strategy': 'PSAR'}
        prev_above = cur_above
    return trades

def bt_tsmom(h1_df, lot, cap, sl, tp, trail_act, trail_dist, max_hold, atr_floor=0.1):
    df = h1_df.copy(); df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR'])
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
        if np.isnan(atr[i]) or atr[i] < atr_floor: continue
        if np.isnan(score[i]) or np.isnan(score[i-1]): continue
        if score[i] > 0 and score[i-1] <= 0:
            pos = {'dir': 'BUY', 'entry': c[i]+SPREAD/2, 'bar': i, 'time': times[i], 'atr': atr[i], 'strategy': 'TSMOM'}
        elif score[i] < 0 and score[i-1] >= 0:
            pos = {'dir': 'SELL', 'entry': c[i]-SPREAD/2, 'bar': i, 'time': times[i], 'atr': atr[i], 'strategy': 'TSMOM'}
    return trades

def bt_sess_bo(h1_df, lot, cap, sl, tp, trail_act, trail_dist, max_hold, atr_floor=0.1):
    df = h1_df.copy(); df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR'])
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
        if np.isnan(atr[i]) or atr[i] < atr_floor: continue
        hh = max(h[i-j] for j in range(1, lookback+1))
        ll = min(lo[i-j] for j in range(1, lookback+1))
        if c[i] > hh:
            pos = {'dir': 'BUY', 'entry': c[i]+SPREAD/2, 'bar': i, 'time': times[i], 'atr': atr[i], 'strategy': 'SESS_BO'}
        elif c[i] < ll:
            pos = {'dir': 'SELL', 'entry': c[i]-SPREAD/2, 'bar': i, 'time': times[i], 'atr': atr[i], 'strategy': 'SESS_BO'}
    return trades

def bt_dual_thrust(h1_df, lot, cap, sl, tp, trail_act, trail_dist, max_hold, atr_floor=0.1):
    df = h1_df.copy(); df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR'])
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
        if np.isnan(atr[i]) or atr[i] < atr_floor: continue
        hh = np.max(h[i-n_bars:i]); lc = np.min(c[i-n_bars:i])
        hc = np.max(c[i-n_bars:i]); ll = np.min(lo[i-n_bars:i])
        rng = max(hh - lc, hc - ll)
        if c[i] > o[i] + k * rng:
            pos = {'dir': 'BUY', 'entry': c[i]+SPREAD/2, 'bar': i, 'time': times[i], 'atr': atr[i], 'strategy': 'DUAL_THRUST'}
        elif c[i] < o[i] - k * rng:
            pos = {'dir': 'SELL', 'entry': c[i]-SPREAD/2, 'bar': i, 'time': times[i], 'atr': atr[i], 'strategy': 'DUAL_THRUST'}
    return trades

def bt_chandelier(h1_df, lot, cap, sl, tp, trail_act, trail_dist, max_hold, atr_floor=0.1):
    df = h1_df.copy()
    df['ATR'] = compute_atr(df, period=22)
    df['EMA'] = df['Close'].ewm(span=100, adjust=False).mean()
    df = df.dropna(subset=['ATR', 'EMA'])
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
        if np.isnan(atr[i]) or atr[i] < atr_floor: continue
        if direction[i] == 1 and direction[i-1] != 1 and c[i] > ema[i]:
            pos = {'dir': 'BUY', 'entry': c[i]+SPREAD/2, 'bar': i, 'time': times[i], 'atr': atr[i], 'strategy': 'CHANDELIER'}
        elif direction[i] == -1 and direction[i-1] != -1 and c[i] < ema[i]:
            pos = {'dir': 'SELL', 'entry': c[i]-SPREAD/2, 'bar': i, 'time': times[i], 'atr': atr[i], 'strategy': 'CHANDELIER'}
    return trades

STRAT_BT = {
    'L8_MAX': bt_keltner, 'PSAR': bt_psar, 'TSMOM': bt_tsmom,
    'SESS_BO': bt_sess_bo, 'DUAL_THRUST': bt_dual_thrust, 'CHANDELIER': bt_chandelier,
}

def run_all_strategies(h1_df, atr_floor=0.1):
    all_trades = {}
    for name in STRAT_ORDER:
        cfg = LIVE_CONFIG[name]
        trades = STRAT_BT[name](h1_df, cfg['lot'], cfg['cap'], cfg['sl'], cfg['tp'],
                                 cfg['trail_act'], cfg['trail_dist'], cfg['max_hold'], atr_floor)
        all_trades[name] = trades
    return all_trades


# ═══════════════════════════════════════════════════════════════
# Phase 9: Era Segmented Analysis
# ═══════════════════════════════════════════════════════════════

def phase9_era(all_trades):
    print(f"\n{'='*120}")
    print(f"  PHASE 9: Era Segmented Analysis (Full / Hike / Cut / Recent 3Y)")
    print(f"{'='*120}")

    results = {}

    # Per strategy per era
    for name in STRAT_ORDER:
        trades = all_trades[name]
        print(f"\n  --- {name} (lot={LIVE_CONFIG[name]['lot']}, cap=${LIVE_CONFIG[name]['cap']}) ---")
        print(f"  {'Era':<12} {'N':>6} {'Sharpe':>7} {'PnL':>10} {'WR%':>6} {'MaxDD':>8} {'R_mult':>7}")

        strat_eras = {}
        for era_name in ['full', 'hike', 'cut', 'recent_3y']:
            era_trades = filter_trades_by_era(trades, era_name)
            s = _stats(era_trades)
            pnl_s = f"${s['pnl']:>9,.0f}" if s['pnl'] >= 0 else f"-${abs(s['pnl']):>8,.0f}"
            print(f"  {era_name:<12} {s['n']:>6} {s['sharpe']:>7.3f} {pnl_s} {s['wr']:>5.1f}% ${s['max_dd']:>7,.0f} {s['r_mult']:>7.3f}")
            strat_eras[era_name] = s
        results[name] = strat_eras

    # Portfolio per era
    print(f"\n  --- PORTFOLIO ---")
    print(f"  {'Era':<12} {'N':>6} {'Sharpe':>7} {'PnL':>10} {'MaxDD':>8}")

    portfolio_eras = {}
    for era_name in ['full', 'hike', 'cut', 'recent_3y']:
        all_era_trades = []
        for name in STRAT_ORDER:
            all_era_trades.extend(filter_trades_by_era(all_trades[name], era_name))
        daily = _trades_to_daily(all_era_trades)
        sh = _sharpe(daily)
        pnl = float(daily.sum()) if len(daily) > 0 else 0
        eq = daily.cumsum() if len(daily) > 0 else pd.Series(dtype=float)
        dd = float((np.maximum.accumulate(eq) - eq).max()) if len(eq) > 1 else 0
        n_trades = len(all_era_trades)
        pnl_s = f"${pnl:>9,.0f}" if pnl >= 0 else f"-${abs(pnl):>8,.0f}"
        print(f"  {era_name:<12} {n_trades:>6} {sh:>7.3f} {pnl_s} ${dd:>7,.0f}")
        portfolio_eras[era_name] = {'n': n_trades, 'sharpe': round(sh, 3),
                                     'pnl': round(pnl, 0), 'max_dd': round(dd, 0)}
    results['PORTFOLIO'] = portfolio_eras

    # Interpretation
    port_full_sh = portfolio_eras['full']['sharpe']
    port_hike_sh = portfolio_eras['hike']['sharpe']
    port_cut_sh = portfolio_eras['cut']['sharpe']
    port_recent_sh = portfolio_eras['recent_3y']['sharpe']

    print(f"\n  Interpretation:")
    hike_ratio = port_hike_sh / port_full_sh if port_full_sh > 0 else 0
    cut_ratio = port_cut_sh / port_full_sh if port_full_sh > 0 else 0
    recent_ratio = port_recent_sh / port_full_sh if port_full_sh > 0 else 0
    print(f"    Hike/Full ratio: {hike_ratio:.2f} {'(OK)' if hike_ratio > 0.7 else '(CONCERN: rate-sensitive)'}")
    print(f"    Cut/Full ratio:  {cut_ratio:.2f} {'(OK)' if cut_ratio < 1.5 else '(CONCERN: easing-biased)'}")
    print(f"    Recent/Full:     {recent_ratio:.2f} {'(OK)' if 0.7 < recent_ratio < 1.5 else '(CONCERN: regime shift)'}")
    all_positive = all(portfolio_eras[e]['sharpe'] > 0 for e in ['full', 'hike', 'cut', 'recent_3y'])
    print(f"    All 4 eras positive: {'YES — structural edge' if all_positive else 'NO — regime-dependent'}")

    return results


# ═══════════════════════════════════════════════════════════════
# Phase 10: ATR Floor Filter Sweep
# ═══════════════════════════════════════════════════════════════

def phase10_atr_floor(h1_df):
    print(f"\n{'='*120}")
    print(f"  PHASE 10: ATR Floor Filter Sweep")
    print(f"  Testing: skip entries when ATR < threshold to avoid low-vol losses")
    print(f"{'='*120}")

    atr_series = compute_atr(h1_df).dropna()
    q10, q15, q20, q25, q30 = np.percentile(atr_series, [10, 15, 20, 25, 30])
    print(f"\n  ATR percentiles: Q10=${q10:.2f}, Q15=${q15:.2f}, Q20=${q20:.2f}, Q25=${q25:.2f}, Q30=${q30:.2f}")

    ATR_FLOORS = [0.1, 0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]

    print(f"\n  10A: Portfolio-level ATR floor sweep")
    print(f"  {'ATR_floor':>10} {'~Pctl':>6} {'N':>6} {'Sharpe':>7} {'PnL':>10} {'MaxDD':>8} {'dSharpe':>8} {'dPnL':>10}")

    baseline_sh = None
    baseline_pnl = None
    sweep_results = []

    for floor in ATR_FLOORS:
        all_trades = run_all_strategies(h1_df, atr_floor=floor)
        all_t = []
        for name in STRAT_ORDER:
            all_t.extend(all_trades[name])
        daily = _trades_to_daily(all_t)
        sh = _sharpe(daily)
        pnl = float(daily.sum()) if len(daily) > 0 else 0
        eq = daily.cumsum() if len(daily) > 0 else pd.Series(dtype=float)
        dd = float((np.maximum.accumulate(eq) - eq).max()) if len(eq) > 1 else 0
        n = len(all_t)

        pctl = float((atr_series < floor).sum() / len(atr_series) * 100)

        if baseline_sh is None:
            baseline_sh = sh; baseline_pnl = pnl

        d_sh = sh - baseline_sh
        d_pnl = pnl - baseline_pnl
        pnl_s = f"${pnl:>9,.0f}" if pnl >= 0 else f"-${abs(pnl):>8,.0f}"
        d_pnl_s = f"+${d_pnl:>8,.0f}" if d_pnl >= 0 else f"-${abs(d_pnl):>8,.0f}"
        print(f"  {floor:>10.1f} {pctl:>5.1f}% {n:>6} {sh:>7.3f} {pnl_s} ${dd:>7,.0f} "
              f"{d_sh:>+8.3f} {d_pnl_s}")

        sweep_results.append({
            'atr_floor': floor, 'pctl': round(pctl, 1), 'n': n,
            'sharpe': round(sh, 3), 'pnl': round(pnl, 0), 'max_dd': round(dd, 0),
            'd_sharpe': round(d_sh, 3), 'd_pnl': round(d_pnl, 0),
        })

    # 10B: Per-strategy impact at best portfolio floor
    best = max(sweep_results, key=lambda x: x['sharpe'])
    best_floor = best['atr_floor']
    print(f"\n  Best portfolio ATR floor: {best_floor} (Sharpe {best['sharpe']:.3f}, "
          f"dSharpe {best['d_sharpe']:+.3f})")

    print(f"\n  10B: Per-strategy impact at ATR floor = {best_floor}")
    print(f"  {'Strategy':<15} {'N_base':>7} {'N_filter':>8} {'Skipped':>8} "
          f"{'Sh_base':>8} {'Sh_filter':>9} {'dSharpe':>8}")

    per_strat = {}
    base_trades = run_all_strategies(h1_df, atr_floor=0.1)
    filt_trades = run_all_strategies(h1_df, atr_floor=best_floor)

    for name in STRAT_ORDER:
        s_base = _stats(base_trades[name])
        s_filt = _stats(filt_trades[name])
        skipped = s_base['n'] - s_filt['n']
        d_sh = s_filt['sharpe'] - s_base['sharpe']
        print(f"  {name:<15} {s_base['n']:>7} {s_filt['n']:>8} {skipped:>8} "
              f"{s_base['sharpe']:>8.3f} {s_filt['sharpe']:>9.3f} {d_sh:>+8.3f}")
        per_strat[name] = {
            'n_base': s_base['n'], 'n_filter': s_filt['n'], 'skipped': skipped,
            'sharpe_base': s_base['sharpe'], 'sharpe_filter': s_filt['sharpe'],
            'd_sharpe': round(d_sh, 3),
        }

    # 10C: Era check at best floor — make sure it doesn't break any era
    print(f"\n  10C: Era check at ATR floor = {best_floor}")
    print(f"  {'Era':<12} {'Sh_base':>8} {'Sh_filter':>9} {'dSharpe':>8}")

    era_check = {}
    for era_name in ['full', 'hike', 'cut', 'recent_3y']:
        base_era = []
        filt_era = []
        for name in STRAT_ORDER:
            base_era.extend(filter_trades_by_era(base_trades[name], era_name))
            filt_era.extend(filter_trades_by_era(filt_trades[name], era_name))
        sh_base = _sharpe(_trades_to_daily(base_era))
        sh_filt = _sharpe(_trades_to_daily(filt_era))
        d = sh_filt - sh_base
        print(f"  {era_name:<12} {sh_base:>8.3f} {sh_filt:>9.3f} {d:>+8.3f}")
        era_check[era_name] = {'sharpe_base': round(sh_base, 3), 'sharpe_filter': round(sh_filt, 3),
                                'd_sharpe': round(d, 3)}

    # 10D: Yearly consistency at best floor
    print(f"\n  10D: Yearly comparison at ATR floor = {best_floor}")
    print(f"  {'Year':>6} {'PnL_base':>10} {'PnL_filter':>11} {'dPnL':>10} {'Sh_base':>8} {'Sh_filter':>9}")

    base_all = []
    filt_all = []
    for name in STRAT_ORDER:
        base_all.extend(base_trades[name])
        filt_all.extend(filt_trades[name])
    daily_base = _trades_to_daily(base_all)
    daily_filt = _trades_to_daily(filt_all)
    years = sorted(set(daily_base.index.year) | set(daily_filt.index.year))

    yearly_comparison = {}
    for yr in years:
        db = daily_base[daily_base.index.year == yr] if len(daily_base) > 0 else pd.Series(dtype=float)
        df_y = daily_filt[daily_filt.index.year == yr] if len(daily_filt) > 0 else pd.Series(dtype=float)
        pnl_b = float(db.sum()); pnl_f = float(df_y.sum()); d_pnl = pnl_f - pnl_b
        sh_b = _sharpe(db); sh_f = _sharpe(df_y)
        d_pnl_s = f"+${d_pnl:>8,.0f}" if d_pnl >= 0 else f"-${abs(d_pnl):>8,.0f}"
        print(f"  {yr:>6} ${pnl_b:>9,.0f} ${pnl_f:>10,.0f} {d_pnl_s} {sh_b:>8.2f} {sh_f:>9.2f}")
        yearly_comparison[str(yr)] = {
            'pnl_base': round(pnl_b, 0), 'pnl_filter': round(pnl_f, 0),
            'd_pnl': round(d_pnl, 0), 'sharpe_base': round(sh_b, 2), 'sharpe_filter': round(sh_f, 2),
        }

    return {
        'sweep': sweep_results,
        'best_floor': best_floor,
        'per_strategy': per_strat,
        'era_check': era_check,
        'yearly': yearly_comparison,
    }


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 120, flush=True)
    print("  R187b — Era Segmented + ATR Floor Filter Sweep", flush=True)
    print("  Extension of R187 | Live config synced 2026-05-09", flush=True)
    print("=" * 120, flush=True)

    h1_df = load_h1()
    mean_atr = compute_atr(h1_df).dropna().mean()
    print(f"  H1 mean ATR: ${mean_atr:.2f}")

    print(f"\n  Running baseline (atr_floor=0.1)...", flush=True)
    all_trades = run_all_strategies(h1_df, atr_floor=0.1)
    for name in STRAT_ORDER:
        print(f"    {name}: {len(all_trades[name])} trades", flush=True)

    p9 = phase9_era(all_trades)
    p10 = phase10_atr_floor(h1_df)

    elapsed = time.time() - t0
    print(f"\n  Runtime: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    save = {'phase9': p9, 'phase10': p10, 'runtime_s': round(elapsed, 1)}
    out_path = OUTPUT_DIR / "r187b_results.json"
    with open(out_path, 'w') as f:
        json.dump(save, f, indent=2, default=str)
    print(f"  Saved: {out_path}")
    print(f"{'='*120}")


if __name__ == "__main__":
    main()

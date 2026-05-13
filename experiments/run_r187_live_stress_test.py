#!/usr/bin/env python3
"""
R187 — Live Portfolio Stress Test & Fragility Audit
=====================================================
Comprehensive stress test using ACTUAL live lot sizes and caps (synced 2026-05-09).
Addresses gaps in R171 (old lots) and R174 (old baselines).

Phase 1: Portfolio baseline with real lots/caps + per-strategy + yearly
Phase 2: Worst-case drawdown analysis (top-10 DD, underwater, consecutive losses)
Phase 3: High ATR / high gold price stress test
Phase 4: Strategy correlation & simultaneous loss analysis
Phase 5: Monte Carlo stress (parameter perturbation + strategy dropout)
Phase 6: Alpha decay detection (rolling Sharpe + recent vs historical)
Phase 7: Live gap quantification (if trade log exists)
Phase 8: Fragility scorecard (traffic light dashboard)
"""
import sys, os, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
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

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

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


# ═══════════════════════════════════════════════════════════════
# Strategy backtests (with live params)
# ═══════════════════════════════════════════════════════════════

def bt_keltner(h1_df, lot, cap, sl, tp, trail_act, trail_dist, max_hold):
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
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        if np.isnan(adx[i]) or adx[i] < 14: continue
        if c[i] > kc_u[i] and c[i] > ema[i]:
            pos = {'dir': 'BUY', 'entry': c[i]+SPREAD/2, 'bar': i, 'time': times[i], 'atr': atr[i], 'strategy': 'L8_MAX'}
        elif c[i] < kc_l[i] and c[i] < ema[i]:
            pos = {'dir': 'SELL', 'entry': c[i]-SPREAD/2, 'bar': i, 'time': times[i], 'atr': atr[i], 'strategy': 'L8_MAX'}
    return trades

def bt_psar(h1_df, lot, cap, sl, tp, trail_act, trail_dist, max_hold):
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
    df['PSAR'] = psar
    df['ATR'] = compute_atr(df)
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
        if np.isnan(atr[i]) or atr[i] < 0.1: prev_above = cur_above; continue
        if cur_above and not prev_above:
            pos = {'dir': 'BUY', 'entry': c[i]+SPREAD/2, 'bar': i, 'time': times[i], 'atr': atr[i], 'strategy': 'PSAR'}
        elif not cur_above and prev_above:
            pos = {'dir': 'SELL', 'entry': c[i]-SPREAD/2, 'bar': i, 'time': times[i], 'atr': atr[i], 'strategy': 'PSAR'}
        prev_above = cur_above
    return trades

def bt_tsmom(h1_df, lot, cap, sl, tp, trail_act, trail_dist, max_hold):
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
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        if np.isnan(score[i]) or np.isnan(score[i-1]): continue
        if score[i] > 0 and score[i-1] <= 0:
            pos = {'dir': 'BUY', 'entry': c[i]+SPREAD/2, 'bar': i, 'time': times[i], 'atr': atr[i], 'strategy': 'TSMOM'}
        elif score[i] < 0 and score[i-1] >= 0:
            pos = {'dir': 'SELL', 'entry': c[i]-SPREAD/2, 'bar': i, 'time': times[i], 'atr': atr[i], 'strategy': 'TSMOM'}
    return trades

def bt_sess_bo(h1_df, lot, cap, sl, tp, trail_act, trail_dist, max_hold):
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
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        hh = max(h[i-j] for j in range(1, lookback+1))
        ll = min(lo[i-j] for j in range(1, lookback+1))
        if c[i] > hh:
            pos = {'dir': 'BUY', 'entry': c[i]+SPREAD/2, 'bar': i, 'time': times[i], 'atr': atr[i], 'strategy': 'SESS_BO'}
        elif c[i] < ll:
            pos = {'dir': 'SELL', 'entry': c[i]-SPREAD/2, 'bar': i, 'time': times[i], 'atr': atr[i], 'strategy': 'SESS_BO'}
    return trades

def bt_dual_thrust(h1_df, lot, cap, sl, tp, trail_act, trail_dist, max_hold):
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
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        hh = np.max(h[i-n_bars:i]); lc = np.min(c[i-n_bars:i])
        hc = np.max(c[i-n_bars:i]); ll = np.min(lo[i-n_bars:i])
        rng = max(hh - lc, hc - ll)
        if c[i] > o[i] + k * rng:
            pos = {'dir': 'BUY', 'entry': c[i]+SPREAD/2, 'bar': i, 'time': times[i], 'atr': atr[i], 'strategy': 'DUAL_THRUST'}
        elif c[i] < o[i] - k * rng:
            pos = {'dir': 'SELL', 'entry': c[i]-SPREAD/2, 'bar': i, 'time': times[i], 'atr': atr[i], 'strategy': 'DUAL_THRUST'}
    return trades

def bt_chandelier(h1_df, lot, cap, sl, tp, trail_act, trail_dist, max_hold):
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
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        if direction[i] == 1 and direction[i-1] != 1 and c[i] > ema[i]:
            pos = {'dir': 'BUY', 'entry': c[i]+SPREAD/2, 'bar': i, 'time': times[i], 'atr': atr[i], 'strategy': 'CHANDELIER'}
        elif direction[i] == -1 and direction[i-1] != -1 and c[i] < ema[i]:
            pos = {'dir': 'SELL', 'entry': c[i]-SPREAD/2, 'bar': i, 'time': times[i], 'atr': atr[i], 'strategy': 'CHANDELIER'}
    return trades

STRAT_BT = {
    'L8_MAX': bt_keltner, 'PSAR': bt_psar, 'TSMOM': bt_tsmom,
    'SESS_BO': bt_sess_bo, 'DUAL_THRUST': bt_dual_thrust, 'CHANDELIER': bt_chandelier,
}


# ═══════════════════════════════════════════════════════════════
# Stats helpers
# ═══════════════════════════════════════════════════════════════

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
        return {'n': 0, 'sharpe': 0, 'pnl': 0, 'wr': 0, 'max_dd': 0, 'r_mult': 0, 'avg_win': 0, 'avg_loss': 0}
    daily = _trades_to_daily(trades)
    pnls = [t['pnl'] for t in trades]; n = len(trades)
    wins = [p for p in pnls if p > 0]; losses = [p for p in pnls if p <= 0]
    avg_w = np.mean(wins) if wins else 0; avg_l = abs(np.mean(losses)) if losses else 0
    r = avg_w / avg_l if avg_l > 0 else 0
    eq = daily.cumsum()
    dd = float((np.maximum.accumulate(eq) - eq).max()) if len(eq) > 1 else 0
    return {'n': n, 'sharpe': round(_sharpe(daily), 3), 'pnl': round(sum(pnls), 2),
            'wr': round(len(wins)/n*100, 1), 'max_dd': round(dd, 2),
            'r_mult': round(r, 3), 'avg_win': round(avg_w, 2), 'avg_loss': round(avg_l, 2)}

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

def run_all_strategies(h1_df):
    all_trades = {}
    for name in STRAT_ORDER:
        cfg = LIVE_CONFIG[name]
        trades = STRAT_BT[name](h1_df, cfg['lot'], cfg['cap'], cfg['sl'], cfg['tp'],
                                 cfg['trail_act'], cfg['trail_dist'], cfg['max_hold'])
        all_trades[name] = trades
    return all_trades


# ═══════════════════════════════════════════════════════════════
# Phase 1: Portfolio Baseline
# ═══════════════════════════════════════════════════════════════

def phase1_baseline(h1_df, all_trades):
    print(f"\n{'='*120}")
    print(f"  PHASE 1: Portfolio Baseline (Real Live Lots & Caps)")
    print(f"{'='*120}")

    print(f"\n  {'Strategy':<15} {'Lot':>5} {'Cap':>5} {'N':>6} {'Sharpe':>7} {'PnL':>10} {'WR%':>6} "
          f"{'MaxDD':>8} {'R_mult':>7} {'AvgWin':>8} {'AvgLoss':>8}")
    total_daily = pd.Series(dtype=float)
    results = {}
    for name in STRAT_ORDER:
        cfg = LIVE_CONFIG[name]
        s = _stats(all_trades[name])
        d = _trades_to_daily(all_trades[name])
        total_daily = total_daily.add(d, fill_value=0)
        pnl_s = f"${s['pnl']:>9,.0f}" if s['pnl'] >= 0 else f"-${abs(s['pnl']):>8,.0f}"
        print(f"  {name:<15} {cfg['lot']:>5.2f} ${cfg['cap']:>4} {s['n']:>6} {s['sharpe']:>7.3f} {pnl_s} "
              f"{s['wr']:>5.1f}% ${s['max_dd']:>7,.0f} {s['r_mult']:>7.3f} ${s['avg_win']:>7.2f} ${s['avg_loss']:>7.2f}")
        results[name] = s

    total_daily = total_daily.sort_index()
    port_sharpe = _sharpe(total_daily)
    port_pnl = float(total_daily.sum())
    eq = total_daily.cumsum()
    port_dd = float((np.maximum.accumulate(eq) - eq).max())
    port_dd_pct = port_dd / CAPITAL * 100

    print(f"\n  PORTFOLIO: Sharpe={port_sharpe:.3f}, PnL=${port_pnl:,.0f}, "
          f"MaxDD=${port_dd:,.0f} ({port_dd_pct:.1f}% of ${CAPITAL:,})")

    # Yearly breakdown
    years = sorted(set(total_daily.index.year))
    print(f"\n  {'Year':>6} {'PnL':>10} {'Sharpe':>7} {'MaxDD':>8} {'Days':>5}")
    yearly = {}
    for yr in years:
        yr_d = total_daily[total_daily.index.year == yr]
        yr_pnl = float(yr_d.sum())
        yr_sh = _sharpe(yr_d)
        yr_eq = yr_d.cumsum()
        yr_dd = float((np.maximum.accumulate(yr_eq) - yr_eq).max()) if len(yr_eq) > 1 else 0
        print(f"  {yr:>6} ${yr_pnl:>9,.0f} {yr_sh:>7.2f} ${yr_dd:>7,.0f} {len(yr_d):>5}")
        yearly[yr] = {'pnl': round(yr_pnl, 0), 'sharpe': round(yr_sh, 2), 'max_dd': round(yr_dd, 0)}

    return {'strategies': results, 'portfolio': {
        'sharpe': round(port_sharpe, 3), 'pnl': round(port_pnl, 0),
        'max_dd': round(port_dd, 0), 'max_dd_pct': round(port_dd_pct, 1)},
        'yearly': yearly, '_daily': total_daily}


# ═══════════════════════════════════════════════════════════════
# Phase 2: Worst-Case Drawdown
# ═══════════════════════════════════════════════════════════════

def phase2_drawdown(p1):
    print(f"\n{'='*120}")
    print(f"  PHASE 2: Worst-Case Drawdown Analysis")
    print(f"{'='*120}")

    daily = p1['_daily']
    eq = daily.cumsum()
    peak = np.maximum.accumulate(eq)
    underwater = eq - peak

    # Top 10 drawdowns
    dd_periods = []
    in_dd = False; dd_start = None; dd_low = 0; dd_low_date = None
    for i, (date, uw) in enumerate(underwater.items()):
        if uw < 0:
            if not in_dd:
                in_dd = True; dd_start = date; dd_low = uw; dd_low_date = date
            if uw < dd_low:
                dd_low = uw; dd_low_date = date
        else:
            if in_dd:
                dd_periods.append({
                    'start': str(dd_start), 'trough': str(dd_low_date),
                    'recovery': str(date), 'depth': round(float(dd_low), 2),
                    'duration_days': (date - dd_start).days,
                })
                in_dd = False
    if in_dd:
        dd_periods.append({
            'start': str(dd_start), 'trough': str(dd_low_date),
            'recovery': 'ONGOING', 'depth': round(float(dd_low), 2),
            'duration_days': (daily.index[-1] - dd_start).days,
        })

    dd_periods.sort(key=lambda x: x['depth'])
    top10 = dd_periods[:10]

    print(f"\n  Top 10 Drawdowns:")
    print(f"  {'#':>3} {'Depth':>10} {'Start':>12} {'Trough':>12} {'Recovery':>12} {'Days':>6}")
    for i, dd in enumerate(top10):
        print(f"  {i+1:>3} ${dd['depth']:>9,.0f} {dd['start'][:10]:>12} {dd['trough'][:10]:>12} "
              f"{dd['recovery'][:12]:>12} {dd['duration_days']:>6}")

    # Consecutive losses
    losing_streaks = []; current_streak = 0; max_streak = 0; streak_pnl = 0; max_streak_pnl = 0
    for pnl in daily.values:
        if pnl < 0:
            current_streak += 1; streak_pnl += pnl
            if current_streak > max_streak:
                max_streak = current_streak; max_streak_pnl = streak_pnl
        else:
            if current_streak > 0:
                losing_streaks.append(current_streak)
            current_streak = 0; streak_pnl = 0

    print(f"\n  Max consecutive losing days: {max_streak} (total loss: ${max_streak_pnl:,.0f})")
    print(f"  Avg losing streak: {np.mean(losing_streaks):.1f} days" if losing_streaks else "  No losing streaks")
    print(f"  Max single-day loss: ${daily.min():,.0f}")
    print(f"  Max single-day gain: ${daily.max():,.0f}")

    return {'top10_dd': top10, 'max_streak': max_streak,
            'max_streak_pnl': round(float(max_streak_pnl), 0),
            'max_daily_loss': round(float(daily.min()), 0),
            'max_daily_gain': round(float(daily.max()), 0)}


# ═══════════════════════════════════════════════════════════════
# Phase 3: High ATR Stress Test
# ═══════════════════════════════════════════════════════════════

def phase3_atr_stress(h1_df, all_trades):
    print(f"\n{'='*120}")
    print(f"  PHASE 3: High ATR / High Gold Price Stress Test")
    print(f"{'='*120}")

    atr_series = compute_atr(h1_df).dropna()
    q25, q50, q75, q90 = np.percentile(atr_series, [25, 50, 75, 90])

    atr_bins = {
        'low (Q0-25)':     (0, q25),
        'normal (Q25-75)': (q25, q75),
        'high (Q75-90)':   (q75, q90),
        'extreme (Q90+)':  (q90, 9999),
    }

    print(f"\n  ATR quartiles: Q25=${q25:.2f}, Q50=${q50:.2f}, Q75=${q75:.2f}, Q90=${q90:.2f}")

    print(f"\n  Cap adequacy in extreme ATR regime:")
    for name in STRAT_ORDER:
        cfg = LIVE_CONFIG[name]
        cap_atr = cfg['cap'] / (cfg['lot'] * PV * q90) if cfg['cap'] > 0 else float('inf')
        status = "OK" if cap_atr > 1.0 else ("TIGHT" if cap_atr > 0.5 else "DANGER")
        print(f"    {name:<15} Cap=${cfg['cap']:>3} @ lot={cfg['lot']:.2f} → {cap_atr:.2f}x ATR at Q90 [{status}]")

    print(f"\n  Per-regime portfolio Sharpe:")
    print(f"  {'Regime':<20} {'N':>6} {'Sharpe':>7} {'PnL':>10} {'WR%':>6} {'MaxDD':>8}")

    regime_results = {}
    for regime_name, (lo_atr, hi_atr) in atr_bins.items():
        atr_at_bar = compute_atr(h1_df).reindex(h1_df.index)
        regime_trades = []
        for name in STRAT_ORDER:
            for t in all_trades[name]:
                entry_time = pd.Timestamp(t['entry_time'])
                if entry_time in atr_at_bar.index:
                    a = atr_at_bar.loc[entry_time]
                    if not np.isnan(a) and lo_atr <= a < hi_atr:
                        regime_trades.append(t)
                else:
                    idx = atr_at_bar.index.searchsorted(entry_time)
                    if 0 < idx < len(atr_at_bar):
                        a = atr_at_bar.iloc[idx-1]
                        if not np.isnan(a) and lo_atr <= a < hi_atr:
                            regime_trades.append(t)

        s = _stats(regime_trades)
        pnl_s = f"${s['pnl']:>9,.0f}" if s['pnl'] >= 0 else f"-${abs(s['pnl']):>8,.0f}"
        print(f"  {regime_name:<20} {s['n']:>6} {s['sharpe']:>7.3f} {pnl_s} {s['wr']:>5.1f}% ${s['max_dd']:>7,.0f}")
        regime_results[regime_name] = s

    return regime_results


# ═══════════════════════════════════════════════════════════════
# Phase 4: Correlation & Simultaneous Loss
# ═══════════════════════════════════════════════════════════════

def phase4_correlation(all_trades):
    print(f"\n{'='*120}")
    print(f"  PHASE 4: Strategy Correlation & Simultaneous Loss Analysis")
    print(f"{'='*120}")

    strat_daily = {}
    for name in STRAT_ORDER:
        strat_daily[name] = _trades_to_daily(all_trades[name])

    all_dates = sorted(set().union(*[set(d.index) for d in strat_daily.values()]))
    df_daily = pd.DataFrame(index=all_dates)
    for name in STRAT_ORDER:
        df_daily[name] = strat_daily[name].reindex(all_dates).fillna(0)

    # Full correlation
    corr = df_daily.corr()
    print(f"\n  Full-period daily PnL correlation:")
    print(f"  {'':>15}", end='')
    for n in STRAT_ORDER: print(f" {n[:8]:>8}", end='')
    print()
    for n1 in STRAT_ORDER:
        print(f"  {n1:<15}", end='')
        for n2 in STRAT_ORDER:
            print(f" {corr.loc[n1, n2]:>8.3f}", end='')
        print()

    # Loss-day correlation
    portfolio_daily = df_daily.sum(axis=1)
    loss_days = portfolio_daily < 0
    if loss_days.sum() > 10:
        corr_loss = df_daily[loss_days].corr()
        corr_normal = df_daily[~loss_days].corr()
        avg_corr_loss = corr_loss.values[np.triu_indices_from(corr_loss.values, k=1)].mean()
        avg_corr_normal = corr_normal.values[np.triu_indices_from(corr_normal.values, k=1)].mean()
        print(f"\n  Avg pairwise correlation — Normal days: {avg_corr_normal:.3f}, Loss days: {avg_corr_loss:.3f}")
        print(f"  {'GOOD: lower on loss days' if avg_corr_loss < avg_corr_normal else 'WARNING: higher on loss days'}")
    else:
        avg_corr_loss = 0; avg_corr_normal = 0

    # Simultaneous loss
    n_strats_losing = (df_daily < 0).sum(axis=1)
    print(f"\n  Simultaneous loss distribution:")
    for k in range(7):
        count = (n_strats_losing == k).sum()
        pct = count / len(n_strats_losing) * 100
        if count > 0:
            print(f"    {k} strategies losing: {count} days ({pct:.1f}%)")

    max_concurrent = df_daily[df_daily < 0].sum(axis=1).min()
    print(f"\n  Max concurrent single-day loss (all losing strats): ${max_concurrent:,.0f}")

    return {
        'avg_corr_normal': round(avg_corr_normal, 3),
        'avg_corr_loss': round(avg_corr_loss, 3),
        'max_concurrent_loss': round(float(max_concurrent), 0),
    }


# ═══════════════════════════════════════════════════════════════
# Phase 5: Monte Carlo Stress Test
# ═══════════════════════════════════════════════════════════════

def phase5_monte_carlo(h1_df):
    print(f"\n{'='*120}")
    print(f"  PHASE 5: Monte Carlo Stress Test (Real Lots, 300 sims)")
    print(f"{'='*120}")

    N_SIMS = 300
    rng = np.random.RandomState(42)

    # 5A: Parameter perturbation
    print(f"\n  5A: Parameter Perturbation (±15%)...")
    port_sharpes = []
    port_dds = []
    for sim in range(N_SIMS):
        total_daily = pd.Series(dtype=float)
        for name in STRAT_ORDER:
            cfg = LIVE_CONFIG[name]
            p_sl = cfg['sl'] * rng.uniform(0.85, 1.15)
            p_tp = cfg['tp'] * rng.uniform(0.85, 1.15)
            p_ta = cfg['trail_act'] * rng.uniform(0.85, 1.15)
            p_td = cfg['trail_dist'] * rng.uniform(0.85, 1.15)
            trades = STRAT_BT[name](h1_df, cfg['lot'], cfg['cap'], p_sl, p_tp, p_ta, p_td, cfg['max_hold'])
            d = _trades_to_daily(trades)
            total_daily = total_daily.add(d, fill_value=0)
        total_daily = total_daily.sort_index()
        port_sharpes.append(_sharpe(total_daily))
        eq = total_daily.cumsum()
        port_dds.append(float((np.maximum.accumulate(eq) - eq).max()) if len(eq) > 1 else 0)
        if (sim+1) % 100 == 0:
            print(f"    {sim+1}/{N_SIMS} done...", flush=True)

    port_sharpes = np.array(port_sharpes)
    port_dds = np.array(port_dds)
    print(f"\n  Sharpe: 5th={np.percentile(port_sharpes, 5):.3f}, "
          f"median={np.median(port_sharpes):.3f}, 95th={np.percentile(port_sharpes, 95):.3f}")
    print(f"  MaxDD: 5th=${np.percentile(port_dds, 5):,.0f}, "
          f"median=${np.median(port_dds):,.0f}, 95th=${np.percentile(port_dds, 95):,.0f}")

    # 5B: Strategy dropout
    print(f"\n  5B: Strategy Dropout (what if 1-2 strategies fail)...")
    baseline_trades = {}
    for name in STRAT_ORDER:
        cfg = LIVE_CONFIG[name]
        baseline_trades[name] = STRAT_BT[name](h1_df, cfg['lot'], cfg['cap'], cfg['sl'], cfg['tp'],
                                                cfg['trail_act'], cfg['trail_dist'], cfg['max_hold'])

    print(f"  {'Dropped':<30} {'Sharpe':>7} {'PnL':>10} {'MaxDD':>8}")
    dropout_results = []
    for drop in range(1, 3):
        for combo in combinations(STRAT_ORDER, drop):
            remaining = [n for n in STRAT_ORDER if n not in combo]
            total_daily = pd.Series(dtype=float)
            for name in remaining:
                d = _trades_to_daily(baseline_trades[name])
                total_daily = total_daily.add(d, fill_value=0)
            total_daily = total_daily.sort_index()
            sh = _sharpe(total_daily)
            pnl = float(total_daily.sum())
            eq = total_daily.cumsum()
            dd = float((np.maximum.accumulate(eq) - eq).max()) if len(eq) > 1 else 0
            label = "- " + ", ".join(combo)
            print(f"  {label:<30} {sh:>7.3f} ${pnl:>9,.0f} ${dd:>7,.0f}")
            dropout_results.append({'dropped': list(combo), 'sharpe': round(sh, 3),
                                    'pnl': round(pnl, 0), 'max_dd': round(dd, 0)})

    return {
        'param_mc': {
            'sharpe_5th': round(float(np.percentile(port_sharpes, 5)), 3),
            'sharpe_median': round(float(np.median(port_sharpes)), 3),
            'dd_median': round(float(np.median(port_dds)), 0),
            'dd_95th': round(float(np.percentile(port_dds, 95)), 0),
        },
        'dropout': dropout_results,
    }


# ═══════════════════════════════════════════════════════════════
# Phase 6: Alpha Decay Detection
# ═══════════════════════════════════════════════════════════════

def phase6_alpha_decay(all_trades):
    print(f"\n{'='*120}")
    print(f"  PHASE 6: Alpha Decay Detection")
    print(f"{'='*120}")

    results = {}
    for name in STRAT_ORDER:
        trades = all_trades[name]
        if len(trades) < 50: continue
        daily = _trades_to_daily(trades)
        if len(daily) < 200: continue

        rolling_sh = daily.rolling(180, min_periods=90).apply(
            lambda x: float(x.mean() / x.std() * np.sqrt(252)) if x.std() > 0 else 0, raw=True)

        recent_90 = daily.iloc[-90:] if len(daily) >= 90 else daily
        historical = daily.iloc[:-90] if len(daily) >= 90 else daily.iloc[:0]

        recent_sh = _sharpe(recent_90) if len(recent_90) >= 10 else 0
        hist_sh = _sharpe(historical) if len(historical) >= 10 else 0
        ratio = recent_sh / hist_sh if hist_sh > 0 else 0

        last_rolling = rolling_sh.dropna()
        if len(last_rolling) >= 50:
            trend = np.polyfit(range(len(last_rolling[-250:])), last_rolling.values[-250:], 1)[0]
        else:
            trend = 0

        status = "GREEN" if ratio > 0.7 and trend > -0.005 else ("YELLOW" if ratio > 0.4 else "RED")
        print(f"  {name:<15} Recent90={recent_sh:.2f} vs Hist={hist_sh:.2f} "
              f"(ratio={ratio:.2f}) trend={trend:.4f}/day [{status}]")
        results[name] = {'recent_sharpe': round(recent_sh, 2), 'hist_sharpe': round(hist_sh, 2),
                         'ratio': round(ratio, 2), 'trend': round(trend, 5), 'status': status}

    return results


# ═══════════════════════════════════════════════════════════════
# Phase 7: Live Gap (optional)
# ═══════════════════════════════════════════════════════════════

def phase7_live_gap():
    print(f"\n{'='*120}")
    print(f"  PHASE 7: Live vs Backtest Gap Quantification")
    print(f"{'='*120}")

    log_paths = [
        Path("_live/data/gold_trade_log.json"),
        Path("../gold-quant-trading/data/gold_trade_log.json"),
    ]
    for p in log_paths:
        if p.exists():
            print(f"  Found trade log: {p}")
            with open(p) as f:
                log_data = json.load(f)
            n_trades = len(log_data) if isinstance(log_data, list) else len(log_data.get('trades', []))
            print(f"  {n_trades} trades in log")
            return {'status': 'found', 'path': str(p), 'n_trades': n_trades}

    print(f"  SKIP: No trade log found at expected paths")
    print(f"  To enable: copy gold_trade_log.json to _live/data/")
    return {'status': 'skipped'}


# ═══════════════════════════════════════════════════════════════
# Phase 8: Fragility Scorecard
# ═══════════════════════════════════════════════════════════════

def phase8_scorecard(p1, p2, p3, p4, p5, p6):
    print(f"\n{'='*120}")
    print(f"  PHASE 8: Fragility Scorecard")
    print(f"{'='*120}")

    port = p1['portfolio']
    checks = []

    # 1. Portfolio Sharpe
    sh = port['sharpe']
    status = "GREEN" if sh > 4.0 else ("YELLOW" if sh > 3.0 else "RED")
    checks.append(('Portfolio Sharpe', f'{sh:.3f}', '> 3.0', status))

    # 2. MaxDD / Capital
    dd_pct = port['max_dd_pct']
    status = "GREEN" if dd_pct < 20 else ("YELLOW" if dd_pct < 30 else "RED")
    checks.append(('Max DD / Capital', f'{dd_pct:.1f}%', '< 30%', status))

    # 3. Worst single-day loss
    worst_day = abs(p2['max_daily_loss'])
    worst_pct = worst_day / CAPITAL * 100
    status = "GREEN" if worst_pct < 5 else ("YELLOW" if worst_pct < 10 else "RED")
    checks.append(('Worst Day / Capital', f'{worst_pct:.1f}%', '< 5%', status))

    # 4. Max consecutive losses
    streak = p2['max_streak']
    status = "GREEN" if streak <= 5 else ("YELLOW" if streak <= 10 else "RED")
    checks.append(('Max Losing Streak', f'{streak} days', '<= 5', status))

    # 5. Extreme ATR Sharpe
    extreme_key = [k for k in p3 if 'extreme' in k]
    if extreme_key:
        ext_sh = p3[extreme_key[0]]['sharpe']
        ratio = ext_sh / sh if sh > 0 else 0
        status = "GREEN" if ratio > 0.7 else ("YELLOW" if ratio > 0.4 else "RED")
        checks.append(('Extreme ATR Sharpe Ratio', f'{ratio:.2f}', '> 0.7', status))

    # 6. Loss-day correlation
    corr_l = p4['avg_corr_loss']; corr_n = p4['avg_corr_normal']
    status = "GREEN" if corr_l < corr_n else "YELLOW"
    checks.append(('Loss-Day Correlation', f'{corr_l:.3f} vs {corr_n:.3f}', 'loss < normal', status))

    # 7. MC 5th percentile Sharpe
    mc_5th = p5['param_mc']['sharpe_5th']
    status = "GREEN" if mc_5th > 2.5 else ("YELLOW" if mc_5th > 1.5 else "RED")
    checks.append(('MC 5th Pctl Sharpe', f'{mc_5th:.3f}', '> 2.5', status))

    # 8. Alpha decay
    decay_statuses = [v['status'] for v in p6.values()]
    n_red = decay_statuses.count('RED')
    n_yellow = decay_statuses.count('YELLOW')
    status = "RED" if n_red >= 2 else ("YELLOW" if n_red >= 1 or n_yellow >= 3 else "GREEN")
    checks.append(('Alpha Decay', f'{n_red} RED, {n_yellow} YELLOW', '0 RED', status))

    # 9. Yearly consistency
    yearly = p1['yearly']
    losing_years = sum(1 for v in yearly.values() if v['pnl'] < 0)
    status = "GREEN" if losing_years == 0 else ("YELLOW" if losing_years <= 1 else "RED")
    checks.append(('Losing Years', f'{losing_years}/{len(yearly)}', '0', status))

    print(f"\n  {'Dimension':<30} {'Value':>25} {'Threshold':>15} {'Status':>8}")
    print(f"  {'-'*30} {'-'*25} {'-'*15} {'-'*8}")
    for dim, val, thresh, stat in checks:
        marker = {'GREEN': '[OK]', 'YELLOW': '[!!]', 'RED': '[XX]'}.get(stat, '?')
        print(f"  {dim:<30} {val:>25} {thresh:>15} {marker:>8}")

    n_green = sum(1 for _, _, _, s in checks if s == 'GREEN')
    n_yellow = sum(1 for _, _, _, s in checks if s == 'YELLOW')
    n_red = sum(1 for _, _, _, s in checks if s == 'RED')
    total = len(checks)

    print(f"\n  OVERALL: {n_green} GREEN, {n_yellow} YELLOW, {n_red} RED / {total} checks")
    if n_red == 0 and n_yellow <= 2:
        verdict = "ROBUST — portfolio is healthy"
    elif n_red <= 1 and n_yellow <= 3:
        verdict = "ADEQUATE — minor concerns to monitor"
    elif n_red <= 2:
        verdict = "FRAGILE — actionable issues identified"
    else:
        verdict = "CRITICAL — immediate attention needed"
    print(f"  VERDICT: {verdict}")

    return {'checks': [{'dim': d, 'val': v, 'thresh': t, 'status': s} for d, v, t, s in checks],
            'summary': {'green': n_green, 'yellow': n_yellow, 'red': n_red},
            'verdict': verdict}


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 120, flush=True)
    print("  R187 — Live Portfolio Stress Test & Fragility Audit", flush=True)
    print("  Live config synced 2026-05-09 | Capital: $5,000", flush=True)
    print("=" * 120, flush=True)

    h1_df = load_h1()
    mean_atr = compute_atr(h1_df).dropna().mean()
    print(f"  H1 mean ATR: ${mean_atr:.2f}")

    print(f"\n  Running all 6 strategies with live lots/caps...", flush=True)
    all_trades = run_all_strategies(h1_df)
    for name in STRAT_ORDER:
        print(f"    {name}: {len(all_trades[name])} trades", flush=True)

    p1 = phase1_baseline(h1_df, all_trades)
    p2 = phase2_drawdown(p1)
    p3 = phase3_atr_stress(h1_df, all_trades)
    p4 = phase4_correlation(all_trades)
    p5 = phase5_monte_carlo(h1_df)
    p6 = phase6_alpha_decay(all_trades)
    p7 = phase7_live_gap()
    p8 = phase8_scorecard(p1, p2, p3, p4, p5, p6)

    elapsed = time.time() - t0
    print(f"\n  Runtime: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    save = {
        'phase1': {k: v for k, v in p1.items() if k != '_daily'},
        'phase2': p2, 'phase3': p3, 'phase4': p4,
        'phase5': p5, 'phase6': p6, 'phase7': p7, 'phase8': p8,
        'runtime_s': round(elapsed, 1),
    }
    out_path = OUTPUT_DIR / "r187_results.json"
    with open(out_path, 'w') as f:
        json.dump(save, f, indent=2, default=str)
    print(f"  Saved: {out_path}")
    print(f"{'='*120}")


if __name__ == "__main__":
    main()

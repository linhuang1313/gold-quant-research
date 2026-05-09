#!/usr/bin/env python3
"""
R94 — Exit Logic Optimization
================================
Tests three exit improvements:
1. Dynamic TP by ATR percentile (regime-adaptive)
2. PSAR MaxLoss Cap sweep ($5 to $20)
3. Time-decay TP for H1 strategies

Structure: Run each variant at unit lot, K-Fold 6 folds on best combos.
"""
import sys, os, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("results/r94_exit_optimization")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100
SPREAD = 0.30
UNIT_LOT = 0.01
CAPITAL = 5000

FOLDS = [
    ("Fold1", "2015-01-01", "2017-01-01"),
    ("Fold2", "2017-01-01", "2019-01-01"),
    ("Fold3", "2019-01-01", "2021-01-01"),
    ("Fold4", "2021-01-01", "2023-01-01"),
    ("Fold5", "2023-01-01", "2025-01-01"),
    ("Fold6", "2025-01-01", "2026-05-01"),
]


def compute_atr(df, period=14):
    tr = pd.DataFrame({
        'hl': df['High'] - df['Low'],
        'hc': (df['High'] - df['Close'].shift(1)).abs(),
        'lc': (df['Low'] - df['Close'].shift(1)).abs()
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


def _mk(pos, exit_price, exit_time, reason, bar_idx, pnl):
    return {'dir': pos['dir'], 'entry': pos['entry'], 'exit': exit_price,
            'entry_time': pos['time'], 'exit_time': exit_time,
            'pnl': pnl, 'reason': reason, 'bars': bar_idx - pos['bar']}


def _run_exit_with_cap(pos, i, h, lo_v, c, spread, lot, pv, times,
                       sl_atr, tp_atr, trail_act_atr, trail_dist_atr, max_hold, maxloss_cap=0):
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
    if pnl_h >= tp_val: return _mk(pos, c, times[i], "TP", i, tp_val)
    if pnl_l <= -sl_val: return _mk(pos, c, times[i], "SL", i, -sl_val)
    if maxloss_cap > 0 and pnl_c < -maxloss_cap:
        return _mk(pos, c, times[i], "MaxLossCap", i, -maxloss_cap)
    ad = trail_act_atr * pos['atr']; td = trail_dist_atr * pos['atr']
    if pos['dir'] == 'BUY' and h - pos['entry'] >= ad:
        ts_p = h - td
        if lo_v <= ts_p: return _mk(pos, c, times[i], "Trail", i, (ts_p - pos['entry'] - spread) * lot * pv)
    elif pos['dir'] == 'SELL' and pos['entry'] - lo_v >= ad:
        ts_p = lo_v + td
        if h >= ts_p: return _mk(pos, c, times[i], "Trail", i, (pos['entry'] - ts_p - spread) * lot * pv)
    if held >= max_hold: return _mk(pos, c, times[i], "Timeout", i, pnl_c)
    return None


def _run_exit_dynamic_tp(pos, i, h, lo_v, c, spread, lot, pv, times,
                         sl_atr, tp_atr_high, tp_atr_normal, tp_atr_low,
                         trail_act_atr, trail_dist_atr, max_hold, maxloss_cap, atr_pctile):
    if atr_pctile > 0.70:
        tp_atr = tp_atr_high
    elif atr_pctile < 0.30:
        tp_atr = tp_atr_low
    else:
        tp_atr = tp_atr_normal
    return _run_exit_with_cap(pos, i, h, lo_v, c, spread, lot, pv, times,
                              sl_atr, tp_atr, trail_act_atr, trail_dist_atr, max_hold, maxloss_cap)


def _run_exit_time_decay(pos, i, h, lo_v, c, spread, lot, pv, times,
                         sl_atr, tp_atr, trail_act_atr, trail_dist_atr, max_hold, maxloss_cap,
                         decay_start=3, atr_start=0.30, atr_step=0.10):
    held = i - pos['bar']
    if pos['dir'] == 'BUY':
        pnl_c = (c - pos['entry'] - spread) * lot * pv
    else:
        pnl_c = (pos['entry'] - c - spread) * lot * pv

    if held >= decay_start and pnl_c > 0:
        decay_bars = held - decay_start
        min_profit_atr = max(0.0, atr_start - decay_bars * atr_step)
        min_profit = min_profit_atr * pos['atr'] * lot * pv
        if pnl_c >= min_profit and min_profit_atr < tp_atr * 0.5:
            return _mk(pos, c, times[i], "TimeDecayTP", i, pnl_c)

    return _run_exit_with_cap(pos, i, h, lo_v, c, spread, lot, pv, times,
                              sl_atr, tp_atr, trail_act_atr, trail_dist_atr, max_hold, maxloss_cap)


# ═══════════════════════════════════════════════════════════════
# PSAR backtest variants
# ═══════════════════════════════════════════════════════════════

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


def bt_psar_dynamic_tp(h1_df, spread, lot, maxloss_cap=0,
                       sl_atr=4.5, tp_high=20.0, tp_normal=16.0, tp_low=12.0,
                       trail_act=0.20, trail_dist=0.04, max_hold=20):
    df = add_psar(h1_df).dropna(subset=['PSAR_dir', 'ATR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    pdir = df['PSAR_dir'].values; atr = df['ATR'].values
    times = df.index; n = len(df)
    atr_roll_252 = pd.Series(atr).rolling(252, min_periods=50).rank(pct=True).values
    trades = []; pos = None; last_exit = -999; pos_atr_pctile = 0.5
    for i in range(1, n):
        if pos is not None:
            result = _run_exit_dynamic_tp(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                                          sl_atr, tp_high, tp_normal, tp_low,
                                          trail_act, trail_dist, max_hold, maxloss_cap, pos_atr_pctile)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if i-last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        if pdir[i-1] == -1 and pdir[i] == 1:
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
            pos_atr_pctile = atr_roll_252[i] if not np.isnan(atr_roll_252[i]) else 0.5
        elif pdir[i-1] == 1 and pdir[i] == -1:
            pos = {'dir': 'SELL', 'entry': c[i]-spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
            pos_atr_pctile = atr_roll_252[i] if not np.isnan(atr_roll_252[i]) else 0.5
    return trades


def bt_psar_time_decay(h1_df, spread, lot, maxloss_cap=0,
                       sl_atr=4.5, tp_atr=16.0, trail_act=0.20, trail_dist=0.04, max_hold=20,
                       decay_start=3, atr_start=0.30, atr_step=0.10):
    df = add_psar(h1_df).dropna(subset=['PSAR_dir', 'ATR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    pdir = df['PSAR_dir'].values; atr = df['ATR'].values
    times = df.index; n = len(df)
    trades = []; pos = None; last_exit = -999
    for i in range(1, n):
        if pos is not None:
            result = _run_exit_time_decay(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                                          sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap,
                                          decay_start, atr_start, atr_step)
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


# ═══════════════════════════════════════════════════════════════
# TSMOM and SESS_BO with dynamic TP
# ═══════════════════════════════════════════════════════════════

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


def bt_tsmom_dynamic_tp(h1_df, spread, lot, maxloss_cap=0,
                        fast=480, slow=720, sl_atr=4.5,
                        tp_high=8.0, tp_normal=6.0, tp_low=4.0,
                        trail_act=0.14, trail_dist=0.025, max_hold=20):
    df = h1_df.copy(); df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; times = df.index; n = len(df)
    max_lb = max(fast, slow)
    atr_roll_252 = pd.Series(atr).rolling(252, min_periods=50).rank(pct=True).values
    score = np.full(n, np.nan)
    for i in range(max_lb, n):
        s = 0.0
        if c[i-fast] > 0: s += 0.5 * np.sign(c[i]/c[i-fast] - 1.0)
        if c[i-slow] > 0: s += 0.5 * np.sign(c[i]/c[i-slow] - 1.0)
        score[i] = s
    trades = []; pos = None; last_exit = -999; pos_atr_pctile = 0.5
    for i in range(max_lb+1, n):
        if pos is not None:
            result = _run_exit_dynamic_tp(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                                          sl_atr, tp_high, tp_normal, tp_low,
                                          trail_act, trail_dist, max_hold, maxloss_cap, pos_atr_pctile)
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
            pos_atr_pctile = atr_roll_252[i] if not np.isnan(atr_roll_252[i]) else 0.5
        elif score[i] < 0 and score[i-1] >= 0:
            pos = {'dir': 'SELL', 'entry': c[i]-spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
            pos_atr_pctile = atr_roll_252[i] if not np.isnan(atr_roll_252[i]) else 0.5
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


def bt_sess_bo_dynamic_tp(h1_df, spread, lot, maxloss_cap=0,
                          session_hour=12, lookback=4, sl_atr=4.5,
                          tp_high=6.0, tp_normal=4.0, tp_low=3.0,
                          trail_act=0.14, trail_dist=0.025, max_hold=20):
    df = h1_df.copy(); df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; hours = df.index.hour; times = df.index; n = len(df)
    atr_roll_252 = pd.Series(atr).rolling(252, min_periods=50).rank(pct=True).values
    trades = []; pos = None; last_exit = -999; pos_atr_pctile = 0.5
    for i in range(lookback, n):
        if pos is not None:
            result = _run_exit_dynamic_tp(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                                          sl_atr, tp_high, tp_normal, tp_low,
                                          trail_act, trail_dist, max_hold, maxloss_cap, pos_atr_pctile)
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
            pos_atr_pctile = atr_roll_252[i] if not np.isnan(atr_roll_252[i]) else 0.5
        elif c[i] < ll:
            pos = {'dir': 'SELL', 'entry': c[i]-spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
            pos_atr_pctile = atr_roll_252[i] if not np.isnan(atr_roll_252[i]) else 0.5
    return trades


# ═══════════════════════════════════════════════════════════════
# Stats helpers
# ═══════════════════════════════════════════════════════════════

def trades_to_daily_series(trades):
    if not trades: return pd.Series(dtype=float)
    daily = {}
    for t in trades:
        ts = pd.Timestamp(t['exit_time'])
        if ts.tzinfo is not None:
            ts = ts.tz_localize(None)
        d = ts.normalize()
        daily[d] = daily.get(d, 0) + t['pnl']
    dates = sorted(daily.keys())
    return pd.Series([daily[d] for d in dates], index=pd.DatetimeIndex(dates))

def sharpe(daily_pnl):
    if len(daily_pnl) < 10: return 0.0
    arr = np.array(daily_pnl) if not isinstance(daily_pnl, np.ndarray) else daily_pnl
    if arr.std() == 0: return 0.0
    return float(arr.mean() / arr.std() * np.sqrt(252))

def max_dd(daily_pnl):
    arr = np.array(daily_pnl) if not isinstance(daily_pnl, np.ndarray) else daily_pnl
    cum = np.cumsum(arr)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    return float(dd.max()) if len(dd) > 0 else 0.0


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 80, flush=True)
    print("  R94 — Exit Logic Optimization", flush=True)
    print("=" * 80, flush=True)

    from backtest.runner import load_m15, load_h1_aligned, H1_CSV_PATH
    print("\n  Loading data...", flush=True)
    m15_raw = load_m15()
    h1_df = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])
    print(f"  H1: {len(h1_df)} bars", flush=True)

    results = {}

    # ══════════════════════════════════════════════════════════════
    # Phase 1: PSAR Cap Sweep
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 1: PSAR MaxLoss Cap Sweep", flush=True)
    print("=" * 70, flush=True)

    psar_caps = [0, 5, 8, 10, 12, 15, 20]
    psar_results = {}
    for cap in psar_caps:
        trades = bt_psar(h1_df, spread=SPREAD, lot=UNIT_LOT, maxloss_cap=cap)
        ds = trades_to_daily_series(trades)
        sh = sharpe(ds)
        dd = max_dd(ds)
        pnl = sum(t['pnl'] for t in trades)
        wr = sum(1 for t in trades if t['pnl'] > 0) / len(trades) * 100 if trades else 0
        psar_results[f"cap_{cap}"] = {
            'cap': cap, 'n_trades': len(trades), 'sharpe': round(sh, 3),
            'max_dd': round(dd, 2), 'pnl': round(pnl, 2), 'wr': round(wr, 1),
        }
        print(f"    Cap=${cap:>2}: {len(trades)} trades, Sharpe={sh:.3f}, MaxDD=${dd:.1f}, WR={wr:.1f}%", flush=True)

    results['psar_cap_sweep'] = psar_results

    # ══════════════════════════════════════════════════════════════
    # Phase 2: Dynamic TP (all strategies)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 2: Dynamic TP by ATR Regime", flush=True)
    print("=" * 70, flush=True)

    dynamic_tp_results = {}

    psar_base = bt_psar(h1_df, SPREAD, UNIT_LOT, maxloss_cap=5)
    psar_dyn = bt_psar_dynamic_tp(h1_df, SPREAD, UNIT_LOT, maxloss_cap=5,
                                   tp_high=20.0, tp_normal=16.0, tp_low=12.0)
    ds_base = trades_to_daily_series(psar_base)
    ds_dyn = trades_to_daily_series(psar_dyn)
    dynamic_tp_results['PSAR'] = {
        'baseline': {'sharpe': round(sharpe(ds_base), 3), 'n': len(psar_base), 'max_dd': round(max_dd(ds_base), 2)},
        'dynamic_tp': {'sharpe': round(sharpe(ds_dyn), 3), 'n': len(psar_dyn), 'max_dd': round(max_dd(ds_dyn), 2)},
    }
    print(f"    PSAR baseline:   Sharpe={sharpe(ds_base):.3f} ({len(psar_base)} trades)", flush=True)
    print(f"    PSAR dynamic_tp: Sharpe={sharpe(ds_dyn):.3f} ({len(psar_dyn)} trades)", flush=True)

    tsmom_base = bt_tsmom(h1_df, SPREAD, UNIT_LOT, maxloss_cap=0)
    tsmom_dyn = bt_tsmom_dynamic_tp(h1_df, SPREAD, UNIT_LOT, maxloss_cap=0,
                                     tp_high=8.0, tp_normal=6.0, tp_low=4.0)
    ds_base = trades_to_daily_series(tsmom_base)
    ds_dyn = trades_to_daily_series(tsmom_dyn)
    dynamic_tp_results['TSMOM'] = {
        'baseline': {'sharpe': round(sharpe(ds_base), 3), 'n': len(tsmom_base), 'max_dd': round(max_dd(ds_base), 2)},
        'dynamic_tp': {'sharpe': round(sharpe(ds_dyn), 3), 'n': len(tsmom_dyn), 'max_dd': round(max_dd(ds_dyn), 2)},
    }
    print(f"    TSMOM baseline:   Sharpe={sharpe(ds_base):.3f} ({len(tsmom_base)} trades)", flush=True)
    print(f"    TSMOM dynamic_tp: Sharpe={sharpe(ds_dyn):.3f} ({len(tsmom_dyn)} trades)", flush=True)

    sess_base = bt_sess_bo(h1_df, SPREAD, UNIT_LOT, maxloss_cap=35)
    sess_dyn = bt_sess_bo_dynamic_tp(h1_df, SPREAD, UNIT_LOT, maxloss_cap=35,
                                      tp_high=6.0, tp_normal=4.0, tp_low=3.0)
    ds_base = trades_to_daily_series(sess_base)
    ds_dyn = trades_to_daily_series(sess_dyn)
    dynamic_tp_results['SESS_BO'] = {
        'baseline': {'sharpe': round(sharpe(ds_base), 3), 'n': len(sess_base), 'max_dd': round(max_dd(ds_base), 2)},
        'dynamic_tp': {'sharpe': round(sharpe(ds_dyn), 3), 'n': len(sess_dyn), 'max_dd': round(max_dd(ds_dyn), 2)},
    }
    print(f"    SESS_BO baseline:   Sharpe={sharpe(ds_base):.3f} ({len(sess_base)} trades)", flush=True)
    print(f"    SESS_BO dynamic_tp: Sharpe={sharpe(ds_dyn):.3f} ({len(sess_dyn)} trades)", flush=True)

    results['dynamic_tp'] = dynamic_tp_results

    # ══════════════════════════════════════════════════════════════
    # Phase 3: Time-Decay TP
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 3: Time-Decay TP", flush=True)
    print("=" * 70, flush=True)

    time_decay_results = {}
    decay_configs = [
        {'decay_start': 3, 'atr_start': 0.30, 'atr_step': 0.10},
        {'decay_start': 5, 'atr_start': 0.50, 'atr_step': 0.08},
        {'decay_start': 8, 'atr_start': 0.40, 'atr_step': 0.05},
    ]

    for cfg in decay_configs:
        label = f"start{cfg['decay_start']}_atr{cfg['atr_start']}_step{cfg['atr_step']}"
        psar_td = bt_psar_time_decay(h1_df, SPREAD, UNIT_LOT, maxloss_cap=5, **cfg)
        ds = trades_to_daily_series(psar_td)
        sh = sharpe(ds)
        time_decay_results[f"PSAR_{label}"] = {
            'sharpe': round(sh, 3), 'n_trades': len(psar_td), 'config': cfg,
        }
        print(f"    PSAR {label}: Sharpe={sh:.3f} ({len(psar_td)} trades)", flush=True)

    results['time_decay_tp'] = time_decay_results

    # ══════════════════════════════════════════════════════════════
    # Phase 4: K-Fold Validation on best PSAR Cap
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 4: K-Fold Validation", flush=True)
    print("=" * 70, flush=True)

    best_cap = max(psar_results.values(), key=lambda x: x['sharpe'])['cap']
    print(f"    Best PSAR Cap: ${best_cap} (Sharpe={psar_results[f'cap_{best_cap}']['sharpe']:.3f})", flush=True)

    kfold_results = {}
    for cap in [5, best_cap, 0]:
        if cap == best_cap and cap == 5:
            continue
        fold_sharpes = []
        for fold_name, start, end in FOLDS:
            h1_fold = h1_df[(h1_df.index >= start) & (h1_df.index < end)]
            if len(h1_fold) < 100: continue
            trades = bt_psar(h1_fold, SPREAD, UNIT_LOT, maxloss_cap=cap)
            ds = trades_to_daily_series(trades)
            fold_sharpes.append(sharpe(ds))
        positive = sum(1 for s in fold_sharpes if s > 0)
        kfold_results[f"PSAR_cap{cap}"] = {
            'cap': cap, 'fold_sharpes': [round(s, 2) for s in fold_sharpes],
            'positive_folds': positive, 'mean_sharpe': round(float(np.mean(fold_sharpes)), 3),
            'pass': positive >= 4,
        }
        print(f"    Cap=${cap}: {positive}/6 folds positive, mean Sharpe={np.mean(fold_sharpes):.3f}", flush=True)

    results['kfold_psar_cap'] = kfold_results

    # ══════════════════════════════════════════════════════════════
    # Save results
    # ══════════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    results['elapsed_s'] = round(elapsed, 1)
    print(f"\n{'='*80}", flush=True)
    print(f"  R94 COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)
    print(f"{'='*80}", flush=True)

    with open(OUTPUT_DIR / "r94_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved: {OUTPUT_DIR}/r94_results.json", flush=True)


if __name__ == "__main__":
    main()

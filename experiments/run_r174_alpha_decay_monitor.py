#!/usr/bin/env python3
"""
R174 — Adaptive Alpha Decay Monitor
=====================================
Automated system to detect strategy degradation before it causes real losses.
Can be run weekly/monthly as a health check on the P6 portfolio.

Phases:
  1. Current State Snapshot (per-strategy and portfolio metrics)
  2. Rolling K-Fold Health Check (compare with historical baselines)
  3. Parameter Stability Test (L8_MAX parameter drift detection)
  4. Recent Performance Window Analysis (30/60/90/180 day windows)
  5. Strategy Correlation Drift (diversification monitoring)
  6. Summary Report (JSON + TXT, with --compare diff support)

Usage:
  python experiments/run_r174_alpha_decay_monitor.py
  python experiments/run_r174_alpha_decay_monitor.py --quick
  python experiments/run_r174_alpha_decay_monitor.py --compare results/r174_alpha_decay_monitor/snapshots/2026-05-01.json
"""
import sys, os, io, time, json, glob, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from copy import deepcopy

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

OUTPUT_DIR = Path("results/r174_alpha_decay_monitor")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SNAPSHOT_DIR = OUTPUT_DIR / "snapshots"
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100
SPREAD = 0.30

P6_LOTS = {
    'L8_MAX': 0.01, 'PSAR': 0.03, 'TSMOM': 0.04,
    'SESS_BO': 0.04, 'DUAL_THRUST': 0.04, 'CHANDELIER': 0.08,
}

STRAT_ORDER = ['L8_MAX', 'PSAR', 'TSMOM', 'SESS_BO', 'DUAL_THRUST', 'CHANDELIER']

REFERENCE_BASELINES = {
    'L8_MAX':      {'sharpe': 11.23, 'kfold_mean': 10.29, 'kfold_min': 9.78},
    'PSAR':        {'sharpe': 4.13, 'kfold_mean': 4.36, 'kfold_min': 3.43},
    'TSMOM':       {'sharpe': 5.40, 'kfold_mean': 6.02, 'kfold_min': 4.27},
    'SESS_BO':     {'sharpe': 4.50, 'kfold_mean': 5.50, 'kfold_min': 4.00},
    'DUAL_THRUST': {'sharpe': 4.00, 'kfold_mean': 5.88, 'kfold_min': 4.27},
    'CHANDELIER':  {'sharpe': 4.50, 'kfold_mean': 6.35, 'kfold_min': 6.02},
}

L8_PARAM_DEFAULTS = {
    'sl_atr_mult': 3.5,
    'tp_atr_mult': 8.0,
    'trailing_activate_atr': 0.14,
    'trailing_distance_atr': 0.025,
    'keltner_adx_threshold': 14,
    'keltner_max_hold_m15': 20,
}

HEALTH_GREEN = "GREEN"
HEALTH_YELLOW = "YELLOW"
HEALTH_RED = "RED"


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


# ═══════════════════════════════════════════════════════════════
# H1 strategy helpers (from R150)
# ═══════════════════════════════════════════════════════════════

def compute_atr(df, period=14):
    tr = pd.DataFrame({
        'hl': df['High'] - df['Low'],
        'hc': (df['High'] - df['Close'].shift(1)).abs(),
        'lc': (df['Low'] - df['Close'].shift(1)).abs(),
    }).max(axis=1)
    return tr.rolling(period).mean()


def add_psar(df, af_start=0.01, af_max=0.05):
    df = df.copy()
    n = len(df)
    psar = np.zeros(n)
    direction = np.ones(n)
    af = af_start
    ep = df['High'].iloc[0]
    psar[0] = df['Low'].iloc[0]
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
    df['PSAR_dir'] = direction
    df['ATR'] = compute_atr(df)
    return df


def _mk(pos, exit_p, exit_time, reason, bar_idx, pnl):
    return {'dir': pos['dir'], 'entry': pos['entry'], 'exit': exit_p,
            'entry_time': pos['time'], 'exit_time': exit_time,
            'pnl': pnl, 'reason': reason, 'bars': bar_idx - pos['bar']}


def _run_exit(pos, i, h, lo_v, c, spread, lot, pv, times,
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
    ad = trail_act_atr * pos['atr']
    td = trail_dist_atr * pos['atr']
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
            result = _run_exit(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                               sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if i - last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        if pdir[i-1] == -1 and pdir[i] == 1:
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif pdir[i-1] == 1 and pdir[i] == -1:
            pos = {'dir': 'SELL', 'entry': c[i]-spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_tsmom(h1_df, spread, lot, maxloss_cap=0,
             fast=480, slow=720, sl_atr=4.5, tp_atr=6.0,
             trail_act=0.14, trail_dist=0.025, max_hold=20):
    df = h1_df.copy()
    df['ATR'] = compute_atr(df)
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
        if i - last_exit < 2: continue
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
    df = h1_df.copy()
    df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; hours = df.index.hour; times = df.index; n = len(df)
    trades = []; pos = None; last_exit = -999
    for i in range(lookback, n):
        if pos is not None:
            result = _run_exit(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                               sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
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
            result = _run_exit(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
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
    atr = df['ATR'].values
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
            direction[i] = direction[i-1]
            continue
        if c[i] > chandelier_short[i-1]:
            direction[i] = 1
        elif c[i] < chandelier_long[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
    trades = []; pos = None; last_exit = -999
    for i in range(period + 2, n):
        if pos is not None:
            result = _run_exit(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                               sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if i - last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        if direction[i] == 1 and direction[i-1] != 1:
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif direction[i] == -1 and direction[i-1] != -1:
            pos = {'dir': 'SELL', 'entry': c[i]-spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_l8_max(data_bundle, spread, lot):
    from backtest.runner import run_variant, LIVE_PARITY_KWARGS
    kw = {**LIVE_PARITY_KWARGS, 'spread_cost': spread, 'initial_capital': 2000,
           'min_lot_size': lot, 'max_lot_size': lot}
    result = run_variant(data_bundle, "L8_MAX", verbose=False, **kw)
    raw_trades = result.get('_trades', [])
    trades = []
    for t in raw_trades:
        trades.append({
            'dir': t.direction, 'entry': t.entry_price, 'exit': t.exit_price,
            'entry_time': t.entry_time, 'exit_time': t.exit_time,
            'pnl': t.pnl, 'reason': t.exit_reason, 'bars': t.bars_held,
        })
    return trades


# ═══════════════════════════════════════════════════════════════
# Stats & metrics helpers
# ═══════════════════════════════════════════════════════════════

def compute_trade_stats(trades, lot):
    if not trades:
        return {'sharpe': 0, 'total_pnl': 0, 'n_trades': 0, 'win_rate': 0,
                'max_dd': 0, 'profit_factor': 0, 'avg_pnl': 0}
    pnls = np.array([t['pnl'] for t in trades])
    n = len(pnls)
    total = pnls.sum()
    wins = (pnls > 0).sum()
    win_rate = wins / n if n > 0 else 0
    avg = pnls.mean()
    std = pnls.std() if n > 1 else 1.0
    sharpe = (avg / std) * np.sqrt(252) if std > 0 else 0
    cumulative = np.cumsum(pnls)
    peak = np.maximum.accumulate(cumulative)
    dd = peak - cumulative
    max_dd = dd.max() if len(dd) > 0 else 0
    gross_profit = pnls[pnls > 0].sum()
    gross_loss = abs(pnls[pnls < 0].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else 99.9
    return {
        'sharpe': round(float(sharpe), 2),
        'total_pnl': round(float(total), 2),
        'n_trades': int(n),
        'win_rate': round(float(win_rate), 4),
        'max_dd': round(float(max_dd), 2),
        'profit_factor': round(float(pf), 2),
        'avg_pnl': round(float(avg), 2),
    }


def build_daily_equity(trades):
    if not trades:
        return pd.Series(dtype=float)
    records = []
    for t in trades:
        et = t['exit_time']
        if hasattr(et, 'date'):
            d = et.date() if not isinstance(et, str) else pd.Timestamp(et).date()
        else:
            d = pd.Timestamp(et).date()
        records.append({'date': d, 'pnl': t['pnl']})
    df = pd.DataFrame(records)
    daily = df.groupby('date')['pnl'].sum()
    return daily.sort_index()


def compute_portfolio_stats(all_trades_dict):
    combined_daily = None
    for strat, trades in all_trades_dict.items():
        daily = build_daily_equity(trades)
        if daily.empty:
            continue
        if combined_daily is None:
            combined_daily = daily.copy()
        else:
            combined_daily = combined_daily.add(daily, fill_value=0)
    if combined_daily is None or combined_daily.empty:
        return {'sharpe': 0, 'total_pnl': 0, 'max_dd': 0, 'n_days': 0}
    total = combined_daily.sum()
    avg = combined_daily.mean()
    std = combined_daily.std()
    sharpe = (avg / std) * np.sqrt(252) if std > 0 else 0
    cumulative = combined_daily.cumsum()
    peak = cumulative.cummax()
    dd = peak - cumulative
    max_dd = dd.max()
    return {
        'sharpe': round(float(sharpe), 2),
        'total_pnl': round(float(total), 2),
        'max_dd': round(float(max_dd), 2),
        'n_days': int(len(combined_daily)),
    }


# ═══════════════════════════════════════════════════════════════
# Phase 1: Current State Snapshot
# ═══════════════════════════════════════════════════════════════

def phase1_snapshot(h1_df, data_bundle):
    print("\n" + "="*70, flush=True)
    print("PHASE 1: Current State Snapshot", flush=True)
    print("="*70, flush=True)

    all_trades = {}
    strat_metrics = {}

    for strat in STRAT_ORDER:
        lot = P6_LOTS[strat]
        print(f"\n  Running {strat} (lot={lot})...", flush=True)
        t0 = time.time()
        if strat == 'L8_MAX':
            trades = bt_l8_max(data_bundle, SPREAD, lot)
        elif strat == 'PSAR':
            trades = bt_psar(h1_df, SPREAD, lot)
        elif strat == 'TSMOM':
            trades = bt_tsmom(h1_df, SPREAD, lot)
        elif strat == 'SESS_BO':
            trades = bt_sess_bo(h1_df, SPREAD, lot)
        elif strat == 'DUAL_THRUST':
            trades = bt_dual_thrust(h1_df, SPREAD, lot)
        elif strat == 'CHANDELIER':
            trades = bt_chandelier(h1_df, SPREAD, lot)
        else:
            trades = []
        elapsed = time.time() - t0
        stats = compute_trade_stats(trades, lot)
        stats['elapsed_s'] = round(elapsed, 1)
        strat_metrics[strat] = stats
        all_trades[strat] = trades
        print(f"    {stats['n_trades']} trades, Sharpe={stats['sharpe']:.2f}, "
              f"PnL=${stats['total_pnl']:.0f}, WR={stats['win_rate']:.1%}, "
              f"MaxDD=${stats['max_dd']:.0f} [{elapsed:.0f}s]", flush=True)

    portfolio = compute_portfolio_stats(all_trades)
    print(f"\n  Portfolio: Sharpe={portfolio['sharpe']:.2f}, "
          f"PnL=${portfolio['total_pnl']:.0f}, MaxDD=${portfolio['max_dd']:.0f}", flush=True)

    print("\n  Comparison with reference baselines:", flush=True)
    print(f"  {'Strategy':<14} {'Current':>8} {'Baseline':>9} {'Delta%':>8} {'Status':>8}", flush=True)
    print(f"  {'-'*50}", flush=True)
    for strat in STRAT_ORDER:
        cur = strat_metrics[strat]['sharpe']
        ref = REFERENCE_BASELINES[strat]['sharpe']
        delta = ((cur - ref) / ref * 100) if ref != 0 else 0
        status = HEALTH_GREEN if delta >= -10 else (HEALTH_YELLOW if delta >= -25 else HEALTH_RED)
        print(f"  {strat:<14} {cur:>8.2f} {ref:>9.2f} {delta:>+7.1f}% {status:>8}", flush=True)

    return {'strat_metrics': strat_metrics, 'portfolio': portfolio, 'all_trades': all_trades}


# ═══════════════════════════════════════════════════════════════
# Phase 2: Rolling K-Fold Health Check
# ═══════════════════════════════════════════════════════════════

KFOLD_WINDOWS = [
    ("Fold1", "2015-01-01", "2017-01-01"),
    ("Fold2", "2017-01-01", "2019-01-01"),
    ("Fold3", "2019-01-01", "2021-01-01"),
    ("Fold4", "2021-01-01", "2023-01-01"),
    ("Fold5", "2023-01-01", "2025-01-01"),
    ("Fold6", "2025-01-01", "2027-01-01"),
]


def run_h1_kfold(h1_df, strat_name, lot):
    fold_results = []
    for fold_name, start, end in KFOLD_WINDOWS:
        ts = pd.Timestamp(start)
        te = pd.Timestamp(end)
        fold_df = h1_df[(h1_df.index >= ts) & (h1_df.index < te)]
        if len(fold_df) < 200:
            continue
        if strat_name == 'PSAR':
            trades = bt_psar(fold_df, SPREAD, lot)
        elif strat_name == 'TSMOM':
            trades = bt_tsmom(fold_df, SPREAD, lot)
        elif strat_name == 'SESS_BO':
            trades = bt_sess_bo(fold_df, SPREAD, lot)
        elif strat_name == 'DUAL_THRUST':
            trades = bt_dual_thrust(fold_df, SPREAD, lot)
        elif strat_name == 'CHANDELIER':
            trades = bt_chandelier(fold_df, SPREAD, lot)
        else:
            continue
        stats = compute_trade_stats(trades, lot)
        stats['fold'] = fold_name
        fold_results.append(stats)
    return fold_results


def run_l8_kfold(data_bundle, lot):
    from backtest.runner import run_kfold, LIVE_PARITY_KWARGS
    kw = {**LIVE_PARITY_KWARGS, 'spread_cost': SPREAD, 'initial_capital': 2000,
           'min_lot_size': lot, 'max_lot_size': lot, 'verbose': False}
    results = run_kfold(data_bundle, kw, n_folds=6, label_prefix="L8_")
    fold_results = []
    for r in results:
        fold_results.append({
            'fold': r.get('fold', ''),
            'sharpe': round(r.get('sharpe', 0), 2),
            'total_pnl': round(r.get('total_pnl', 0), 2),
            'n_trades': r.get('n', 0),
            'win_rate': round(r.get('win_rate', 0), 4),
        })
    return fold_results


def phase2_kfold(h1_df, data_bundle):
    print("\n" + "="*70, flush=True)
    print("PHASE 2: Rolling K-Fold Health Check", flush=True)
    print("="*70, flush=True)

    kfold_results = {}
    alerts = []

    for strat in STRAT_ORDER:
        lot = P6_LOTS[strat]
        print(f"\n  K-Fold: {strat}...", flush=True)
        t0 = time.time()
        if strat == 'L8_MAX':
            folds = run_l8_kfold(data_bundle, lot)
        else:
            folds = run_h1_kfold(h1_df, strat, lot)
        elapsed = time.time() - t0

        sharpes = [f['sharpe'] for f in folds if f.get('sharpe') is not None]
        if not sharpes:
            print(f"    No valid folds", flush=True)
            kfold_results[strat] = {'folds': folds, 'mean': 0, 'min': 0, 'status': HEALTH_RED}
            continue

        mean_s = np.mean(sharpes)
        min_s = np.min(sharpes)
        ref = REFERENCE_BASELINES[strat]

        status = HEALTH_GREEN
        strat_alerts = []

        neg_folds = [f for f, s in zip(folds, sharpes) if s < 0]
        if neg_folds:
            status = HEALTH_RED
            strat_alerts.append(f"Fold(s) with negative Sharpe: {[f['fold'] for f in neg_folds]}")

        mean_drop = ((mean_s - ref['kfold_mean']) / ref['kfold_mean'] * 100) if ref['kfold_mean'] != 0 else 0
        if mean_drop < -20:
            status = HEALTH_RED if status != HEALTH_RED else status
            strat_alerts.append(f"Mean K-Fold Sharpe dropped {mean_drop:.1f}% from baseline")
        elif mean_drop < -10:
            if status == HEALTH_GREEN:
                status = HEALTH_YELLOW

        min_drop = ((min_s - ref['kfold_min']) / ref['kfold_min'] * 100) if ref['kfold_min'] != 0 else 0
        if min_drop < -30:
            status = HEALTH_RED
            strat_alerts.append(f"Min fold Sharpe dropped {min_drop:.1f}% from baseline")

        kfold_results[strat] = {
            'folds': folds, 'mean': round(float(mean_s), 2),
            'min': round(float(min_s), 2), 'status': status, 'alerts': strat_alerts,
        }
        if strat_alerts:
            alerts.extend([f"[{strat}] {a}" for a in strat_alerts])

        print(f"    Mean={mean_s:.2f} (ref={ref['kfold_mean']:.2f}), "
              f"Min={min_s:.2f} (ref={ref['kfold_min']:.2f}), "
              f"Status={status} [{elapsed:.0f}s]", flush=True)
        for f in folds:
            print(f"      {f['fold']}: Sharpe={f['sharpe']:.2f}, N={f.get('n_trades', 0)}", flush=True)

    return {'kfold_results': kfold_results, 'alerts': alerts}


# ═══════════════════════════════════════════════════════════════
# Phase 3: Parameter Stability Test (L8_MAX)
# ═══════════════════════════════════════════════════════════════

def phase3_param_stability(data_bundle):
    print("\n" + "="*70, flush=True)
    print("PHASE 3: Parameter Stability Test (L8_MAX)", flush=True)
    print("="*70, flush=True)

    from backtest.runner import run_variant, LIVE_PARITY_KWARGS
    lot = P6_LOTS['L8_MAX']
    base_kw = {**LIVE_PARITY_KWARGS, 'spread_cost': SPREAD, 'initial_capital': 2000,
               'min_lot_size': lot, 'max_lot_size': lot}

    print("\n  Running baseline...", flush=True)
    base_result = run_variant(data_bundle, "L8_base", verbose=False, **base_kw)
    base_sharpe = base_result.get('sharpe', 0)
    print(f"    Baseline Sharpe: {base_sharpe:.2f}", flush=True)

    param_results = {}
    drift_alerts = []

    perturbations = [-0.10, +0.10]
    params_to_test = {
        'sl_atr_mult': L8_PARAM_DEFAULTS['sl_atr_mult'],
        'tp_atr_mult': L8_PARAM_DEFAULTS['tp_atr_mult'],
        'trailing_activate_atr': L8_PARAM_DEFAULTS['trailing_activate_atr'],
        'trailing_distance_atr': L8_PARAM_DEFAULTS['trailing_distance_atr'],
        'keltner_adx_threshold': L8_PARAM_DEFAULTS['keltner_adx_threshold'],
        'keltner_max_hold_m15': L8_PARAM_DEFAULTS['keltner_max_hold_m15'],
    }

    print(f"\n  {'Parameter':<26} {'Deployed':>9} {'-10%':>9} {'+10%':>9} {'BestVal':>9} {'Drift?':>7}", flush=True)
    print(f"  {'-'*72}", flush=True)

    for param_name, deployed_val in params_to_test.items():
        results_for_param = {'deployed': deployed_val, 'deployed_sharpe': base_sharpe}
        best_sharpe = base_sharpe
        best_val = deployed_val

        perturb_sharpes = {}
        for pct in perturbations:
            test_val = deployed_val * (1 + pct)
            if param_name in ('keltner_adx_threshold', 'keltner_max_hold_m15'):
                test_val = int(round(test_val))
            test_kw = {**base_kw, param_name: test_val}
            r = run_variant(data_bundle, f"L8_{param_name}_{pct:+.0%}", verbose=False, **test_kw)
            s = r.get('sharpe', 0)
            perturb_sharpes[pct] = {'value': test_val, 'sharpe': s}
            if s > best_sharpe:
                best_sharpe = s
                best_val = test_val

        has_drift = (best_val != deployed_val and
                     (best_sharpe - base_sharpe) / max(abs(base_sharpe), 0.01) > 0.05)
        drift_flag = "YES" if has_drift else "no"
        if has_drift:
            drift_alerts.append(f"{param_name}: optimal shifted to {best_val:.4g} "
                                f"(Sharpe {best_sharpe:.2f} vs deployed {base_sharpe:.2f})")

        results_for_param['perturbations'] = {
            str(k): {'value': v['value'], 'sharpe': round(v['sharpe'], 2)}
            for k, v in perturb_sharpes.items()
        }
        results_for_param['best_value'] = best_val
        results_for_param['best_sharpe'] = round(best_sharpe, 2)
        results_for_param['has_drift'] = has_drift
        param_results[param_name] = results_for_param

        s_neg = perturb_sharpes[-0.10]['sharpe']
        s_pos = perturb_sharpes[+0.10]['sharpe']
        print(f"  {param_name:<26} {deployed_val:>9.4g} {s_neg:>9.2f} {s_pos:>9.2f} "
              f"{best_val:>9.4g} {drift_flag:>7}", flush=True)

    if drift_alerts:
        print(f"\n  [ALERT] Parameter drift detected:", flush=True)
        for a in drift_alerts:
            print(f"    - {a}", flush=True)
    else:
        print(f"\n  All parameters stable (no significant drift)", flush=True)

    return {'base_sharpe': round(base_sharpe, 2), 'param_results': param_results,
            'drift_alerts': drift_alerts}


# ═══════════════════════════════════════════════════════════════
# Phase 4: Recent Performance Window Analysis
# ═══════════════════════════════════════════════════════════════

def phase4_recent_windows(h1_df, data_bundle, all_trades):
    print("\n" + "="*70, flush=True)
    print("PHASE 4: Recent Performance Window Analysis", flush=True)
    print("="*70, flush=True)

    windows_days = [30, 60, 90, 180]
    now = h1_df.index[-1]
    window_results = {}
    regime_alerts = []

    for strat in STRAT_ORDER:
        strat_windows = {}
        full_stats = compute_trade_stats(all_trades.get(strat, []), P6_LOTS[strat])
        full_sharpe = full_stats['sharpe']

        for days in windows_days:
            cutoff = now - pd.Timedelta(days=days)
            recent_trades = [t for t in all_trades.get(strat, [])
                            if _get_exit_time(t) >= cutoff]
            stats = compute_trade_stats(recent_trades, P6_LOTS[strat])
            strat_windows[f'{days}d'] = stats

        window_results[strat] = {
            'full_sample': full_stats,
            'windows': strat_windows,
        }

        recent_90 = strat_windows.get('90d', {}).get('sharpe', 0)
        if full_sharpe > 0 and recent_90 < full_sharpe * 0.5:
            regime_alerts.append(
                f"[{strat}] Recent 90d Sharpe ({recent_90:.2f}) < 50% of full-sample ({full_sharpe:.2f})"
            )

    print(f"\n  {'Strategy':<14} {'Full':>7} {'30d':>7} {'60d':>7} {'90d':>7} {'180d':>7} {'Alert':>6}", flush=True)
    print(f"  {'-'*56}", flush=True)
    for strat in STRAT_ORDER:
        wr = window_results[strat]
        full_s = wr['full_sample']['sharpe']
        w30 = wr['windows']['30d']['sharpe']
        w60 = wr['windows']['60d']['sharpe']
        w90 = wr['windows']['90d']['sharpe']
        w180 = wr['windows']['180d']['sharpe']
        alert = "*" if any(strat in a for a in regime_alerts) else ""
        print(f"  {strat:<14} {full_s:>7.2f} {w30:>7.2f} {w60:>7.2f} "
              f"{w90:>7.2f} {w180:>7.2f} {alert:>6}", flush=True)

    if regime_alerts:
        print(f"\n  [ALERT] Regime change detected:", flush=True)
        for a in regime_alerts:
            print(f"    - {a}", flush=True)
    else:
        print(f"\n  No regime change detected in recent windows", flush=True)

    rolling_90d = compute_rolling_sharpe(all_trades, h1_df, window=90)

    return {'window_results': window_results, 'regime_alerts': regime_alerts,
            'rolling_90d_sharpe': rolling_90d}


def _get_exit_time(trade):
    et = trade.get('exit_time')
    if isinstance(et, str):
        return pd.Timestamp(et)
    if hasattr(et, 'tz') and et.tz is not None:
        return et.tz_localize(None) if hasattr(et, 'tz_localize') else et
    return pd.Timestamp(et) if et is not None else pd.Timestamp('2000-01-01')


def compute_rolling_sharpe(all_trades, h1_df, window=90):
    combined_daily = None
    for strat, trades in all_trades.items():
        daily = build_daily_equity(trades)
        if daily.empty:
            continue
        if combined_daily is None:
            combined_daily = daily.copy()
        else:
            combined_daily = combined_daily.add(daily, fill_value=0)
    if combined_daily is None or len(combined_daily) < window:
        return {}
    rolling_mean = combined_daily.rolling(window).mean()
    rolling_std = combined_daily.rolling(window).std()
    rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)
    rolling_sharpe = rolling_sharpe.dropna()
    result = {str(k): round(float(v), 2) for k, v in rolling_sharpe.tail(12).items()}
    return result


# ═══════════════════════════════════════════════════════════════
# Phase 5: Strategy Correlation Drift
# ═══════════════════════════════════════════════════════════════

def phase5_correlation_drift(all_trades, h1_df):
    print("\n" + "="*70, flush=True)
    print("PHASE 5: Strategy Correlation Drift", flush=True)
    print("="*70, flush=True)

    daily_returns = {}
    for strat in STRAT_ORDER:
        daily = build_daily_equity(all_trades.get(strat, []))
        if not daily.empty:
            daily_returns[strat] = daily

    if len(daily_returns) < 2:
        print("  Not enough strategies with trades for correlation analysis", flush=True)
        return {'historical_corr': {}, 'recent_corr': {}, 'drift_alerts': []}

    combined = pd.DataFrame(daily_returns)
    combined = combined.fillna(0)

    full_corr = combined.corr()

    now = combined.index[-1] if len(combined) > 0 else pd.Timestamp.now().date()
    cutoff_90 = now - timedelta(days=90)
    recent = combined[combined.index >= cutoff_90]
    recent_corr = recent.corr() if len(recent) > 30 else full_corr

    drift_alerts = []
    pairs_checked = []
    for i, s1 in enumerate(STRAT_ORDER):
        for j, s2 in enumerate(STRAT_ORDER):
            if j <= i:
                continue
            if s1 not in full_corr.columns or s2 not in full_corr.columns:
                continue
            hist_c = full_corr.loc[s1, s2]
            rec_c = recent_corr.loc[s1, s2] if s1 in recent_corr.columns and s2 in recent_corr.columns else hist_c
            delta = rec_c - hist_c
            pairs_checked.append({'pair': f"{s1}/{s2}", 'historical': round(hist_c, 3),
                                  'recent': round(rec_c, 3), 'delta': round(delta, 3)})
            if delta > 0.3:
                drift_alerts.append(
                    f"{s1}/{s2}: correlation increased by {delta:.3f} "
                    f"(historical={hist_c:.3f}, recent={rec_c:.3f})"
                )

    print(f"\n  {'Pair':<25} {'Historical':>11} {'Recent 90d':>11} {'Delta':>8}", flush=True)
    print(f"  {'-'*57}", flush=True)
    for p in pairs_checked:
        flag = " ***" if p['delta'] > 0.3 else ""
        print(f"  {p['pair']:<25} {p['historical']:>11.3f} {p['recent']:>11.3f} "
              f"{p['delta']:>+8.3f}{flag}", flush=True)

    if drift_alerts:
        print(f"\n  [ALERT] Diversification degradation:", flush=True)
        for a in drift_alerts:
            print(f"    - {a}", flush=True)
    else:
        print(f"\n  Correlations stable — diversification intact", flush=True)

    hist_dict = {f"{r}/{c}": round(float(full_corr.loc[r, c]), 3)
                 for r in full_corr.index for c in full_corr.columns if r < c}
    recent_dict = {f"{r}/{c}": round(float(recent_corr.loc[r, c]), 3)
                   for r in recent_corr.index for c in recent_corr.columns if r < c}

    return {'historical_corr': hist_dict, 'recent_corr': recent_dict,
            'pairs': pairs_checked, 'drift_alerts': drift_alerts}


# ═══════════════════════════════════════════════════════════════
# Phase 6: Summary Report
# ═══════════════════════════════════════════════════════════════

def determine_overall_health(p1, p2, p3, p4, p5):
    red_count = 0
    yellow_count = 0

    for strat in STRAT_ORDER:
        kfold_status = p2.get('kfold_results', {}).get(strat, {}).get('status', HEALTH_GREEN)
        if kfold_status == HEALTH_RED:
            red_count += 1
        elif kfold_status == HEALTH_YELLOW:
            yellow_count += 1

    if p3.get('drift_alerts'):
        yellow_count += 1
    if p4.get('regime_alerts'):
        red_count += len(p4['regime_alerts'])
    if p5.get('drift_alerts'):
        yellow_count += len(p5['drift_alerts'])

    portfolio_sharpe = p1.get('portfolio', {}).get('sharpe', 0)
    if portfolio_sharpe < 1.0:
        red_count += 2

    if red_count >= 2:
        return HEALTH_RED
    elif red_count >= 1 or yellow_count >= 2:
        return HEALTH_YELLOW
    return HEALTH_GREEN


def phase6_summary(p1, p2, p3, p4, p5, compare_path=None):
    print("\n" + "="*70, flush=True)
    print("PHASE 6: Summary Report", flush=True)
    print("="*70, flush=True)

    overall = determine_overall_health(p1, p2, p3, p4, p5)

    per_strategy_health = {}
    for strat in STRAT_ORDER:
        cur_sharpe = p1['strat_metrics'].get(strat, {}).get('sharpe', 0)
        ref_sharpe = REFERENCE_BASELINES[strat]['sharpe']
        kfold_status = p2.get('kfold_results', {}).get(strat, {}).get('status', HEALTH_GREEN)
        sharpe_drop = ((cur_sharpe - ref_sharpe) / ref_sharpe * 100) if ref_sharpe != 0 else 0

        if kfold_status == HEALTH_RED or sharpe_drop < -25:
            status = HEALTH_RED
        elif kfold_status == HEALTH_YELLOW or sharpe_drop < -10:
            status = HEALTH_YELLOW
        else:
            status = HEALTH_GREEN

        per_strategy_health[strat] = {
            'status': status,
            'current_sharpe': cur_sharpe,
            'reference_sharpe': ref_sharpe,
            'kfold_status': kfold_status,
        }

    action_items = []
    all_alerts = (p2.get('alerts', []) + p3.get('drift_alerts', []) +
                  p4.get('regime_alerts', []) + p5.get('drift_alerts', []))
    for alert in all_alerts:
        action_items.append(alert)

    recommendations = []
    if overall == HEALTH_RED:
        recommendations.append("URGENT: Review strategy parameters and consider reducing position sizes")
        recommendations.append("Run full parameter optimization (R127/R128)")
    elif overall == HEALTH_YELLOW:
        recommendations.append("Monitor closely over next 2 weeks")
        recommendations.append("Consider running R131 deep validation")
    else:
        recommendations.append("Portfolio healthy — continue current deployment")
        recommendations.append("Schedule next check in 1 month")

    for strat, health in per_strategy_health.items():
        if health['status'] == HEALTH_RED:
            recommendations.append(f"Review {strat}: consider pausing or reducing lot size")

    summary = {
        'timestamp': datetime.now().isoformat(),
        'overall_health': overall,
        'per_strategy_health': per_strategy_health,
        'portfolio_metrics': p1.get('portfolio', {}),
        'strat_metrics': {k: v for k, v in p1.get('strat_metrics', {}).items()},
        'kfold_summary': {
            strat: {'mean': r.get('mean', 0), 'min': r.get('min', 0), 'status': r.get('status', '')}
            for strat, r in p2.get('kfold_results', {}).items()
        },
        'param_stability': {
            'base_sharpe': p3.get('base_sharpe', 0),
            'drift_detected': len(p3.get('drift_alerts', [])) > 0,
            'drift_params': p3.get('drift_alerts', []),
        },
        'recent_windows': {
            strat: p4.get('window_results', {}).get(strat, {}).get('windows', {})
            for strat in STRAT_ORDER
        },
        'correlation_drift': {
            'alerts': p5.get('drift_alerts', []),
            'pairs': p5.get('pairs', []),
        },
        'action_items': action_items,
        'recommendations': recommendations,
    }

    print(f"\n  {'='*50}", flush=True)
    print(f"  OVERALL PORTFOLIO HEALTH: {overall}", flush=True)
    print(f"  {'='*50}", flush=True)

    print(f"\n  Per-strategy status:", flush=True)
    for strat in STRAT_ORDER:
        h = per_strategy_health[strat]
        print(f"    {strat:<14} {h['status']:<8} (Sharpe: {h['current_sharpe']:.2f} / "
              f"ref: {h['reference_sharpe']:.2f})", flush=True)

    if action_items:
        print(f"\n  Action Items ({len(action_items)}):", flush=True)
        for i, item in enumerate(action_items[:10], 1):
            print(f"    {i}. {item}", flush=True)
        if len(action_items) > 10:
            print(f"    ... and {len(action_items)-10} more", flush=True)

    print(f"\n  Recommendations:", flush=True)
    for r in recommendations:
        print(f"    - {r}", flush=True)

    if compare_path:
        print(f"\n  Comparing with previous snapshot: {compare_path}", flush=True)
        try:
            with open(compare_path, 'r') as f:
                prev = json.load(f)
            print_comparison(prev, summary)
        except Exception as e:
            print(f"    [WARN] Could not load comparison: {e}", flush=True)

    return summary


def print_comparison(prev, current):
    print(f"\n  {'='*50}", flush=True)
    print(f"  COMPARISON: Previous ({prev.get('timestamp', 'unknown')[:10]}) vs Current", flush=True)
    print(f"  {'='*50}", flush=True)

    prev_health = prev.get('overall_health', '?')
    cur_health = current.get('overall_health', '?')
    transition = f"{prev_health} -> {cur_health}"
    print(f"  Overall: {transition}", flush=True)

    print(f"\n  {'Strategy':<14} {'Prev Sharpe':>12} {'Cur Sharpe':>11} {'Change':>8}", flush=True)
    print(f"  {'-'*48}", flush=True)
    for strat in STRAT_ORDER:
        prev_s = prev.get('strat_metrics', {}).get(strat, {}).get('sharpe', 0)
        cur_s = current.get('strat_metrics', {}).get(strat, {}).get('sharpe', 0)
        delta = cur_s - prev_s
        print(f"  {strat:<14} {prev_s:>12.2f} {cur_s:>11.2f} {delta:>+8.2f}", flush=True)

    prev_port = prev.get('portfolio_metrics', {}).get('sharpe', 0)
    cur_port = current.get('portfolio_metrics', {}).get('sharpe', 0)
    print(f"\n  Portfolio Sharpe: {prev_port:.2f} -> {cur_port:.2f} ({cur_port-prev_port:+.2f})", flush=True)

    prev_alerts = len(prev.get('action_items', []))
    cur_alerts = len(current.get('action_items', []))
    print(f"  Alerts: {prev_alerts} -> {cur_alerts}", flush=True)


def save_results(summary, tag=""):
    today = datetime.now().strftime("%Y-%m-%d")

    json_path = SNAPSHOT_DIR / f"{today}.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Saved JSON snapshot: {json_path}", flush=True)

    latest_path = OUTPUT_DIR / "latest.json"
    with open(latest_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Saved latest: {latest_path}", flush=True)

    txt_path = OUTPUT_DIR / f"report_{today}.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"Alpha Decay Monitor Report — {today}\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Overall Health: {summary['overall_health']}\n\n")
        f.write(f"Portfolio: Sharpe={summary['portfolio_metrics'].get('sharpe', 0):.2f}, "
                f"PnL=${summary['portfolio_metrics'].get('total_pnl', 0):.0f}, "
                f"MaxDD=${summary['portfolio_metrics'].get('max_dd', 0):.0f}\n\n")
        f.write(f"Per-Strategy:\n")
        for strat in STRAT_ORDER:
            h = summary['per_strategy_health'].get(strat, {})
            m = summary['strat_metrics'].get(strat, {})
            f.write(f"  {strat:<14} {h.get('status', '?'):<8} "
                    f"Sharpe={m.get('sharpe', 0):.2f}  "
                    f"PnL=${m.get('total_pnl', 0):.0f}  "
                    f"Trades={m.get('n_trades', 0)}  "
                    f"WR={m.get('win_rate', 0):.1%}\n")
        f.write(f"\nK-Fold Summary:\n")
        for strat, kf in summary.get('kfold_summary', {}).items():
            f.write(f"  {strat:<14} Mean={kf.get('mean', 0):.2f}, "
                    f"Min={kf.get('min', 0):.2f}, Status={kf.get('status', '?')}\n")
        f.write(f"\nParam Stability (L8_MAX):\n")
        ps = summary.get('param_stability', {})
        f.write(f"  Base Sharpe: {ps.get('base_sharpe', 0):.2f}\n")
        f.write(f"  Drift detected: {ps.get('drift_detected', False)}\n")
        for d in ps.get('drift_params', []):
            f.write(f"  - {d}\n")
        f.write(f"\nAction Items ({len(summary.get('action_items', []))}):\n")
        for item in summary.get('action_items', []):
            f.write(f"  - {item}\n")
        f.write(f"\nRecommendations:\n")
        for r in summary.get('recommendations', []):
            f.write(f"  - {r}\n")
    print(f"  Saved TXT report: {txt_path}", flush=True)


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="R174 — Alpha Decay Monitor")
    parser.add_argument('--compare', type=str, help='Path to previous snapshot JSON for comparison')
    parser.add_argument('--quick', action='store_true', help='Skip K-Fold and param tests (faster)')
    args = parser.parse_args()

    print("="*70, flush=True)
    print("R174 — Adaptive Alpha Decay Monitor", flush=True)
    print(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"Mode: {'QUICK' if args.quick else 'FULL'}", flush=True)
    print("="*70, flush=True)

    t_start = time.time()

    print("\nLoading data...", flush=True)
    h1_df = load_h1()
    print(f"  H1: {len(h1_df)} bars, {h1_df.index[0]} -> {h1_df.index[-1]}", flush=True)

    from backtest.runner import DataBundle
    print("  Loading M15 + full data bundle for L8_MAX...", flush=True)
    data_bundle = DataBundle.load_default()

    # Phase 1: always run
    p1 = phase1_snapshot(h1_df, data_bundle)
    p1_save = {k: v for k, v in p1.items() if k != 'all_trades'}
    with open(OUTPUT_DIR / "phase1.json", 'w') as f:
        json.dump(p1_save, f, indent=2, default=str)
    print(f"\n  [Phase 1 saved]", flush=True)

    # Phase 2: K-Fold (skip in quick mode)
    if args.quick:
        print("\n  [QUICK MODE] Skipping Phase 2 (K-Fold)", flush=True)
        p2 = {'kfold_results': {s: {'status': HEALTH_GREEN, 'mean': 0, 'min': 0}
               for s in STRAT_ORDER}, 'alerts': []}
    else:
        p2 = phase2_kfold(h1_df, data_bundle)
        with open(OUTPUT_DIR / "phase2.json", 'w') as f:
            json.dump(p2, f, indent=2, default=str)
        print(f"\n  [Phase 2 saved]", flush=True)

    # Phase 3: Param stability (skip in quick mode)
    if args.quick:
        print("\n  [QUICK MODE] Skipping Phase 3 (Param Stability)", flush=True)
        p3 = {'base_sharpe': 0, 'param_results': {}, 'drift_alerts': []}
    else:
        p3 = phase3_param_stability(data_bundle)
        with open(OUTPUT_DIR / "phase3.json", 'w') as f:
            json.dump(p3, f, indent=2, default=str)
        print(f"\n  [Phase 3 saved]", flush=True)

    # Phase 4: Recent windows (always run)
    p4 = phase4_recent_windows(h1_df, data_bundle, p1['all_trades'])
    p4_save = {k: v for k, v in p4.items() if k != 'rolling_90d_sharpe' or isinstance(v, dict)}
    with open(OUTPUT_DIR / "phase4.json", 'w') as f:
        json.dump(p4_save, f, indent=2, default=str)
    print(f"\n  [Phase 4 saved]", flush=True)

    # Phase 5: Correlation drift (always run)
    p5 = phase5_correlation_drift(p1['all_trades'], h1_df)
    with open(OUTPUT_DIR / "phase5.json", 'w') as f:
        json.dump(p5, f, indent=2, default=str)
    print(f"\n  [Phase 5 saved]", flush=True)

    # Phase 6: Summary
    summary = phase6_summary(p1, p2, p3, p4, p5, compare_path=args.compare)
    save_results(summary)

    elapsed_total = time.time() - t_start
    print(f"\n{'='*70}", flush=True)
    print(f"R174 complete in {elapsed_total/60:.1f} minutes", flush=True)
    print(f"Overall health: {summary['overall_health']}", flush=True)
    print(f"Results: {OUTPUT_DIR}/", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()

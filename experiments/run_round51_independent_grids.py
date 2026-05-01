#!/usr/bin/env python3
"""
Round 51 — Independent Strategy Full Parameter Grid Search
===========================================================
Brute-force grid search for 4 independent alpha sources:
  A) D1 Keltner Channel breakout
  B) H4 Keltner Channel breakout
  C) SuperTrend
  D) PSAR

Each strategy: Layer 1 full grid → Top 50 K-Fold 6-Fold validation

USAGE (server)
--------------
    cd /root/gold-quant-research
    nohup python3 -u experiments/run_round51_independent_grids.py \
        > results/round51_results/stdout.txt 2>&1 &
"""
import sys, os, io, time, json, traceback
import multiprocessing as mp
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
from typing import Dict, List, Tuple

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

_script_dir = os.path.dirname(os.path.abspath(__file__))
for _candidate in [os.path.join(_script_dir, '..'), os.path.join(_script_dir, '..', '..'), os.getcwd()]:
    _candidate = os.path.abspath(_candidate)
    if os.path.isdir(os.path.join(_candidate, 'backtest')):
        sys.path.insert(0, _candidate)
        os.chdir(_candidate)
        break

OUTPUT_DIR = Path("results/round51_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_WORKERS = max(1, mp.cpu_count() - 1)
SPREAD = 0.50

KFOLD_FOLDS = [
    ("F1_2015-2016", "2015-01-01", "2016-12-31"),
    ("F2_2017-2018", "2017-01-01", "2018-12-31"),
    ("F3_2019-2020", "2019-01-01", "2020-12-31"),
    ("F4_2021-2022", "2021-01-01", "2022-12-31"),
    ("F5_2023-2024", "2023-01-01", "2024-12-31"),
    ("F6_2025-2026", "2025-01-01", "2026-12-31"),
]


# ═══════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════

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


def add_kc(df, ema_period=20, atr_period=14, mult=1.5):
    df = df.copy()
    df['EMA'] = df['Close'].ewm(span=ema_period, adjust=False).mean()
    tr = pd.DataFrame({
        'hl': df['High'] - df['Low'],
        'hc': (df['High'] - df['Close'].shift(1)).abs(),
        'lc': (df['Low'] - df['Close'].shift(1)).abs(),
    }).max(axis=1)
    df['ATR'] = tr.rolling(atr_period).mean()
    df['KC_upper'] = df['EMA'] + mult * df['ATR']
    df['KC_lower'] = df['EMA'] - mult * df['ATR']
    df['ADX'] = compute_adx(df, atr_period)
    return df


def calc_stats(trades, label=""):
    if not trades:
        return {'label': label, 'n': 0, 'sharpe': 0, 'total_pnl': 0,
                'win_rate': 0, 'max_dd': 0, 'avg_win': 0, 'avg_loss': 0}
    pnls = [t['pnl'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    eq = np.cumsum(pnls)
    dd = (np.maximum.accumulate(eq + 2000) - (eq + 2000)).max()
    daily = {}
    for t in trades:
        d = pd.Timestamp(t['exit_time']).date()
        daily[d] = daily.get(d, 0) + t['pnl']
    da = np.array(list(daily.values()))
    sh = float(da.mean() / da.std() * np.sqrt(252)) if len(da) > 1 and da.std() > 0 else 0.0
    return {
        'label': label, 'n': len(pnls), 'sharpe': round(sh, 2),
        'total_pnl': round(sum(pnls), 2), 'win_rate': round(len(wins)/len(pnls)*100, 1),
        'max_dd': round(dd, 2),
        'avg_win': round(np.mean(wins), 2) if wins else 0,
        'avg_loss': round(abs(np.mean(losses)), 2) if losses else 0,
    }


def save_checkpoint(data, filename):
    path = OUTPUT_DIR / filename
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, default=str)
    print(f"  [Checkpoint] {path} ({len(data) if isinstance(data, list) else 'dict'})", flush=True)


def load_checkpoint(filename):
    path = OUTPUT_DIR / filename
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def fmt(x):
    return f"${x:,.0f}" if abs(x) >= 1 else f"${x:.2f}"


def write_ranking(results, filename, title):
    valid = [r for r in results if r.get('sharpe', 0) > 0 and r.get('n', 0) > 0]
    valid.sort(key=lambda x: x['sharpe'], reverse=True)
    lines = [title, "=" * 100, f"Total: {len(valid)} positive Sharpe / {len(results)} total", ""]
    lines.append(f"{'Rank':>4} {'Label':>50} {'Sharpe':>8} {'PnL':>12} {'N':>6} "
                 f"{'MaxDD':>10} {'WR':>6} {'AvgW':>8} {'AvgL':>8}")
    lines.append(f"{'':>4} {'-'*50} {'-'*8} {'-'*12} {'-'*6} {'-'*10} {'-'*6} {'-'*8} {'-'*8}")
    for i, r in enumerate(valid[:100], 1):
        lines.append(f"{i:>4} {r['label']:>50} {r['sharpe']:>8.2f} {fmt(r['total_pnl']):>12} "
                     f"{r['n']:>6} {fmt(r['max_dd']):>10} {r['win_rate']:>5.1f}% "
                     f"{r.get('avg_win',0):>8.2f} {r.get('avg_loss',0):>8.2f}")
    with open(OUTPUT_DIR / filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  Ranking saved: {filename}", flush=True)


# ═══════════════════════════════════════════════════════════════
# A) KC Breakout Backtest (works for D1 and H4)
# ═══════════════════════════════════════════════════════════════

def backtest_kc(df_prepared, label, adx_thresh=18,
                sl_atr=3.5, tp_atr=8.0, trail_act_atr=0.28, trail_dist_atr=0.06,
                max_hold=20, spread=SPREAD, lot=0.03):
    """KC breakout backtest on pre-prepared dataframe (with KC/ADX columns)."""
    df = df_prepared
    trades = []; pos = None
    close = df['Close'].values; high = df['High'].values; low = df['Low'].values
    kc_up = df['KC_upper'].values; kc_lo = df['KC_lower'].values
    atr = df['ATR'].values; adx_arr = df['ADX'].values
    times = df.index; n = len(df); last_exit = -999

    for i in range(1, n):
        c = close[i]; h = high[i]; lo_v = low[i]
        cur_atr = atr[i]; cur_adx = adx_arr[i]
        if pos is not None:
            held = i - pos['bar']
            if pos['dir'] == 'BUY':
                pnl_h = (h - pos['entry'] - spread) * lot * 100
                pnl_l = (lo_v - pos['entry'] - spread) * lot * 100
                pnl_c = (c - pos['entry'] - spread) * lot * 100
            else:
                pnl_h = (pos['entry'] - lo_v - spread) * lot * 100
                pnl_l = (pos['entry'] - h - spread) * lot * 100
                pnl_c = (pos['entry'] - c - spread) * lot * 100
            tp_val = tp_atr * pos['atr'] * lot * 100
            sl_val = sl_atr * pos['atr'] * lot * 100
            exited = False
            if pnl_h >= tp_val:
                trades.append(_mk(pos, c, times[i], "TP", i, tp_val)); exited = True
            elif pnl_l <= -sl_val:
                trades.append(_mk(pos, c, times[i], "SL", i, -sl_val)); exited = True
            else:
                act_dist = trail_act_atr * pos['atr']; trail_d = trail_dist_atr * pos['atr']
                if pos['dir'] == 'BUY' and h - pos['entry'] >= act_dist:
                    ts_p = h - trail_d
                    if lo_v <= ts_p:
                        trades.append(_mk(pos, c, times[i], "Trail", i,
                                          (ts_p - pos['entry'] - spread) * lot * 100)); exited = True
                elif pos['dir'] == 'SELL' and pos['entry'] - lo_v >= act_dist:
                    ts_p = lo_v + trail_d
                    if h >= ts_p:
                        trades.append(_mk(pos, c, times[i], "Trail", i,
                                          (pos['entry'] - ts_p - spread) * lot * 100)); exited = True
                if not exited and held >= max_hold:
                    trades.append(_mk(pos, c, times[i], "Timeout", i, pnl_c)); exited = True
            if exited: pos = None; last_exit = i; continue
        if pos is not None: continue
        if i - last_exit < 2: continue
        if np.isnan(cur_adx) or cur_adx < adx_thresh: continue
        if np.isnan(cur_atr) or cur_atr < 0.1: continue
        prev_c = close[i-1]
        if prev_c > kc_up[i-1]:
            pos = {'dir': 'BUY', 'entry': c + spread/2, 'bar': i, 'time': times[i], 'atr': cur_atr}
        elif prev_c < kc_lo[i-1]:
            pos = {'dir': 'SELL', 'entry': c - spread/2, 'bar': i, 'time': times[i], 'atr': cur_atr}

    return calc_stats(trades, label)


def _mk(pos, exit_p, exit_time, reason, bar_idx, pnl):
    return {'dir': pos['dir'], 'entry': pos['entry'], 'exit': exit_p,
            'entry_time': pos['time'], 'exit_time': exit_time,
            'pnl': pnl, 'reason': reason, 'bars': bar_idx - pos['bar']}


# ═══════════════════════════════════════════════════════════════
# B) SuperTrend Backtest
# ═══════════════════════════════════════════════════════════════

def add_supertrend(df, period=10, factor=3.0):
    df = df.copy()
    tr = pd.DataFrame({
        'hl': df['High'] - df['Low'],
        'hc': (df['High'] - df['Close'].shift(1)).abs(),
        'lc': (df['Low'] - df['Close'].shift(1)).abs(),
    }).max(axis=1)
    atr = tr.rolling(period).mean()
    hl2 = (df['High'] + df['Low']) / 2
    upper = hl2 + factor * atr
    lower = hl2 - factor * atr

    st_upper = upper.copy(); st_lower = lower.copy()
    direction = pd.Series(1, index=df.index)
    for i in range(1, len(df)):
        if lower.iloc[i] > st_lower.iloc[i-1] or df['Close'].iloc[i-1] < st_lower.iloc[i-1]:
            st_lower.iloc[i] = lower.iloc[i]
        else:
            st_lower.iloc[i] = st_lower.iloc[i-1]
        if upper.iloc[i] < st_upper.iloc[i-1] or df['Close'].iloc[i-1] > st_upper.iloc[i-1]:
            st_upper.iloc[i] = upper.iloc[i]
        else:
            st_upper.iloc[i] = st_upper.iloc[i-1]
        if direction.iloc[i-1] == 1:
            direction.iloc[i] = -1 if df['Close'].iloc[i] < st_lower.iloc[i] else 1
        else:
            direction.iloc[i] = 1 if df['Close'].iloc[i] > st_upper.iloc[i] else -1

    df['ST_dir'] = direction
    df['ATR'] = atr
    return df


def backtest_supertrend(df_prepared, label, sl_atr=3.5, tp_atr=8.0,
                        trail_act_atr=0.28, trail_dist_atr=0.06,
                        max_hold=50, spread=SPREAD, lot=0.03):
    df = df_prepared.dropna(subset=['ST_dir', 'ATR'])
    trades = []; pos = None
    close = df['Close'].values; high = df['High'].values; low = df['Low'].values
    st_dir = df['ST_dir'].values; atr = df['ATR'].values
    times = df.index; n = len(df); last_exit = -999

    for i in range(1, n):
        c = close[i]; h = high[i]; lo_v = low[i]; cur_atr = atr[i]
        if pos is not None:
            held = i - pos['bar']
            if pos['dir'] == 'BUY':
                pnl_h = (h - pos['entry'] - spread) * lot * 100
                pnl_l = (lo_v - pos['entry'] - spread) * lot * 100
                pnl_c = (c - pos['entry'] - spread) * lot * 100
            else:
                pnl_h = (pos['entry'] - lo_v - spread) * lot * 100
                pnl_l = (pos['entry'] - h - spread) * lot * 100
                pnl_c = (pos['entry'] - c - spread) * lot * 100
            tp_val = tp_atr * pos['atr'] * lot * 100
            sl_val = sl_atr * pos['atr'] * lot * 100
            exited = False
            if pnl_h >= tp_val:
                trades.append(_mk(pos, c, times[i], "TP", i, tp_val)); exited = True
            elif pnl_l <= -sl_val:
                trades.append(_mk(pos, c, times[i], "SL", i, -sl_val)); exited = True
            else:
                act_dist = trail_act_atr * pos['atr']; trail_d = trail_dist_atr * pos['atr']
                if pos['dir'] == 'BUY' and h - pos['entry'] >= act_dist:
                    ts_p = h - trail_d
                    if lo_v <= ts_p:
                        trades.append(_mk(pos, c, times[i], "Trail", i,
                                          (ts_p - pos['entry'] - spread) * lot * 100)); exited = True
                elif pos['dir'] == 'SELL' and pos['entry'] - lo_v >= act_dist:
                    ts_p = lo_v + trail_d
                    if h >= ts_p:
                        trades.append(_mk(pos, c, times[i], "Trail", i,
                                          (pos['entry'] - ts_p - spread) * lot * 100)); exited = True
                if not exited and held >= max_hold:
                    trades.append(_mk(pos, c, times[i], "Timeout", i, pnl_c)); exited = True
            if exited: pos = None; last_exit = i; continue
        if pos is not None: continue
        if i - last_exit < 2: continue
        if np.isnan(cur_atr) or cur_atr < 0.1: continue
        prev_dir = st_dir[i-1]; cur_dir_v = st_dir[i]
        if prev_dir == -1 and cur_dir_v == 1:
            pos = {'dir': 'BUY', 'entry': c + spread/2, 'bar': i, 'time': times[i], 'atr': cur_atr}
        elif prev_dir == 1 and cur_dir_v == -1:
            pos = {'dir': 'SELL', 'entry': c - spread/2, 'bar': i, 'time': times[i], 'atr': cur_atr}

    return calc_stats(trades, label)


# ═══════════════════════════════════════════════════════════════
# C) PSAR Backtest
# ═══════════════════════════════════════════════════════════════

def add_psar(df, af_start=0.02, af_max=0.20):
    df = df.copy()
    n = len(df); psar = np.zeros(n); direction = np.ones(n)
    af = af_start; ep = df['High'].iloc[0]
    psar[0] = df['Low'].iloc[0]

    for i in range(1, n):
        prev_psar = psar[i-1]
        if direction[i-1] == 1:
            psar[i] = prev_psar + af * (ep - prev_psar)
            psar[i] = min(psar[i], df['Low'].iloc[i-1], df['Low'].iloc[max(0,i-2)])
            if df['Low'].iloc[i] < psar[i]:
                direction[i] = -1; psar[i] = ep; ep = df['Low'].iloc[i]; af = af_start
            else:
                direction[i] = 1
                if df['High'].iloc[i] > ep:
                    ep = df['High'].iloc[i]; af = min(af + af_start, af_max)
        else:
            psar[i] = prev_psar + af * (ep - prev_psar)
            psar[i] = max(psar[i], df['High'].iloc[i-1], df['High'].iloc[max(0,i-2)])
            if df['High'].iloc[i] > psar[i]:
                direction[i] = 1; psar[i] = ep; ep = df['High'].iloc[i]; af = af_start
            else:
                direction[i] = -1
                if df['Low'].iloc[i] < ep:
                    ep = df['Low'].iloc[i]; af = min(af + af_start, af_max)

    df['PSAR'] = psar; df['PSAR_dir'] = direction
    tr = pd.DataFrame({
        'hl': df['High'] - df['Low'],
        'hc': (df['High'] - df['Close'].shift(1)).abs(),
        'lc': (df['Low'] - df['Close'].shift(1)).abs(),
    }).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()
    return df


def backtest_psar(df_prepared, label, sl_atr=3.5, tp_atr=8.0,
                  trail_act_atr=0.28, trail_dist_atr=0.06,
                  max_hold=50, spread=SPREAD, lot=0.03):
    df = df_prepared.dropna(subset=['PSAR_dir', 'ATR'])
    trades = []; pos = None
    close = df['Close'].values; high = df['High'].values; low = df['Low'].values
    psar_dir = df['PSAR_dir'].values; atr = df['ATR'].values
    times = df.index; n = len(df); last_exit = -999

    for i in range(1, n):
        c = close[i]; h = high[i]; lo_v = low[i]; cur_atr = atr[i]
        if pos is not None:
            held = i - pos['bar']
            if pos['dir'] == 'BUY':
                pnl_h = (h - pos['entry'] - spread) * lot * 100
                pnl_l = (lo_v - pos['entry'] - spread) * lot * 100
                pnl_c = (c - pos['entry'] - spread) * lot * 100
            else:
                pnl_h = (pos['entry'] - lo_v - spread) * lot * 100
                pnl_l = (pos['entry'] - h - spread) * lot * 100
                pnl_c = (pos['entry'] - c - spread) * lot * 100
            tp_val = tp_atr * pos['atr'] * lot * 100
            sl_val = sl_atr * pos['atr'] * lot * 100
            exited = False
            if pnl_h >= tp_val:
                trades.append(_mk(pos, c, times[i], "TP", i, tp_val)); exited = True
            elif pnl_l <= -sl_val:
                trades.append(_mk(pos, c, times[i], "SL", i, -sl_val)); exited = True
            else:
                act_dist = trail_act_atr * pos['atr']; trail_d = trail_dist_atr * pos['atr']
                if pos['dir'] == 'BUY' and h - pos['entry'] >= act_dist:
                    ts_p = h - trail_d
                    if lo_v <= ts_p:
                        trades.append(_mk(pos, c, times[i], "Trail", i,
                                          (ts_p - pos['entry'] - spread) * lot * 100)); exited = True
                elif pos['dir'] == 'SELL' and pos['entry'] - lo_v >= act_dist:
                    ts_p = lo_v + trail_d
                    if h >= ts_p:
                        trades.append(_mk(pos, c, times[i], "Trail", i,
                                          (pos['entry'] - ts_p - spread) * lot * 100)); exited = True
                if not exited and held >= max_hold:
                    trades.append(_mk(pos, c, times[i], "Timeout", i, pnl_c)); exited = True
            if exited: pos = None; last_exit = i; continue
        if pos is not None: continue
        if i - last_exit < 2: continue
        if np.isnan(cur_atr) or cur_atr < 0.1: continue
        prev_d = psar_dir[i-1]; cur_d = psar_dir[i]
        if prev_d == -1 and cur_d == 1:
            pos = {'dir': 'BUY', 'entry': c + spread/2, 'bar': i, 'time': times[i], 'atr': cur_atr}
        elif prev_d == 1 and cur_d == -1:
            pos = {'dir': 'SELL', 'entry': c - spread/2, 'bar': i, 'time': times[i], 'atr': cur_atr}

    return calc_stats(trades, label)


# ═══════════════════════════════════════════════════════════════
# Grid definitions
# ═══════════════════════════════════════════════════════════════

D1_KC_GRID = {
    'ema': [10, 15, 20, 25, 30, 40],
    'mult': [1.0, 1.5, 2.0, 2.5, 3.0],
    'adx': [10, 14, 18, 22, 26],
    'sl': [2.0, 3.0, 3.5, 4.5],
    'tp': [6.0, 8.0, 12.0, 16.0],
    'mh': [5, 8, 12, 15, 20, 30],
    'trail': [(0.20, 0.05), (0.30, 0.08), (0.40, 0.10), (0.50, 0.15)],
}

H4_KC_GRID = {
    'ema': [10, 15, 20, 25, 30, 40],
    'mult': [1.0, 1.2, 1.5, 2.0, 2.5],
    'adx': [10, 14, 18, 22, 26],
    'sl': [2.0, 3.0, 3.5, 4.5],
    'tp': [6.0, 8.0, 12.0, 16.0],
    'mh': [10, 15, 20, 30, 50],
    'trail': [(0.20, 0.04), (0.28, 0.06), (0.40, 0.10), (0.50, 0.15)],
}

ST_GRID = {
    'period': [5, 7, 10, 14, 20, 30],
    'factor': [1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
    'sl': [2.0, 3.0, 3.5, 4.5],
    'tp': [6.0, 8.0, 12.0, 16.0],
    'mh': [10, 20, 30, 50, 80],
    'trail': [(0.20, 0.04), (0.28, 0.06), (0.40, 0.10)],
}

PSAR_GRID = {
    'af_start': [0.005, 0.01, 0.02, 0.03, 0.05],
    'af_max': [0.05, 0.10, 0.15, 0.20, 0.30],
    'sl': [2.0, 3.0, 3.5, 4.5],
    'tp': [6.0, 8.0, 12.0, 16.0],
    'mh': [10, 20, 30, 50, 80],
    'trail': [(0.20, 0.04), (0.28, 0.06), (0.40, 0.10)],
}


# ═══════════════════════════════════════════════════════════════
# Worker functions for multiprocessing
# ═══════════════════════════════════════════════════════════════

def _worker_kc(args):
    df_path, label, ema, mult, adx, sl, tp, mh, trail_a, trail_d = args
    df = pd.read_pickle(df_path)
    df = add_kc(df, ema, 14, mult)
    df = df.dropna()
    return backtest_kc(df, label, adx_thresh=adx, sl_atr=sl, tp_atr=tp,
                       trail_act_atr=trail_a, trail_dist_atr=trail_d, max_hold=mh)


def _worker_st(args):
    df_path, label, sl, tp, mh, trail_a, trail_d = args
    df = pd.read_pickle(df_path)
    return backtest_supertrend(df, label, sl_atr=sl, tp_atr=tp,
                               trail_act_atr=trail_a, trail_dist_atr=trail_d, max_hold=mh)


def _worker_psar(args):
    df_path, label, sl, tp, mh, trail_a, trail_d = args
    df = pd.read_pickle(df_path)
    return backtest_psar(df, label, sl_atr=sl, tp_atr=tp,
                         trail_act_atr=trail_a, trail_dist_atr=trail_d, max_hold=mh)


# ═══════════════════════════════════════════════════════════════
# Strategy runners
# ═══════════════════════════════════════════════════════════════

def run_kc_grid(h1_df, timeframe, grid, checkpoint_name):
    tf_label = timeframe.upper()
    print(f"\n{'='*80}")
    print(f"  {tf_label} Keltner Channel — Full Parameter Grid")
    print(f"{'='*80}")

    existing = load_checkpoint(checkpoint_name)
    if existing:
        print(f"  [Resume] Found {len(existing)} results, skipping", flush=True)
        return existing

    if timeframe == 'd1':
        df = h1_df.resample('D').agg({'Open':'first','High':'max','Low':'min',
                                       'Close':'last','Volume':'sum'}).dropna()
    elif timeframe == 'h4':
        df = h1_df.resample('4h').agg({'Open':'first','High':'max','Low':'min',
                                        'Close':'last','Volume':'sum'}).dropna()
    else:
        df = h1_df

    tmp_path = OUTPUT_DIR / f"_tmp_{tf_label}.pkl"
    df.to_pickle(tmp_path)

    tasks = []
    for ema, mult, adx, sl, tp, mh in product(
        grid['ema'], grid['mult'], grid['adx'], grid['sl'], grid['tp'], grid['mh']
    ):
        for trail_a, trail_d in grid['trail']:
            label = f"{tf_label}_E{ema}_M{mult}_ADX{adx}_SL{sl}_TP{tp}_MH{mh}_T{trail_a}/{trail_d}"
            tasks.append((str(tmp_path), label, ema, mult, adx, sl, tp, mh, trail_a, trail_d))

    total = len(tasks)
    print(f"  Total combos: {total:,}", flush=True)
    print(f"  Workers: {MAX_WORKERS}", flush=True)

    t0 = time.time()
    all_results = []
    batch_size = max(100, total // 20)

    with mp.Pool(MAX_WORKERS) as pool:
        for i, result in enumerate(pool.imap_unordered(_worker_kc, tasks, chunksize=4)):
            all_results.append(result)
            if (i + 1) % batch_size == 0 or (i + 1) == total:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (total - i - 1) / rate if rate > 0 else 0
                print(f"    {i+1}/{total} ({(i+1)/total*100:.0f}%) | "
                      f"{elapsed/60:.1f}min | ETA {eta/60:.1f}min", flush=True)
                save_checkpoint(all_results, checkpoint_name)

    tmp_path.unlink(missing_ok=True)

    elapsed = time.time() - t0
    valid = [r for r in all_results if r.get('sharpe', 0) > 0]
    print(f"\n  Done: {len(valid)} positive / {len(all_results)} total in {elapsed/60:.1f}min")

    save_checkpoint(all_results, checkpoint_name)
    write_ranking(all_results, checkpoint_name.replace('.json', '_ranking.txt'),
                  f"R51 {tf_label} KC Grid Results")
    return all_results


def run_st_grid(h1_df, checkpoint_name="st_h1_grid.json"):
    print(f"\n{'='*80}")
    print(f"  H1 SuperTrend — Full Parameter Grid")
    print(f"{'='*80}")

    existing = load_checkpoint(checkpoint_name)
    if existing:
        print(f"  [Resume] Found {len(existing)} results, skipping", flush=True)
        return existing

    indicator_combos = list(product(ST_GRID['period'], ST_GRID['factor']))
    print(f"  Pre-computing {len(indicator_combos)} SuperTrend indicator sets...", flush=True)
    precomp = {}
    for idx, (period, factor) in enumerate(indicator_combos):
        key = f"P{period}_F{factor}"
        tmp = OUTPUT_DIR / f"_tmp_ST_{key}.pkl"
        df_st = add_supertrend(h1_df.copy(), period, factor)
        df_st.to_pickle(tmp)
        precomp[key] = str(tmp)
        print(f"    [{idx+1}/{len(indicator_combos)}] {key} done", flush=True)
    print(f"  Pre-computation done!", flush=True)

    tasks = []
    for period, factor, sl, tp, mh in product(
        ST_GRID['period'], ST_GRID['factor'], ST_GRID['sl'], ST_GRID['tp'], ST_GRID['mh']
    ):
        for trail_a, trail_d in ST_GRID['trail']:
            key = f"P{period}_F{factor}"
            label = f"ST_P{period}_F{factor}_SL{sl}_TP{tp}_MH{mh}_T{trail_a}/{trail_d}"
            tasks.append((precomp[key], label, sl, tp, mh, trail_a, trail_d))

    total = len(tasks)
    print(f"  Total combos: {total:,}", flush=True)

    t0 = time.time()
    all_results = []
    batch_size = max(100, total // 20)

    with mp.Pool(MAX_WORKERS) as pool:
        for i, result in enumerate(pool.imap_unordered(_worker_st, tasks, chunksize=8)):
            all_results.append(result)
            if (i + 1) % batch_size == 0 or (i + 1) == total:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (total - i - 1) / rate if rate > 0 else 0
                print(f"    {i+1}/{total} ({(i+1)/total*100:.0f}%) | "
                      f"{elapsed/60:.1f}min | ETA {eta/60:.1f}min", flush=True)
                save_checkpoint(all_results, checkpoint_name)

    for p in precomp.values():
        Path(p).unlink(missing_ok=True)

    elapsed = time.time() - t0
    valid = [r for r in all_results if r.get('sharpe', 0) > 0]
    print(f"\n  Done: {len(valid)} positive / {len(all_results)} total in {elapsed/60:.1f}min")

    save_checkpoint(all_results, checkpoint_name)
    write_ranking(all_results, checkpoint_name.replace('.json', '_ranking.txt'),
                  "R51 H1 SuperTrend Grid Results")
    return all_results


def run_psar_grid(h1_df, checkpoint_name="psar_h1_grid.json"):
    print(f"\n{'='*80}")
    print(f"  H1 PSAR — Full Parameter Grid")
    print(f"{'='*80}")

    existing = load_checkpoint(checkpoint_name)
    if existing:
        print(f"  [Resume] Found {len(existing)} results, skipping", flush=True)
        return existing

    indicator_combos = list(product(PSAR_GRID['af_start'], PSAR_GRID['af_max']))
    print(f"  Pre-computing {len(indicator_combos)} PSAR indicator sets...", flush=True)
    precomp = {}
    for idx, (af_s, af_m) in enumerate(indicator_combos):
        key = f"AF{af_s}_MX{af_m}"
        tmp = OUTPUT_DIR / f"_tmp_PSAR_{key}.pkl"
        df_psar = add_psar(h1_df.copy(), af_s, af_m)
        df_psar.to_pickle(tmp)
        precomp[key] = str(tmp)
        print(f"    [{idx+1}/{len(indicator_combos)}] {key} done", flush=True)
    print(f"  Pre-computation done!", flush=True)

    tasks = []
    for af_s, af_m, sl, tp, mh in product(
        PSAR_GRID['af_start'], PSAR_GRID['af_max'], PSAR_GRID['sl'],
        PSAR_GRID['tp'], PSAR_GRID['mh']
    ):
        for trail_a, trail_d in PSAR_GRID['trail']:
            key = f"AF{af_s}_MX{af_m}"
            label = f"PSAR_AF{af_s}_MX{af_m}_SL{sl}_TP{tp}_MH{mh}_T{trail_a}/{trail_d}"
            tasks.append((precomp[key], label, sl, tp, mh, trail_a, trail_d))

    total = len(tasks)
    print(f"  Total combos: {total:,}", flush=True)

    t0 = time.time()
    all_results = []
    batch_size = max(100, total // 20)

    with mp.Pool(MAX_WORKERS) as pool:
        for i, result in enumerate(pool.imap_unordered(_worker_psar, tasks, chunksize=8)):
            all_results.append(result)
            if (i + 1) % batch_size == 0 or (i + 1) == total:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (total - i - 1) / rate if rate > 0 else 0
                print(f"    {i+1}/{total} ({(i+1)/total*100:.0f}%) | "
                      f"{elapsed/60:.1f}min | ETA {eta/60:.1f}min", flush=True)
                save_checkpoint(all_results, checkpoint_name)

    for p in precomp.values():
        Path(p).unlink(missing_ok=True)

    elapsed = time.time() - t0
    valid = [r for r in all_results if r.get('sharpe', 0) > 0]
    print(f"\n  Done: {len(valid)} positive / {len(all_results)} total in {elapsed/60:.1f}min")

    save_checkpoint(all_results, checkpoint_name)
    write_ranking(all_results, checkpoint_name.replace('.json', '_ranking.txt'),
                  "R51 H1 PSAR Grid Results")
    return all_results


# ═══════════════════════════════════════════════════════════════
# K-Fold validation on top N
# ═══════════════════════════════════════════════════════════════

def run_kfold_top(h1_df, results, strategy_type, timeframe, top_n=50):
    print(f"\n{'='*80}")
    print(f"  K-Fold 6-Fold: Top {top_n} {strategy_type} ({timeframe})")
    print(f"{'='*80}")

    ckpt = f"kfold_{strategy_type}_{timeframe}.json"
    existing = load_checkpoint(ckpt)
    if existing:
        print(f"  [Resume] Found checkpoint, skipping", flush=True)
        return existing

    valid = [r for r in results if r.get('sharpe', 0) > 0 and r.get('n', 0) > 10]
    valid.sort(key=lambda x: x['sharpe'], reverse=True)
    candidates = valid[:top_n]

    print(f"  Candidates: {len(candidates)}")

    kfold_results = []
    for ci, cand in enumerate(candidates):
        label = cand['label']
        params = _parse_params(label, strategy_type)
        fold_sharpes = []

        for fname, start, end in KFOLD_FOLDS:
            fold_h1 = h1_df[start:end]
            if len(fold_h1) < 100:
                continue

            if strategy_type == 'kc':
                if timeframe == 'd1':
                    fold_df = fold_h1.resample('D').agg({'Open':'first','High':'max','Low':'min',
                                                          'Close':'last','Volume':'sum'}).dropna()
                else:
                    fold_df = fold_h1.resample('4h').agg({'Open':'first','High':'max','Low':'min',
                                                           'Close':'last','Volume':'sum'}).dropna()
                fold_df = add_kc(fold_df, params['ema'], 14, params['mult'])
                fold_df = fold_df.dropna()
                r = backtest_kc(fold_df, f"{fname}_{label}", adx_thresh=params['adx'],
                                sl_atr=params['sl'], tp_atr=params['tp'],
                                trail_act_atr=params['trail_a'], trail_dist_atr=params['trail_d'],
                                max_hold=params['mh'])
            elif strategy_type == 'st':
                fold_df = add_supertrend(fold_h1, params['period'], params['factor'])
                r = backtest_supertrend(fold_df, f"{fname}_{label}", sl_atr=params['sl'],
                                        tp_atr=params['tp'], trail_act_atr=params['trail_a'],
                                        trail_dist_atr=params['trail_d'], max_hold=params['mh'])
            elif strategy_type == 'psar':
                fold_df = add_psar(fold_h1, params['af_start'], params['af_max'])
                r = backtest_psar(fold_df, f"{fname}_{label}", sl_atr=params['sl'],
                                  tp_atr=params['tp'], trail_act_atr=params['trail_a'],
                                  trail_dist_atr=params['trail_d'], max_hold=params['mh'])
            else:
                continue
            fold_sharpes.append(r['sharpe'])

        passed = len(fold_sharpes) == 6 and all(s > 0 for s in fold_sharpes) and np.mean(fold_sharpes) > 1.5
        kfold_results.append({
            'label': label, 'full_sharpe': cand['sharpe'],
            'kfold_mean': round(np.mean(fold_sharpes), 2) if fold_sharpes else 0,
            'kfold_min': round(min(fold_sharpes), 2) if fold_sharpes else 0,
            'kfold_folds': [round(s, 2) for s in fold_sharpes],
            'passed': passed,
        })

        p_str = "PASS" if passed else "FAIL"
        print(f"  [{ci+1}/{len(candidates)}] {label[:55]:>55} "
              f"Full={cand['sharpe']:.2f} KF={np.mean(fold_sharpes):.2f} {p_str}", flush=True)

    n_passed = sum(1 for k in kfold_results if k['passed'])
    print(f"\n  Passed: {n_passed}/{len(kfold_results)}")

    save_checkpoint(kfold_results, ckpt)
    return kfold_results


def _parse_params(label, strategy_type):
    parts = label.split('_')
    p = {}
    for part in parts:
        if part.startswith('E') and not part.startswith('EMA'):
            try: p['ema'] = int(part[1:])
            except: pass
        elif part.startswith('M') and not part.startswith('MH') and not part.startswith('MX'):
            try: p['mult'] = float(part[1:])
            except: pass
        elif part.startswith('ADX'):
            try: p['adx'] = int(part[3:])
            except: pass
        elif part.startswith('SL'):
            try: p['sl'] = float(part[2:])
            except: pass
        elif part.startswith('TP'):
            try: p['tp'] = float(part[2:])
            except: pass
        elif part.startswith('MH'):
            try: p['mh'] = int(part[2:])
            except: pass
        elif part.startswith('P') and strategy_type == 'st':
            try: p['period'] = int(part[1:])
            except: pass
        elif part.startswith('F') and strategy_type == 'st':
            try: p['factor'] = float(part[1:])
            except: pass
        elif part.startswith('AF') and strategy_type == 'psar':
            try: p['af_start'] = float(part[2:])
            except: pass
        elif part.startswith('MX'):
            try: p['af_max'] = float(part[2:])
            except: pass
        elif '/' in part and part.startswith('T'):
            try:
                a, d = part[1:].split('/')
                p['trail_a'] = float(a); p['trail_d'] = float(d)
            except: pass
    p.setdefault('trail_a', 0.28); p.setdefault('trail_d', 0.06)
    p.setdefault('mh', 20); p.setdefault('sl', 3.5); p.setdefault('tp', 8.0)
    return p


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    t0_total = time.time()

    print("=" * 80)
    print("  R51: Independent Strategy Full Parameter Grid Search")
    print(f"  Spread: ${SPREAD}")
    print(f"  Workers: {MAX_WORKERS}")
    print("=" * 80)

    print("\n  Loading H1 data...")
    from backtest.runner import DataBundle
    data = DataBundle.load_default()
    h1_df = data.h1_df.copy()
    print(f"  H1: {len(h1_df)} bars ({h1_df.index[0]} -> {h1_df.index[-1]})")

    d1_combos = (len(D1_KC_GRID['ema']) * len(D1_KC_GRID['mult']) * len(D1_KC_GRID['adx']) *
                 len(D1_KC_GRID['sl']) * len(D1_KC_GRID['tp']) * len(D1_KC_GRID['mh']) *
                 len(D1_KC_GRID['trail']))
    h4_combos = (len(H4_KC_GRID['ema']) * len(H4_KC_GRID['mult']) * len(H4_KC_GRID['adx']) *
                 len(H4_KC_GRID['sl']) * len(H4_KC_GRID['tp']) * len(H4_KC_GRID['mh']) *
                 len(H4_KC_GRID['trail']))
    st_combos = (len(ST_GRID['period']) * len(ST_GRID['factor']) * len(ST_GRID['sl']) *
                 len(ST_GRID['tp']) * len(ST_GRID['mh']) * len(ST_GRID['trail']))
    psar_combos = (len(PSAR_GRID['af_start']) * len(PSAR_GRID['af_max']) * len(PSAR_GRID['sl']) *
                   len(PSAR_GRID['tp']) * len(PSAR_GRID['mh']) * len(PSAR_GRID['trail']))

    print(f"\n  Grid sizes: D1 KC={d1_combos:,} | H4 KC={h4_combos:,} | "
          f"ST={st_combos:,} | PSAR={psar_combos:,} | Total={d1_combos+h4_combos+st_combos+psar_combos:,}")

    # --- A) D1 Keltner ---
    d1_results = run_kc_grid(h1_df, 'd1', D1_KC_GRID, "d1_kc_grid.json")
    d1_kfold = run_kfold_top(h1_df, d1_results, 'kc', 'd1')

    # --- B) H4 Keltner ---
    h4_results = run_kc_grid(h1_df, 'h4', H4_KC_GRID, "h4_kc_grid.json")
    h4_kfold = run_kfold_top(h1_df, h4_results, 'kc', 'h4')

    # --- C) SuperTrend ---
    st_results = run_st_grid(h1_df, "st_h1_grid.json")
    st_kfold = run_kfold_top(h1_df, st_results, 'st', 'h1')

    # --- D) PSAR ---
    psar_results = run_psar_grid(h1_df, "psar_h1_grid.json")
    psar_kfold = run_kfold_top(h1_df, psar_results, 'psar', 'h1')

    # --- Final Summary ---
    elapsed_total = time.time() - t0_total
    print(f"\n{'='*80}")
    print(f"  R51 COMPLETE — {elapsed_total/3600:.1f}h total")
    print(f"{'='*80}")

    for name, kf in [("D1 KC", d1_kfold), ("H4 KC", h4_kfold),
                      ("SuperTrend", st_kfold), ("PSAR", psar_kfold)]:
        n_pass = sum(1 for k in kf if k.get('passed'))
        n_total = len(kf)
        best = max(kf, key=lambda x: x.get('kfold_mean', 0)) if kf else {}
        print(f"  {name:>12}: {n_pass}/{n_total} passed K-Fold | "
              f"Best: {best.get('label','N/A')[:40]} KF_mean={best.get('kfold_mean',0):.2f}")

    print(f"\n  Results in: {OUTPUT_DIR}")

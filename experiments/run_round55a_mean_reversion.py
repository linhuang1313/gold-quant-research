#!/usr/bin/env python3
"""
Round 55a — Mean-Reversion Strategy Full Parameter Grid Search
===============================================================
Brute-force grid search for 3 mean-reversion alpha sources on XAUUSD H1:
  A1) Bollinger Band Bounce (BB + RSI + ADX filter)
  A2) RSI Mean Reversion (crossback with confirmation bars)
  A3) Session Breakout (session-range breakout)

Each strategy: Layer 1 full grid → Top 50 K-Fold 6-Fold validation

USAGE (server)
--------------
    cd /root/gold-quant-research
    nohup python3 -u experiments/run_round55a_mean_reversion.py \
        > results/round55a_results/stdout.txt 2>&1 &
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

OUTPUT_DIR = Path("results/round55a_results")
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

def _mk(pos, exit_p, exit_time, reason, bar_idx, pnl):
    return {'dir': pos['dir'], 'entry': pos['entry'], 'exit': exit_p,
            'entry_time': pos['time'], 'exit_time': exit_time,
            'pnl': pnl, 'reason': reason, 'bars': bar_idx - pos['bar']}


def compute_atr(df):
    tr = pd.DataFrame({
        'hl': df['High'] - df['Low'],
        'hc': (df['High'] - df['Close'].shift(1)).abs(),
        'lc': (df['Low'] - df['Close'].shift(1)).abs(),
    }).max(axis=1)
    return tr.rolling(14).mean()


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


def compute_rsi(series, period):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


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
# A1) Bollinger Band Bounce Backtest
# ═══════════════════════════════════════════════════════════════

def add_bb_indicators(df, bb_period=20, bb_std=2.0, rsi_period=14):
    df = df.copy()
    df['BB_mid'] = df['Close'].rolling(bb_period).mean()
    rolling_std = df['Close'].rolling(bb_period).std()
    df['BB_upper'] = df['BB_mid'] + bb_std * rolling_std
    df['BB_lower'] = df['BB_mid'] - bb_std * rolling_std
    df['RSI'] = compute_rsi(df['Close'], rsi_period)
    df['ADX'] = compute_adx(df)
    df['ATR'] = compute_atr(df)
    return df


def backtest_bb(df_prepared, label, rsi_ob=70, rsi_os=30, adx_max=25,
                sl_atr=3.5, tp_atr=8.0, trail_act_atr=0.28, trail_dist_atr=0.06,
                max_hold=20, spread=SPREAD, lot=0.03):
    df = df_prepared.dropna(subset=['BB_upper', 'BB_lower', 'RSI', 'ADX', 'ATR'])
    trades = []; pos = None
    close = df['Close'].values; high = df['High'].values; low = df['Low'].values
    bb_up = df['BB_upper'].values; bb_lo = df['BB_lower'].values
    rsi = df['RSI'].values; adx_arr = df['ADX'].values; atr = df['ATR'].values
    times = df.index; n = len(df); last_exit = -999

    for i in range(1, n):
        c = close[i]; h = high[i]; lo_v = low[i]
        cur_atr = atr[i]
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
        cur_adx = adx_arr[i]
        if np.isnan(cur_adx) or cur_adx > adx_max: continue

        prev_c = close[i-1]; cur_rsi = rsi[i]
        if np.isnan(cur_rsi): continue

        # BUY: close crosses below BB_lower AND RSI < oversold
        if prev_c >= bb_lo[i-1] and c < bb_lo[i] and cur_rsi < rsi_os:
            pos = {'dir': 'BUY', 'entry': c + spread/2, 'bar': i, 'time': times[i], 'atr': cur_atr}
        # SELL: close crosses above BB_upper AND RSI > overbought
        elif prev_c <= bb_up[i-1] and c > bb_up[i] and cur_rsi > rsi_ob:
            pos = {'dir': 'SELL', 'entry': c - spread/2, 'bar': i, 'time': times[i], 'atr': cur_atr}

    return calc_stats(trades, label)


# ═══════════════════════════════════════════════════════════════
# A2) RSI Mean Reversion Backtest
# ═══════════════════════════════════════════════════════════════

def add_rsi_indicators(df, rsi_period=14):
    df = df.copy()
    df['RSI'] = compute_rsi(df['Close'], rsi_period)
    df['ATR'] = compute_atr(df)
    return df


def backtest_rsi_mr(df_prepared, label, rsi_os=30, rsi_ob=70, confirm_bars=2,
                    sl_atr=3.5, tp_atr=8.0, trail_act_atr=0.28, trail_dist_atr=0.06,
                    max_hold=20, spread=SPREAD, lot=0.03):
    df = df_prepared.dropna(subset=['RSI', 'ATR'])
    trades = []; pos = None
    close = df['Close'].values; high = df['High'].values; low = df['Low'].values
    rsi = df['RSI'].values; atr = df['ATR'].values
    times = df.index; n = len(df); last_exit = -999

    for i in range(confirm_bars, n):
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
        cur_rsi = rsi[i]
        if np.isnan(cur_rsi): continue

        # BUY: RSI was below oversold for confirm_bars, now crosses back above
        if cur_rsi >= rsi_os:
            all_below = True
            for j in range(1, confirm_bars + 1):
                if rsi[i - j] >= rsi_os:
                    all_below = False
                    break
            if all_below:
                pos = {'dir': 'BUY', 'entry': c + spread/2, 'bar': i, 'time': times[i], 'atr': cur_atr}
                continue

        # SELL: RSI was above overbought for confirm_bars, now crosses back below
        if cur_rsi <= rsi_ob:
            all_above = True
            for j in range(1, confirm_bars + 1):
                if rsi[i - j] <= rsi_ob:
                    all_above = False
                    break
            if all_above:
                pos = {'dir': 'SELL', 'entry': c - spread/2, 'bar': i, 'time': times[i], 'atr': cur_atr}

    return calc_stats(trades, label)


# ═══════════════════════════════════════════════════════════════
# A3) Session Breakout Backtest
# ═══════════════════════════════════════════════════════════════

SESSION_DEFS = {
    "asian":      (0, 7),
    "london":     (8, 11),
    "ny_peak":    (12, 16),
    "late":       (17, 23),
    "peak_12_14": (12, 14),
}


def add_session_indicators(df):
    df = df.copy()
    df['ATR'] = compute_atr(df)
    df['Hour'] = df.index.hour
    return df


def backtest_session(df_prepared, label, session="london", lookback_bars=3,
                     sl_atr=3.0, tp_atr=6.0, trail_act_atr=0.20, trail_dist_atr=0.04,
                     max_hold=10, spread=SPREAD, lot=0.03):
    df = df_prepared.dropna(subset=['ATR'])
    trades = []; pos = None
    close = df['Close'].values; high = df['High'].values; low = df['Low'].values
    atr = df['ATR'].values; hours = df['Hour'].values
    times = df.index; n = len(df); last_exit = -999

    sess_start, sess_end = SESSION_DEFS[session]

    for i in range(lookback_bars, n):
        c = close[i]; h = high[i]; lo_v = low[i]; cur_atr = atr[i]; cur_hour = hours[i]
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

        # Only trigger at session open bar
        if cur_hour != sess_start: continue
        # Check previous bar was NOT in this session (new session start)
        if i > 0 and hours[i-1] == sess_start: continue

        range_high = max(high[i - lookback_bars:i])
        range_low = min(low[i - lookback_bars:i])

        if c > range_high:
            pos = {'dir': 'BUY', 'entry': c + spread/2, 'bar': i, 'time': times[i], 'atr': cur_atr}
        elif c < range_low:
            pos = {'dir': 'SELL', 'entry': c - spread/2, 'bar': i, 'time': times[i], 'atr': cur_atr}

    return calc_stats(trades, label)


# ═══════════════════════════════════════════════════════════════
# Grid definitions
# ═══════════════════════════════════════════════════════════════

BB_GRID = {
    'bb_period': [10, 15, 20, 25, 30, 40],
    'bb_std': [1.5, 2.0, 2.5, 3.0],
    'rsi_period': [7, 14, 21],
    'rsi_ob_os': [(70, 30), (75, 25), (80, 20)],
    'adx_max': [20, 25, 30, 999],
    'sl': [2.0, 3.0, 3.5, 4.5],
    'tp': [4.0, 6.0, 8.0, 12.0],
    'mh': [10, 20, 30, 50],
    'trail': [(0.14, 0.025), (0.20, 0.04), (0.28, 0.06)],
}

RSI_MR_GRID = {
    'rsi_period': [5, 7, 10, 14, 21],
    'thresholds': [(35, 65), (30, 70), (25, 75), (20, 80)],
    'confirm_bars': [1, 2, 3],
    'sl': [2.0, 3.0, 3.5, 4.5],
    'tp': [4.0, 6.0, 8.0, 12.0],
    'mh': [10, 20, 30, 50],
    'trail': [(0.14, 0.025), (0.20, 0.04), (0.28, 0.06)],
}

SESSION_GRID = {
    'sessions': ["london", "ny_peak", "peak_12_14"],
    'lookback_bars': [2, 3, 4, 6],
    'sl': [2.0, 3.0, 4.5],
    'tp': [4.0, 6.0, 8.0],
    'mh': [6, 10, 15, 20],
    'trail': [(0.14, 0.025), (0.20, 0.04), (0.28, 0.06)],
}


# ═══════════════════════════════════════════════════════════════
# Worker functions for multiprocessing
# ═══════════════════════════════════════════════════════════════

def _worker_bb(args):
    df_path, label, rsi_ob, rsi_os, adx_max, sl, tp, mh, trail_a, trail_d = args
    df = pd.read_pickle(df_path)
    return backtest_bb(df, label, rsi_ob=rsi_ob, rsi_os=rsi_os, adx_max=adx_max,
                       sl_atr=sl, tp_atr=tp, trail_act_atr=trail_a,
                       trail_dist_atr=trail_d, max_hold=mh)


def _worker_rsi(args):
    df_path, label, rsi_os, rsi_ob, confirm_bars, sl, tp, mh, trail_a, trail_d = args
    df = pd.read_pickle(df_path)
    return backtest_rsi_mr(df, label, rsi_os=rsi_os, rsi_ob=rsi_ob,
                           confirm_bars=confirm_bars, sl_atr=sl, tp_atr=tp,
                           trail_act_atr=trail_a, trail_dist_atr=trail_d, max_hold=mh)


def _worker_session(args):
    df_path, label, session, lookback_bars, sl, tp, mh, trail_a, trail_d = args
    df = pd.read_pickle(df_path)
    return backtest_session(df, label, session=session, lookback_bars=lookback_bars,
                            sl_atr=sl, tp_atr=tp, trail_act_atr=trail_a,
                            trail_dist_atr=trail_d, max_hold=mh)


# ═══════════════════════════════════════════════════════════════
# Strategy runners
# ═══════════════════════════════════════════════════════════════

def run_bb_grid(h1_df, checkpoint_name="bb_bounce_grid.json"):
    print(f"\n{'='*80}")
    print(f"  A1: Bollinger Band Bounce — Full Parameter Grid")
    print(f"{'='*80}")

    existing = load_checkpoint(checkpoint_name)
    if existing:
        print(f"  [Resume] Found {len(existing)} results, skipping", flush=True)
        return existing

    indicator_combos = list(product(BB_GRID['bb_period'], BB_GRID['bb_std'], BB_GRID['rsi_period']))
    print(f"  Pre-computing {len(indicator_combos)} BB indicator sets...", flush=True)
    precomp = {}
    for idx, (bb_p, bb_s, rsi_p) in enumerate(indicator_combos):
        key = f"BBP{bb_p}_BBS{bb_s}_RSI{rsi_p}"
        tmp = OUTPUT_DIR / f"_tmp_BB_{key}.pkl"
        df_bb = add_bb_indicators(h1_df.copy(), bb_p, bb_s, rsi_p)
        df_bb.to_pickle(tmp)
        precomp[key] = str(tmp)
        if (idx + 1) % 10 == 0 or (idx + 1) == len(indicator_combos):
            print(f"    [{idx+1}/{len(indicator_combos)}] done", flush=True)
    print(f"  Pre-computation done!", flush=True)

    tasks = []
    for bb_p, bb_s, rsi_p in product(BB_GRID['bb_period'], BB_GRID['bb_std'], BB_GRID['rsi_period']):
        key = f"BBP{bb_p}_BBS{bb_s}_RSI{rsi_p}"
        for (rsi_ob, rsi_os), adx_max, sl, tp, mh in product(
            BB_GRID['rsi_ob_os'], BB_GRID['adx_max'], BB_GRID['sl'], BB_GRID['tp'], BB_GRID['mh']
        ):
            for trail_a, trail_d in BB_GRID['trail']:
                label = (f"BB_P{bb_p}_S{bb_s}_RSI{rsi_p}_OB{rsi_ob}_OS{rsi_os}_ADX{adx_max}"
                         f"_SL{sl}_TP{tp}_MH{mh}_T{trail_a}/{trail_d}")
                tasks.append((precomp[key], label, rsi_ob, rsi_os, adx_max,
                              sl, tp, mh, trail_a, trail_d))

    total = len(tasks)
    print(f"  Total combos: {total:,}", flush=True)
    print(f"  Workers: {MAX_WORKERS}", flush=True)

    t0 = time.time()
    all_results = []
    batch_size = max(100, total // 20)

    with mp.Pool(MAX_WORKERS) as pool:
        for i, result in enumerate(pool.imap_unordered(_worker_bb, tasks, chunksize=8)):
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
                  "R55a A1 BB Bounce Grid Results")
    return all_results


def run_rsi_mr_grid(h1_df, checkpoint_name="rsi_mr_grid.json"):
    print(f"\n{'='*80}")
    print(f"  A2: RSI Mean Reversion — Full Parameter Grid")
    print(f"{'='*80}")

    existing = load_checkpoint(checkpoint_name)
    if existing:
        print(f"  [Resume] Found {len(existing)} results, skipping", flush=True)
        return existing

    indicator_combos = list(RSI_MR_GRID['rsi_period'])
    print(f"  Pre-computing {len(indicator_combos)} RSI indicator sets...", flush=True)
    precomp = {}
    for idx, rsi_p in enumerate(indicator_combos):
        key = f"RSI{rsi_p}"
        tmp = OUTPUT_DIR / f"_tmp_RSI_{key}.pkl"
        df_rsi = add_rsi_indicators(h1_df.copy(), rsi_p)
        df_rsi.to_pickle(tmp)
        precomp[key] = str(tmp)
        print(f"    [{idx+1}/{len(indicator_combos)}] {key} done", flush=True)
    print(f"  Pre-computation done!", flush=True)

    tasks = []
    for rsi_p in RSI_MR_GRID['rsi_period']:
        key = f"RSI{rsi_p}"
        for (os_t, ob_t), cb, sl, tp, mh in product(
            RSI_MR_GRID['thresholds'], RSI_MR_GRID['confirm_bars'],
            RSI_MR_GRID['sl'], RSI_MR_GRID['tp'], RSI_MR_GRID['mh']
        ):
            for trail_a, trail_d in RSI_MR_GRID['trail']:
                label = (f"RSIMR_P{rsi_p}_OS{os_t}_OB{ob_t}_CB{cb}"
                         f"_SL{sl}_TP{tp}_MH{mh}_T{trail_a}/{trail_d}")
                tasks.append((precomp[key], label, os_t, ob_t, cb,
                              sl, tp, mh, trail_a, trail_d))

    total = len(tasks)
    print(f"  Total combos: {total:,}", flush=True)
    print(f"  Workers: {MAX_WORKERS}", flush=True)

    t0 = time.time()
    all_results = []
    batch_size = max(100, total // 20)

    with mp.Pool(MAX_WORKERS) as pool:
        for i, result in enumerate(pool.imap_unordered(_worker_rsi, tasks, chunksize=8)):
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
                  "R55a A2 RSI Mean Reversion Grid Results")
    return all_results


def run_session_grid(h1_df, checkpoint_name="session_bo_grid.json"):
    print(f"\n{'='*80}")
    print(f"  A3: Session Breakout — Full Parameter Grid")
    print(f"{'='*80}")

    existing = load_checkpoint(checkpoint_name)
    if existing:
        print(f"  [Resume] Found {len(existing)} results, skipping", flush=True)
        return existing

    print(f"  Pre-computing session indicators...", flush=True)
    tmp = OUTPUT_DIR / "_tmp_SESSION.pkl"
    df_sess = add_session_indicators(h1_df.copy())
    df_sess.to_pickle(tmp)
    precomp_path = str(tmp)
    print(f"  Pre-computation done!", flush=True)

    tasks = []
    for sess, lb, sl, tp, mh in product(
        SESSION_GRID['sessions'], SESSION_GRID['lookback_bars'],
        SESSION_GRID['sl'], SESSION_GRID['tp'], SESSION_GRID['mh']
    ):
        for trail_a, trail_d in SESSION_GRID['trail']:
            label = (f"SESS_{sess}_LB{lb}_SL{sl}_TP{tp}_MH{mh}_T{trail_a}/{trail_d}")
            tasks.append((precomp_path, label, sess, lb, sl, tp, mh, trail_a, trail_d))

    total = len(tasks)
    print(f"  Total combos: {total:,}", flush=True)
    print(f"  Workers: {MAX_WORKERS}", flush=True)

    t0 = time.time()
    all_results = []
    batch_size = max(100, total // 20)

    with mp.Pool(MAX_WORKERS) as pool:
        for i, result in enumerate(pool.imap_unordered(_worker_session, tasks, chunksize=8)):
            all_results.append(result)
            if (i + 1) % batch_size == 0 or (i + 1) == total:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (total - i - 1) / rate if rate > 0 else 0
                print(f"    {i+1}/{total} ({(i+1)/total*100:.0f}%) | "
                      f"{elapsed/60:.1f}min | ETA {eta/60:.1f}min", flush=True)
                save_checkpoint(all_results, checkpoint_name)

    Path(precomp_path).unlink(missing_ok=True)

    elapsed = time.time() - t0
    valid = [r for r in all_results if r.get('sharpe', 0) > 0]
    print(f"\n  Done: {len(valid)} positive / {len(all_results)} total in {elapsed/60:.1f}min")

    save_checkpoint(all_results, checkpoint_name)
    write_ranking(all_results, checkpoint_name.replace('.json', '_ranking.txt'),
                  "R55a A3 Session Breakout Grid Results")
    return all_results


# ═══════════════════════════════════════════════════════════════
# K-Fold validation on top N
# ═══════════════════════════════════════════════════════════════

def run_kfold_top(h1_df, results, strategy_type, top_n=50):
    print(f"\n{'='*80}")
    print(f"  K-Fold 6-Fold: Top {top_n} {strategy_type}")
    print(f"{'='*80}")

    ckpt = f"kfold_{strategy_type}.json"
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

            if strategy_type == 'bb':
                fold_df = add_bb_indicators(fold_h1, params['bb_period'],
                                            params['bb_std'], params['rsi_period'])
                r = backtest_bb(fold_df, f"{fname}_{label}",
                                rsi_ob=params['rsi_ob'], rsi_os=params['rsi_os'],
                                adx_max=params['adx_max'], sl_atr=params['sl'],
                                tp_atr=params['tp'], trail_act_atr=params['trail_a'],
                                trail_dist_atr=params['trail_d'], max_hold=params['mh'])
            elif strategy_type == 'rsi_mr':
                fold_df = add_rsi_indicators(fold_h1, params['rsi_period'])
                r = backtest_rsi_mr(fold_df, f"{fname}_{label}",
                                    rsi_os=params['rsi_os'], rsi_ob=params['rsi_ob'],
                                    confirm_bars=params['confirm_bars'],
                                    sl_atr=params['sl'], tp_atr=params['tp'],
                                    trail_act_atr=params['trail_a'],
                                    trail_dist_atr=params['trail_d'], max_hold=params['mh'])
            elif strategy_type == 'session':
                fold_df = add_session_indicators(fold_h1)
                r = backtest_session(fold_df, f"{fname}_{label}",
                                     session=params['session'],
                                     lookback_bars=params['lookback_bars'],
                                     sl_atr=params['sl'], tp_atr=params['tp'],
                                     trail_act_atr=params['trail_a'],
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

    if strategy_type == 'bb':
        for part in parts:
            if part.startswith('P') and 'RSI' not in part:
                try: p['bb_period'] = int(part[1:])
                except: pass
            elif part.startswith('S') and part[1:].replace('.', '', 1).isdigit():
                try: p['bb_std'] = float(part[1:])
                except: pass
            elif part.startswith('RSI'):
                try: p['rsi_period'] = int(part[3:])
                except: pass
            elif part.startswith('OB'):
                try: p['rsi_ob'] = int(part[2:])
                except: pass
            elif part.startswith('OS'):
                try: p['rsi_os'] = int(part[2:])
                except: pass
            elif part.startswith('ADX'):
                try: p['adx_max'] = int(part[3:])
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
            elif '/' in part and part.startswith('T'):
                try:
                    a, d = part[1:].split('/')
                    p['trail_a'] = float(a); p['trail_d'] = float(d)
                except: pass
        p.setdefault('bb_period', 20); p.setdefault('bb_std', 2.0)
        p.setdefault('rsi_period', 14); p.setdefault('rsi_ob', 70); p.setdefault('rsi_os', 30)
        p.setdefault('adx_max', 25)

    elif strategy_type == 'rsi_mr':
        for part in parts:
            if part.startswith('P') and part[1:].isdigit():
                try: p['rsi_period'] = int(part[1:])
                except: pass
            elif part.startswith('OS'):
                try: p['rsi_os'] = int(part[2:])
                except: pass
            elif part.startswith('OB'):
                try: p['rsi_ob'] = int(part[2:])
                except: pass
            elif part.startswith('CB'):
                try: p['confirm_bars'] = int(part[2:])
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
            elif '/' in part and part.startswith('T'):
                try:
                    a, d = part[1:].split('/')
                    p['trail_a'] = float(a); p['trail_d'] = float(d)
                except: pass
        p.setdefault('rsi_period', 14); p.setdefault('rsi_os', 30); p.setdefault('rsi_ob', 70)
        p.setdefault('confirm_bars', 2)

    elif strategy_type == 'session':
        # Label format: SESS_london_LB3_SL3.0_TP6.0_MH10_T0.2/0.04
        # The session name is between SESS_ and _LB
        full = label
        if full.startswith('SESS_'):
            rest = full[5:]
            # Find session name: everything before _LB
            lb_idx = rest.find('_LB')
            if lb_idx >= 0:
                p['session'] = rest[:lb_idx]
                rest = rest[lb_idx:]

        for part in parts:
            if part.startswith('LB'):
                try: p['lookback_bars'] = int(part[2:])
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
            elif '/' in part and part.startswith('T'):
                try:
                    a, d = part[1:].split('/')
                    p['trail_a'] = float(a); p['trail_d'] = float(d)
                except: pass
        p.setdefault('session', 'london'); p.setdefault('lookback_bars', 3)

    p.setdefault('trail_a', 0.28); p.setdefault('trail_d', 0.06)
    p.setdefault('mh', 20); p.setdefault('sl', 3.5); p.setdefault('tp', 8.0)
    return p


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    t0_total = time.time()

    print("=" * 80)
    print("  R55a: Mean-Reversion Strategy Full Parameter Grid Search")
    print(f"  Spread: ${SPREAD}")
    print(f"  Workers: {MAX_WORKERS}")
    print("=" * 80)

    print("\n  Loading H1 data...")
    from backtest.runner import DataBundle
    data = DataBundle.load_default()
    h1_df = data.h1_df.copy()
    print(f"  H1: {len(h1_df)} bars ({h1_df.index[0]} -> {h1_df.index[-1]})")

    bb_combos = (len(BB_GRID['bb_period']) * len(BB_GRID['bb_std']) * len(BB_GRID['rsi_period']) *
                 len(BB_GRID['rsi_ob_os']) * len(BB_GRID['adx_max']) *
                 len(BB_GRID['sl']) * len(BB_GRID['tp']) * len(BB_GRID['mh']) *
                 len(BB_GRID['trail']))
    rsi_combos = (len(RSI_MR_GRID['rsi_period']) * len(RSI_MR_GRID['thresholds']) *
                  len(RSI_MR_GRID['confirm_bars']) *
                  len(RSI_MR_GRID['sl']) * len(RSI_MR_GRID['tp']) * len(RSI_MR_GRID['mh']) *
                  len(RSI_MR_GRID['trail']))
    sess_combos = (len(SESSION_GRID['sessions']) * len(SESSION_GRID['lookback_bars']) *
                   len(SESSION_GRID['sl']) * len(SESSION_GRID['tp']) * len(SESSION_GRID['mh']) *
                   len(SESSION_GRID['trail']))

    print(f"\n  Grid sizes: BB={bb_combos:,} | RSI_MR={rsi_combos:,} | "
          f"Session={sess_combos:,} | Total={bb_combos+rsi_combos+sess_combos:,}")

    # --- A1) Bollinger Band Bounce ---
    bb_results = run_bb_grid(h1_df, "bb_bounce_grid.json")
    bb_kfold = run_kfold_top(h1_df, bb_results, 'bb')

    # --- A2) RSI Mean Reversion ---
    rsi_results = run_rsi_mr_grid(h1_df, "rsi_mr_grid.json")
    rsi_kfold = run_kfold_top(h1_df, rsi_results, 'rsi_mr')

    # --- A3) Session Breakout ---
    sess_results = run_session_grid(h1_df, "session_bo_grid.json")
    sess_kfold = run_kfold_top(h1_df, sess_results, 'session')

    # --- Final Summary ---
    elapsed_total = time.time() - t0_total
    print(f"\n{'='*80}")
    print(f"  R55a COMPLETE — {elapsed_total/3600:.1f}h total")
    print(f"{'='*80}")

    for name, kf in [("BB Bounce", bb_kfold), ("RSI MR", rsi_kfold),
                      ("Session BO", sess_kfold)]:
        n_pass = sum(1 for k in kf if k.get('passed'))
        n_total = len(kf)
        best = max(kf, key=lambda x: x.get('kfold_mean', 0)) if kf else {}
        print(f"  {name:>12}: {n_pass}/{n_total} passed K-Fold | "
              f"Best: {best.get('label','N/A')[:40]} KF_mean={best.get('kfold_mean',0):.2f}")

    print(f"\n  Results in: {OUTPUT_DIR}")

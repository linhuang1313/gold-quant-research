#!/usr/bin/env python3
"""
R60 — Mean-Reversion Strategy Validation + Correlation with Keltner
====================================================================
1. Load R55a K-Fold results (BB, RSI MR, Session BO) from JSON.
2. Filter strategies that passed K-Fold (positive Sharpe in >= 4/6 folds).
3. For each passing strategy, run with Cap$37 + 0.05 lot to get daily PnL.
4. Run Keltner baseline to get its daily PnL series.
5. Compute Pearson/Spearman correlation between each MR strategy and Keltner.
6. Test portfolio combinations at various lot ratios.
7. Report which combination gives best Sharpe improvement and MaxDD reduction.

Output → results/r60_mean_rev_validate/
"""
import sys, os, io, time, json, traceback
import multiprocessing as mp
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict
from itertools import product
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

OUTPUT_DIR = "results/r60_mean_rev_validate"
MAX_WORKERS = 6
FIXED_LOT = 0.05
SPREAD = 0.30

FOLDS = [
    ("Fold1", "2015-01-01", "2017-01-01"),
    ("Fold2", "2017-01-01", "2019-01-01"),
    ("Fold3", "2019-01-01", "2021-01-01"),
    ("Fold4", "2021-01-01", "2023-01-01"),
    ("Fold5", "2023-01-01", "2025-01-01"),
    ("Fold6", "2025-01-01", "2026-05-01"),
]

R55A_DIR = Path("results/round55a_results")

KC_LOT_GRID = [0.03, 0.04, 0.05]
MR_LOT_GRID = [0.01, 0.02, 0.03]


def fmt(v):
    return f"${v:>10,.0f}" if v >= 0 else f"-${abs(v):>9,.0f}"


def get_base():
    from backtest.runner import LIVE_PARITY_KWARGS
    return {**LIVE_PARITY_KWARGS, 'min_lot_size': FIXED_LOT, 'max_lot_size': FIXED_LOT,
            'maxloss_cap': 37}


def _run_one(args):
    label, kw, spread, start, end = args
    from backtest.runner import DataBundle, run_variant
    data = DataBundle.load_custom()
    if start and end:
        data = data.slice(start, end)
    s = run_variant(data, label, verbose=False, spread_cost=spread, **kw)
    return {
        'label': label, 'n': s['n'], 'sharpe': s['sharpe'],
        'total_pnl': s['total_pnl'], 'win_rate': s['win_rate'],
        'max_dd': s['max_dd'],
    }


def _run_one_trades(args):
    label, kw, spread, start, end = args
    from backtest.runner import DataBundle, run_variant
    data = DataBundle.load_custom()
    if start and end:
        data = data.slice(start, end)
    s = run_variant(data, label, verbose=False, spread_cost=spread, **kw)
    trades = s.get('_trades', [])
    trade_list = []
    for t in trades:
        trade_list.append({
            'entry_time': str(t.entry_time)[:19],
            'exit_time': str(t.exit_time)[:19],
            'direction': t.direction,
            'pnl': round(t.pnl, 2),
            'exit_reason': t.exit_reason or '',
            'bars_held': t.bars_held,
            'lots': t.lots,
        })
    return {'label': label, 'trades': trade_list, 'n': s['n'], 'sharpe': s['sharpe'],
            'total_pnl': s['total_pnl'], 'win_rate': s['win_rate'], 'max_dd': s['max_dd']}


def run_pool(tasks, func=_run_one):
    n_workers = min(MAX_WORKERS, len(tasks))
    with mp.Pool(n_workers) as pool:
        return pool.map(func, tasks)


# ═══════════════════════════════════════════════════════════════
# R55a self-contained backtest functions (copied with trade-list output)
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


def _backtest_bb_trades(df, label, rsi_ob=70, rsi_os=30, adx_max=25,
                        sl_atr=3.5, tp_atr=8.0, trail_act_atr=0.28, trail_dist_atr=0.06,
                        max_hold=20, spread=SPREAD, lot=0.05,
                        bb_period=20, bb_std=2.0, rsi_period=14):
    df = df.copy()
    df['BB_mid'] = df['Close'].rolling(bb_period).mean()
    rolling_std = df['Close'].rolling(bb_period).std()
    df['BB_upper'] = df['BB_mid'] + bb_std * rolling_std
    df['BB_lower'] = df['BB_mid'] - bb_std * rolling_std
    df['RSI'] = compute_rsi(df['Close'], rsi_period)
    df['ADX'] = compute_adx(df)
    df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['BB_upper', 'BB_lower', 'RSI', 'ADX', 'ATR'])

    trades = []; pos = None
    close = df['Close'].values; high = df['High'].values; low = df['Low'].values
    bb_up = df['BB_upper'].values; bb_lo = df['BB_lower'].values
    rsi = df['RSI'].values; adx_arr = df['ADX'].values; atr = df['ATR'].values
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
        cur_adx = adx_arr[i]
        if np.isnan(cur_adx) or cur_adx > adx_max: continue
        prev_c = close[i-1]; cur_rsi = rsi[i]
        if np.isnan(cur_rsi): continue
        if prev_c >= bb_lo[i-1] and c < bb_lo[i] and cur_rsi < rsi_os:
            pos = {'dir': 'BUY', 'entry': c + spread/2, 'bar': i, 'time': times[i], 'atr': cur_atr}
        elif prev_c <= bb_up[i-1] and c > bb_up[i] and cur_rsi > rsi_ob:
            pos = {'dir': 'SELL', 'entry': c - spread/2, 'bar': i, 'time': times[i], 'atr': cur_atr}

    return trades


def _backtest_rsi_mr_trades(df, label, rsi_os=30, rsi_ob=70, confirm_bars=2,
                            sl_atr=3.5, tp_atr=8.0, trail_act_atr=0.28, trail_dist_atr=0.06,
                            max_hold=20, spread=SPREAD, lot=0.05, rsi_period=14):
    df = df.copy()
    df['RSI'] = compute_rsi(df['Close'], rsi_period)
    df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['RSI', 'ATR'])

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
        if cur_rsi >= rsi_os:
            all_below = all(rsi[i - j] < rsi_os for j in range(1, confirm_bars + 1))
            if all_below:
                pos = {'dir': 'BUY', 'entry': c + spread/2, 'bar': i, 'time': times[i], 'atr': cur_atr}
                continue
        if cur_rsi <= rsi_ob:
            all_above = all(rsi[i - j] > rsi_ob for j in range(1, confirm_bars + 1))
            if all_above:
                pos = {'dir': 'SELL', 'entry': c - spread/2, 'bar': i, 'time': times[i], 'atr': cur_atr}

    return trades


SESSION_DEFS = {
    "asian":      (0, 7),
    "london":     (8, 11),
    "ny_peak":    (12, 16),
    "late":       (17, 23),
    "peak_12_14": (12, 14),
}


def _backtest_session_trades(df, label, session="london", lookback_bars=3,
                             sl_atr=3.0, tp_atr=6.0, trail_act_atr=0.20, trail_dist_atr=0.04,
                             max_hold=10, spread=SPREAD, lot=0.05):
    df = df.copy()
    df['ATR'] = compute_atr(df)
    df['Hour'] = df.index.hour
    df = df.dropna(subset=['ATR'])

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
        if cur_hour != sess_start: continue
        if i > 0 and hours[i-1] == sess_start: continue
        range_high = max(high[i - lookback_bars:i])
        range_low = min(low[i - lookback_bars:i])
        if c > range_high:
            pos = {'dir': 'BUY', 'entry': c + spread/2, 'bar': i, 'time': times[i], 'atr': cur_atr}
        elif c < range_low:
            pos = {'dir': 'SELL', 'entry': c - spread/2, 'bar': i, 'time': times[i], 'atr': cur_atr}

    return trades


# ═══════════════════════════════════════════════════════════════
# R55a label parser (reused from R55a)
# ═══════════════════════════════════════════════════════════════

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
        full = label
        if full.startswith('SESS_'):
            rest = full[5:]
            lb_idx = rest.find('_LB')
            if lb_idx >= 0:
                p['session'] = rest[:lb_idx]
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
# Helpers
# ═══════════════════════════════════════════════════════════════

def _trades_to_daily_pnl(trades):
    """Convert trade list (dicts with exit_time, pnl) to {date_str: sum_pnl}."""
    daily = defaultdict(float)
    for t in trades:
        d = str(t['exit_time'])[:10]
        daily[d] += t['pnl']
    return dict(daily)


def _sharpe_from_daily(daily_dict):
    if not daily_dict:
        return 0.0
    vals = np.array(list(daily_dict.values()))
    if len(vals) < 2 or vals.std() == 0:
        return 0.0
    return float(vals.mean() / vals.std() * np.sqrt(252))


def _max_dd_from_daily(daily_dict, capital=2000):
    if not daily_dict:
        return 0.0
    dates = sorted(daily_dict.keys())
    eq = capital
    peak = eq
    max_dd = 0
    for d in dates:
        eq += daily_dict[d]
        if eq > peak:
            peak = eq
        dd = peak - eq
        if dd > max_dd:
            max_dd = dd
    return max_dd


def _compute_correlation(daily_a, daily_b):
    """Pearson and Spearman on overlapping dates."""
    from scipy import stats as sp_stats
    common = sorted(set(daily_a.keys()) & set(daily_b.keys()))
    if len(common) < 10:
        return {'pearson': 0.0, 'spearman': 0.0, 'n_overlap': len(common)}
    a = np.array([daily_a[d] for d in common])
    b = np.array([daily_b[d] for d in common])
    pearson_r = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0
    spearman_r = float(sp_stats.spearmanr(a, b).statistic) if len(common) >= 3 else 0.0
    return {'pearson': round(pearson_r, 4), 'spearman': round(spearman_r, 4),
            'n_overlap': len(common)}


def _portfolio_sharpe(daily_pnl_list, lot_ratios):
    """Combine multiple daily PnL series at given lot ratios → portfolio Sharpe."""
    all_dates = set()
    for d in daily_pnl_list:
        all_dates.update(d.keys())
    all_dates = sorted(all_dates)
    combined = defaultdict(float)
    for daily, ratio in zip(daily_pnl_list, lot_ratios):
        for d in all_dates:
            combined[d] += daily.get(d, 0) * ratio
    return _sharpe_from_daily(dict(combined)), _max_dd_from_daily(dict(combined))


# ═══════════════════════════════════════════════════════════════
# Phase 1: Load R55a K-Fold results
# ═══════════════════════════════════════════════════════════════

def load_r55a_results():
    """Load R55a kfold JSON files and filter strategies that passed."""
    files = {
        'bb': 'kfold_bb.json',
        'rsi_mr': 'kfold_rsi_mr.json',
        'session': 'kfold_session.json',
    }

    passed = []
    summary_lines = []

    for stype, fname in files.items():
        path = R55A_DIR / fname
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping {stype}")
            summary_lines.append(f"  {stype}: FILE NOT FOUND ({path})")
            continue
        try:
            with open(path, 'r') as f:
                kfold_data = json.load(f)
        except Exception as e:
            print(f"  ERROR loading {path}: {e}")
            summary_lines.append(f"  {stype}: LOAD ERROR ({e})")
            continue

        total = len(kfold_data)
        n_passed = 0
        for entry in kfold_data:
            fold_sharpes = entry.get('kfold_folds', [])
            pos_folds = sum(1 for s in fold_sharpes if s > 0)
            if pos_folds >= 4:
                passed.append({
                    'label': entry['label'],
                    'type': stype,
                    'full_sharpe': entry.get('full_sharpe', 0),
                    'kfold_mean': entry.get('kfold_mean', 0),
                    'kfold_folds': fold_sharpes,
                    'pos_folds': pos_folds,
                })
                n_passed += 1

        summary_lines.append(f"  {stype}: {n_passed}/{total} passed (>= 4/6 positive folds)")

    return passed, summary_lines


# ═══════════════════════════════════════════════════════════════
# Phase 2: Run strategies to get daily PnL
# ═══════════════════════════════════════════════════════════════

def run_strategy_daily_pnl(h1_df, strategy_type, params, lot, spread):
    """Run one R55a strategy type, return trade list."""
    if strategy_type == 'bb':
        return _backtest_bb_trades(h1_df, params.get('label', ''),
                                   rsi_ob=params['rsi_ob'], rsi_os=params['rsi_os'],
                                   adx_max=params['adx_max'],
                                   sl_atr=params['sl'], tp_atr=params['tp'],
                                   trail_act_atr=params['trail_a'],
                                   trail_dist_atr=params['trail_d'],
                                   max_hold=params['mh'], spread=spread, lot=lot,
                                   bb_period=params['bb_period'],
                                   bb_std=params['bb_std'],
                                   rsi_period=params['rsi_period'])
    elif strategy_type == 'rsi_mr':
        return _backtest_rsi_mr_trades(h1_df, params.get('label', ''),
                                       rsi_os=params['rsi_os'], rsi_ob=params['rsi_ob'],
                                       confirm_bars=params['confirm_bars'],
                                       sl_atr=params['sl'], tp_atr=params['tp'],
                                       trail_act_atr=params['trail_a'],
                                       trail_dist_atr=params['trail_d'],
                                       max_hold=params['mh'], spread=spread, lot=lot,
                                       rsi_period=params['rsi_period'])
    elif strategy_type == 'session':
        return _backtest_session_trades(h1_df, params.get('label', ''),
                                        session=params['session'],
                                        lookback_bars=params['lookback_bars'],
                                        sl_atr=params['sl'], tp_atr=params['tp'],
                                        trail_act_atr=params['trail_a'],
                                        trail_dist_atr=params['trail_d'],
                                        max_hold=params['mh'], spread=spread, lot=lot)
    return []


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t0 = time.time()
    print(f"\n{'=' * 70}")
    print(f"R60: Mean-Reversion Strategy Validation + Correlation with Keltner")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Lot={FIXED_LOT}  Spread=${SPREAD}  Cap=$37")
    print(f"{'=' * 70}")

    # ── Step 1: Load R55a results ──
    print("\n" + "=" * 70)
    print("Step 1: Load R55a K-Fold Results")
    print("=" * 70)

    passed_strategies, summary_lines = load_r55a_results()
    for line in summary_lines:
        print(line)
    print(f"\n  Total passing strategies: {len(passed_strategies)}")

    with open(f"{OUTPUT_DIR}/r55a_kfold_summary.txt", 'w', encoding='utf-8') as f:
        f.write("R60 Step 1: R55a K-Fold Pass Summary\n")
        f.write("=" * 80 + "\n\n")
        for line in summary_lines:
            f.write(line + "\n")
        f.write(f"\nTotal passing: {len(passed_strategies)}\n\n")
        if passed_strategies:
            header = f"{'Type':<10} {'Label':<55} {'FullSh':>7} {'KF_Mean':>8} {'Pos':>4}"
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")
            for s in sorted(passed_strategies, key=lambda x: x['kfold_mean'], reverse=True):
                f.write(f"{s['type']:<10} {s['label'][:55]:<55} "
                        f"{s['full_sharpe']:>7.2f} {s['kfold_mean']:>8.2f} "
                        f"{s['pos_folds']:>3}/6\n")

    if not passed_strategies:
        print("\n  No strategies passed K-Fold. Skipping correlation/portfolio analysis.")
        with open(f"{OUTPUT_DIR}/r60_summary.txt", 'w', encoding='utf-8') as f:
            f.write("R60 Summary: No R55a strategies passed K-Fold. Nothing to validate.\n")
        return

    # ── Step 2: Load H1 data, run Keltner baseline + MR strategies ──
    print("\n" + "=" * 70)
    print("Step 2: Run Keltner Baseline + MR Strategies for Daily PnL")
    print("=" * 70)

    from backtest.runner import DataBundle
    data = DataBundle.load_default()
    h1_df = data.h1_df.copy()
    print(f"  H1 data: {len(h1_df)} bars ({h1_df.index[0]} → {h1_df.index[-1]})")

    # Keltner baseline via engine
    print("  Running Keltner baseline...")
    base = get_base()
    kc_result = run_pool([("KC_Baseline", base, SPREAD, None, None)], func=_run_one_trades)[0]
    kc_daily = _trades_to_daily_pnl(kc_result['trades'])
    kc_sharpe = _sharpe_from_daily(kc_daily)
    print(f"  Keltner: N={kc_result['n']}  Sharpe={kc_result['sharpe']:.2f}  "
          f"PnL={fmt(kc_result['total_pnl'])}")

    # Run each passing MR strategy
    print(f"\n  Running {len(passed_strategies)} MR strategies...")
    mr_results = []
    for i, strat in enumerate(passed_strategies):
        params = _parse_params(strat['label'], strat['type'])
        params['label'] = strat['label']
        try:
            trades = run_strategy_daily_pnl(h1_df, strat['type'], params,
                                            lot=FIXED_LOT, spread=SPREAD)
            daily = _trades_to_daily_pnl(trades)
            sharpe = _sharpe_from_daily(daily)
            total_pnl = sum(t['pnl'] for t in trades)
            wins = sum(1 for t in trades if t['pnl'] > 0)
            wr = (wins / len(trades) * 100) if trades else 0
            max_dd = _max_dd_from_daily(daily)

            mr_results.append({
                'label': strat['label'], 'type': strat['type'],
                'n': len(trades), 'sharpe': round(sharpe, 2),
                'total_pnl': round(total_pnl, 2), 'win_rate': round(wr, 1),
                'max_dd': round(max_dd, 2),
                'kfold_mean': strat['kfold_mean'],
                '_daily': daily,
            })
            print(f"    [{i+1}/{len(passed_strategies)}] {strat['type']:<8} "
                  f"N={len(trades):>4}  Sharpe={sharpe:>6.2f}  {strat['label'][:40]}")
        except Exception as e:
            print(f"    [{i+1}/{len(passed_strategies)}] ERROR: {strat['label'][:40]} — {e}")
            traceback.print_exc()

    if not mr_results:
        print("\n  No MR strategies produced trades. Aborting.")
        with open(f"{OUTPUT_DIR}/r60_summary.txt", 'w', encoding='utf-8') as f:
            f.write("R60 Summary: No MR strategies produced trades.\n")
        return

    # ── Step 3: Correlation analysis ──
    print("\n" + "=" * 70)
    print("Step 3: Correlation Analysis (MR vs Keltner)")
    print("=" * 70)

    corr_results = []
    for mr in mr_results:
        corr = _compute_correlation(kc_daily, mr['_daily'])
        corr_results.append({
            'label': mr['label'], 'type': mr['type'],
            'mr_sharpe': mr['sharpe'], 'mr_n': mr['n'],
            **corr,
        })
        print(f"  {mr['type']:<8} Pearson={corr['pearson']:>+.4f}  "
              f"Spearman={corr['spearman']:>+.4f}  Overlap={corr['n_overlap']}  "
              f"{mr['label'][:35]}")

    corr_results.sort(key=lambda x: x['pearson'])

    with open(f"{OUTPUT_DIR}/correlation_matrix.txt", 'w', encoding='utf-8') as f:
        f.write("R60 Step 3: MR vs Keltner Daily PnL Correlation\n")
        f.write(f"Keltner Baseline: N={kc_result['n']}  Sharpe={kc_sharpe:.4f}\n")
        f.write("=" * 100 + "\n\n")
        f.write("Sorted by Pearson (lowest = most diversifying):\n\n")
        header = (f"{'Type':<10} {'Label':<45} {'Pearson':>8} {'Spearman':>9} "
                  f"{'Overlap':>8} {'MR_Sh':>7}")
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for c in corr_results:
            f.write(f"{c['type']:<10} {c['label'][:45]:<45} "
                    f"{c['pearson']:>+8.4f} {c['spearman']:>+9.4f} "
                    f"{c['n_overlap']:>8} {c['mr_sharpe']:>7.2f}\n")
        f.write(f"\nMean Pearson:  {np.mean([c['pearson'] for c in corr_results]):.4f}\n")
        f.write(f"Mean Spearman: {np.mean([c['spearman'] for c in corr_results]):.4f}\n")

    # ── Step 4: Portfolio combination grid ──
    print("\n" + "=" * 70)
    print("Step 4: Portfolio Combinations (Keltner + MR)")
    print("=" * 70)

    # Normalize Keltner daily PnL per lot (it was run at FIXED_LOT)
    kc_daily_per_lot = {d: v / FIXED_LOT for d, v in kc_daily.items()}

    portfolio_results = []
    for mr in mr_results:
        mr_daily_per_lot = {d: v / FIXED_LOT for d, v in mr['_daily'].items()}
        for kc_lot, mr_lot in product(KC_LOT_GRID, MR_LOT_GRID):
            kc_scaled = {d: v * kc_lot for d, v in kc_daily_per_lot.items()}
            mr_scaled = {d: v * mr_lot for d, v in mr_daily_per_lot.items()}
            port_sharpe, port_dd = _portfolio_sharpe(
                [kc_scaled, mr_scaled], [1.0, 1.0])
            kc_only_sh = _sharpe_from_daily({d: v * kc_lot for d, v in kc_daily_per_lot.items()})
            portfolio_results.append({
                'mr_label': mr['label'], 'mr_type': mr['type'],
                'kc_lot': kc_lot, 'mr_lot': mr_lot,
                'port_sharpe': round(port_sharpe, 4),
                'port_maxdd': round(port_dd, 2),
                'kc_only_sharpe': round(kc_only_sh, 4),
                'sharpe_delta': round(port_sharpe - kc_only_sh, 4),
                'mr_sharpe': mr['sharpe'],
            })

    portfolio_results.sort(key=lambda x: x['port_sharpe'], reverse=True)

    with open(f"{OUTPUT_DIR}/portfolio_combos.txt", 'w', encoding='utf-8') as f:
        f.write("R60 Step 4: Portfolio Combinations (Keltner + MR)\n")
        f.write("=" * 130 + "\n\n")
        f.write("Top 30 by Portfolio Sharpe:\n\n")
        header = (f"{'Rank':>4} {'MR_Type':<10} {'KC_Lot':>7} {'MR_Lot':>7} "
                  f"{'Port_Sh':>9} {'KC_Sh':>8} {'Delta':>8} {'MaxDD':>10} "
                  f"{'MR_Label':<45}")
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for i, p in enumerate(portfolio_results[:30], 1):
            f.write(f"{i:>4} {p['mr_type']:<10} {p['kc_lot']:>7.2f} {p['mr_lot']:>7.2f} "
                    f"{p['port_sharpe']:>9.4f} {p['kc_only_sharpe']:>8.4f} "
                    f"{p['sharpe_delta']:>+8.4f} {fmt(p['port_maxdd']):>10} "
                    f"{p['mr_label'][:45]}\n")

        f.write(f"\n\nAll combos with positive Sharpe delta:\n\n")
        pos_delta = [p for p in portfolio_results if p['sharpe_delta'] > 0]
        f.write(f"  {len(pos_delta)} / {len(portfolio_results)} combos improve Sharpe\n\n")
        if pos_delta:
            by_type = defaultdict(list)
            for p in pos_delta:
                by_type[p['mr_type']].append(p)
            for stype in sorted(by_type.keys()):
                items = by_type[stype]
                best = max(items, key=lambda x: x['sharpe_delta'])
                f.write(f"  {stype}: {len(items)} positive combos, "
                        f"best delta={best['sharpe_delta']:+.4f} "
                        f"(KC={best['kc_lot']}, MR={best['mr_lot']})\n")

    with open(f"{OUTPUT_DIR}/portfolio_combos.json", 'w') as f:
        json.dump(portfolio_results[:50], f, indent=2, default=str)

    print(f"\n  {len(portfolio_results)} portfolio combos evaluated")
    print("  Top 5:")
    for p in portfolio_results[:5]:
        print(f"    KC={p['kc_lot']} + {p['mr_type']:<8} MR={p['mr_lot']}  "
              f"Sharpe={p['port_sharpe']:.4f} ({p['sharpe_delta']:>+.4f})  "
              f"MaxDD={fmt(p['port_maxdd'])}")

    # ── Summary ──
    total_time = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"R60 Complete — Total: {total_time:.0f}s ({total_time / 60:.1f}min)")
    print(f"{'=' * 70}")

    with open(f"{OUTPUT_DIR}/r60_summary.txt", 'w', encoding='utf-8') as f:
        f.write(f"R60 Mean-Reversion Validation — Summary\n{'=' * 60}\n")
        f.write(f"Total time: {total_time:.0f}s ({total_time / 60:.1f}min)\n")
        f.write(f"Lot={FIXED_LOT}  Spread=${SPREAD}  Cap=$37\n\n")

        f.write(f"R55a strategies loaded: {len(passed_strategies)} passed K-Fold\n")
        for line in summary_lines:
            f.write(line + "\n")

        f.write(f"\nKeltner Baseline: N={kc_result['n']}  Sharpe={kc_sharpe:.4f}  "
                f"PnL={fmt(kc_result['total_pnl'])}\n")

        f.write(f"\nMR Strategies Run: {len(mr_results)}\n")
        for mr in sorted(mr_results, key=lambda x: x['sharpe'], reverse=True)[:5]:
            f.write(f"  {mr['type']:<8} Sharpe={mr['sharpe']:>6.2f}  N={mr['n']:>4}  "
                    f"{mr['label'][:45]}\n")

        f.write(f"\nCorrelation with Keltner (lowest Pearson = best diversifier):\n")
        for c in corr_results[:5]:
            f.write(f"  {c['type']:<8} Pearson={c['pearson']:>+.4f}  "
                    f"Spearman={c['spearman']:>+.4f}  {c['label'][:40]}\n")

        f.write(f"\nBest Portfolio Combinations:\n")
        for p in portfolio_results[:5]:
            f.write(f"  KC={p['kc_lot']} + {p['mr_type']:<8} MR={p['mr_lot']}  "
                    f"Sharpe={p['port_sharpe']:.4f} ({p['sharpe_delta']:>+.4f})  "
                    f"MaxDD={fmt(p['port_maxdd'])}\n")

        pos_delta = [p for p in portfolio_results if p['sharpe_delta'] > 0]
        f.write(f"\nConclusion: {len(pos_delta)}/{len(portfolio_results)} portfolio combos "
                f"improve Sharpe over Keltner-only.\n")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()

#!/usr/bin/env python3
"""
R61 — 6-Strategy Portfolio Optimization (Cap$37 + 0.05 lot)
============================================================
Extends R56 with updated Cap$37, SPREAD=0.30, FIXED_LOT=0.05.

6 strategies:
  1. L8_MAX   (M15 Keltner Channel, H1 filter, EqCurve, MaxLoss$37)
  2. D1_KC    (D1 Keltner Channel, EMA10/Mult2.5, ADX>=18)
  3. H4_KC    (H4 Keltner Channel, EMA15/Mult2.5, ADX>=10)
  4. PSAR     (H1 Parabolic SAR, AF 0.01/0.05)
  5. TSMOM    (H1 Time Series Momentum, fast=480/slow=720)
  6. SESS_BO  (H1 Session Breakout, peak_12_14 LB3)

Step 1: Generate daily PnL for 6 strategies (fixed lot=0.03 base)
Step 2: Brute-force lot grid [0, 0.01, 0.02, 0.03], total<=0.15
Step 3: K-Fold 6-fold validation on top 10 combos
"""
import sys, os, io, time, json, multiprocessing as mp
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

OUTPUT_DIR = Path("results/r61_6strat_portfolio")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_WORKERS = 6
FIXED_LOT = 0.05
SPREAD = 0.30
BASE_LOT = 0.03

FOLDS = [
    ("Fold1", "2015-01-01", "2017-01-01"),
    ("Fold2", "2017-01-01", "2019-01-01"),
    ("Fold3", "2019-01-01", "2021-01-01"),
    ("Fold4", "2021-01-01", "2023-01-01"),
    ("Fold5", "2023-01-01", "2025-01-01"),
    ("Fold6", "2025-01-01", "2026-05-01"),
]

STRAT_NAMES = ['L8_MAX', 'D1_KC', 'H4_KC', 'PSAR', 'TSMOM', 'SESS_BO']
STRAT_KEYS  = ['l8', 'd1', 'h4', 'psar', 'ts', 'sb']

LOT_GRID = [0.00, 0.01, 0.02, 0.03]
MAX_TOTAL_LOT = 0.15

def fmt(v):
    return f"${v:>10,.0f}" if v >= 0 else f"-${abs(v):>9,.0f}"


# ═══════════════════════════════════════════════════════════════
# Indicator helpers (self-contained, no engine dependency)
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


def compute_atr(df, period=14):
    tr = pd.DataFrame({
        'hl': df['High'] - df['Low'],
        'hc': (df['High'] - df['Close'].shift(1)).abs(),
        'lc': (df['Low'] - df['Close'].shift(1)).abs(),
    }).max(axis=1)
    return tr.rolling(period).mean()


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


def add_psar(df, af_start=0.02, af_max=0.20):
    df = df.copy()
    n = len(df); psar = np.zeros(n); direction = np.ones(n)
    af = af_start; ep = df['High'].iloc[0]; psar[0] = df['Low'].iloc[0]
    for i in range(1, n):
        prev_psar = psar[i-1]
        if direction[i-1] == 1:
            psar[i] = prev_psar + af * (ep - prev_psar)
            psar[i] = min(psar[i], df['Low'].iloc[i-1], df['Low'].iloc[max(0, i-2)])
            if df['Low'].iloc[i] < psar[i]:
                direction[i] = -1; psar[i] = ep; ep = df['Low'].iloc[i]; af = af_start
            else:
                direction[i] = 1
                if df['High'].iloc[i] > ep:
                    ep = df['High'].iloc[i]; af = min(af + af_start, af_max)
        else:
            psar[i] = prev_psar + af * (ep - prev_psar)
            psar[i] = max(psar[i], df['High'].iloc[i-1], df['High'].iloc[max(0, i-2)])
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


# ═══════════════════════════════════════════════════════════════
# Simple backtest → trade list
# ═══════════════════════════════════════════════════════════════

def _mk(pos, exit_p, exit_time, reason, bar_idx, pnl):
    return {'dir': pos['dir'], 'entry': pos['entry'], 'exit': exit_p,
            'entry_time': pos['time'], 'exit_time': exit_time,
            'pnl': pnl, 'reason': reason, 'bars': bar_idx - pos['bar']}


def backtest_kc_trades(df_prepared, adx_thresh=18, sl_atr=3.5, tp_atr=8.0,
                       trail_act_atr=0.28, trail_dist_atr=0.06, max_hold=20,
                       spread=SPREAD, lot=BASE_LOT):
    df = df_prepared; trades = []; pos = None
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
    return trades


def backtest_psar_trades(df_prepared, sl_atr=3.5, tp_atr=8.0,
                         trail_act_atr=0.28, trail_dist_atr=0.06, max_hold=50,
                         spread=SPREAD, lot=BASE_LOT):
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
    return trades


def backtest_tsmom_trades(h1_df, fast=480, slow=720, sl_atr=4.5, tp_atr=6.0,
                          trail_act_atr=0.14, trail_dist_atr=0.025, max_hold=20,
                          spread=SPREAD, lot=BASE_LOT):
    df = h1_df.copy()
    if 'ATR' not in df.columns:
        df['ATR'] = compute_atr(df)

    close = df['Close'].values; high = df['High'].values; low = df['Low'].values
    atr = df['ATR'].values; times = df.index; n = len(close)

    weights = [(fast, 0.5), (slow, 0.5)]
    max_lb = max(lb for lb, _ in weights)
    score = np.full(n, np.nan)
    for i in range(max_lb, n):
        s = 0.0
        for lb, w in weights:
            if i >= lb:
                s += w * np.sign(close[i] / close[i - lb] - 1.0)
        score[i] = s

    trades = []; pos = None; last_exit = -999
    for i in range(max_lb + 1, n):
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
                ad = trail_act_atr * pos['atr']; td = trail_dist_atr * pos['atr']
                if pos['dir'] == 'BUY' and h - pos['entry'] >= ad:
                    ts_p = h - td
                    if lo_v <= ts_p:
                        trades.append(_mk(pos, c, times[i], "Trail", i,
                                          (ts_p - pos['entry'] - spread) * lot * 100)); exited = True
                elif pos['dir'] == 'SELL' and pos['entry'] - lo_v >= ad:
                    ts_p = lo_v + td
                    if h >= ts_p:
                        trades.append(_mk(pos, c, times[i], "Trail", i,
                                          (pos['entry'] - ts_p - spread) * lot * 100)); exited = True
                if not exited and held >= max_hold:
                    trades.append(_mk(pos, c, times[i], "Timeout", i, pnl_c)); exited = True
            if not exited and not np.isnan(score[i]):
                if pos['dir'] == 'BUY' and score[i] < 0:
                    trades.append(_mk(pos, c, times[i], "Reversal", i, pnl_c)); exited = True
                elif pos['dir'] == 'SELL' and score[i] > 0:
                    trades.append(_mk(pos, c, times[i], "Reversal", i, pnl_c)); exited = True
            if exited: pos = None; last_exit = i; continue

        if pos is not None or i - last_exit < 2: continue
        if np.isnan(score[i]) or np.isnan(cur_atr) or cur_atr < 0.1: continue
        if score[i] > 0 and score[i - 1] <= 0:
            pos = {'dir': 'BUY', 'entry': c + spread/2, 'bar': i, 'time': times[i], 'atr': cur_atr}
        elif score[i] < 0 and score[i - 1] >= 0:
            pos = {'dir': 'SELL', 'entry': c - spread/2, 'bar': i, 'time': times[i], 'atr': cur_atr}
    return trades


def backtest_session_trades(h1_df, session="peak_12_14", lookback_bars=3,
                            sl_atr=3.0, tp_atr=6.0, trail_act_atr=0.14,
                            trail_dist_atr=0.025, max_hold=20,
                            spread=SPREAD, lot=BASE_LOT):
    SESSION_DEFS = {
        "asian":      (0, 7),
        "london":     (8, 11),
        "ny_peak":    (12, 16),
        "late":       (17, 23),
        "peak_12_14": (12, 14),
    }
    df = h1_df.copy()
    df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR'])

    trades = []; pos = None
    close = df['Close'].values; high = df['High'].values; low = df['Low'].values
    atr = df['ATR'].values; hours = df.index.hour
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
        range_low  = min(low[i - lookback_bars:i])

        if c > range_high:
            pos = {'dir': 'BUY', 'entry': c + spread/2, 'bar': i, 'time': times[i], 'atr': cur_atr}
        elif c < range_low:
            pos = {'dir': 'SELL', 'entry': c - spread/2, 'bar': i, 'time': times[i], 'atr': cur_atr}

    return trades


# ═══════════════════════════════════════════════════════════════
# PnL helpers
# ═══════════════════════════════════════════════════════════════

def trades_to_daily_pnl(trades):
    daily = {}
    for t in trades:
        d = pd.Timestamp(t['exit_time']).date()
        daily[d] = daily.get(d, 0) + t['pnl']
    if not daily:
        return pd.Series(dtype=float)
    return pd.Series(daily).sort_index()


def calc_portfolio_stats(daily_pnl_series, label=""):
    if daily_pnl_series.empty or daily_pnl_series.sum() == 0:
        return {'label': label, 'sharpe': 0, 'total_pnl': 0, 'max_dd': 0,
                'n_days': 0, 'avg_daily': 0, 'volatility': 0}
    arr = daily_pnl_series.values
    eq = np.cumsum(arr)
    dd = (np.maximum.accumulate(eq) - eq).max()
    sh = float(arr.mean() / arr.std() * np.sqrt(252)) if arr.std() > 0 else 0
    return {
        'label': label, 'sharpe': round(sh, 3),
        'total_pnl': round(float(arr.sum()), 2), 'max_dd': round(float(dd), 2),
        'n_days': len(arr), 'avg_daily': round(float(arr.mean()), 2),
        'volatility': round(float(arr.std()), 2),
    }


def save_text(filename, text):
    path = OUTPUT_DIR / filename
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"  [Saved] {path}", flush=True)


def save_checkpoint(data, filename):
    path = OUTPUT_DIR / filename
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, default=str)
    print(f"  [Checkpoint] {path}", flush=True)


def load_checkpoint(filename):
    path = OUTPUT_DIR / filename
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


# ═══════════════════════════════════════════════════════════════
# Step 1: Generate daily PnL for 6 strategies (full range)
# ═══════════════════════════════════════════════════════════════

def _run_l8_max(m15_df, h1_df, lot=BASE_LOT, maxloss_cap=37):
    from backtest.runner import DataBundle, run_variant, LIVE_PARITY_KWARGS
    kw = {
        **LIVE_PARITY_KWARGS,
        'maxloss_cap': maxloss_cap,
        'spread_cost': SPREAD, 'initial_capital': 2000,
        'min_lot_size': lot, 'max_lot_size': lot,
    }
    data = DataBundle(m15_df, h1_df)
    result = run_variant(data, "L8_MAX", verbose=False, **kw)
    raw = result.get('_trades', [])
    trades = []
    for t in raw:
        pnl = t.pnl if hasattr(t, 'pnl') else t.get('pnl', 0)
        ext = t.exit_time if hasattr(t, 'exit_time') else t.get('exit_time', '')
        entry_t = t.entry_time if hasattr(t, 'entry_time') else t.get('entry_time', '')
        direction = t.direction if hasattr(t, 'direction') else t.get('direction', '')
        trades.append({'pnl': pnl, 'exit_time': ext, 'entry_time': entry_t, 'direction': direction})
    return trades


def generate_daily_pnl_all(h1_df, m15_df):
    print("\n  [Step 1] Generating daily PnL for 6 strategies...", flush=True)
    daily_pnls = {}
    strat_info = []

    # 1. L8_MAX
    print("    L8_MAX (M15 engine)...", flush=True)
    l8_trades_raw = _run_l8_max(m15_df, h1_df, lot=BASE_LOT, maxloss_cap=37)
    daily_pnls['L8_MAX'] = trades_to_daily_pnl(l8_trades_raw)
    strat_info.append(('L8_MAX', len(l8_trades_raw), sum(t['pnl'] for t in l8_trades_raw)))
    print(f"      L8_MAX: {len(l8_trades_raw)} trades, {fmt(sum(t['pnl'] for t in l8_trades_raw))}", flush=True)

    # 2. D1_KC
    print("    D1_KC...", flush=True)
    d1_df = h1_df.resample('D').agg({'Open': 'first', 'High': 'max', 'Low': 'min',
                                      'Close': 'last', 'Volume': 'sum'}).dropna()
    d1_df = add_kc(d1_df, 10, 14, 2.5); d1_df = d1_df.dropna()
    d1_trades = backtest_kc_trades(d1_df, adx_thresh=18, sl_atr=3.0, tp_atr=6.0,
                                    trail_act_atr=0.20, trail_dist_atr=0.04, max_hold=240)
    daily_pnls['D1_KC'] = trades_to_daily_pnl(d1_trades)
    strat_info.append(('D1_KC', len(d1_trades), sum(t['pnl'] for t in d1_trades)))
    print(f"      D1_KC: {len(d1_trades)} trades, {fmt(sum(t['pnl'] for t in d1_trades))}", flush=True)

    # 3. H4_KC
    print("    H4_KC...", flush=True)
    h4_df = h1_df.resample('4h').agg({'Open': 'first', 'High': 'max', 'Low': 'min',
                                       'Close': 'last', 'Volume': 'sum'}).dropna()
    h4_df = add_kc(h4_df, 15, 14, 2.5); h4_df = h4_df.dropna()
    h4_trades = backtest_kc_trades(h4_df, adx_thresh=10, sl_atr=3.0, tp_atr=6.0,
                                    trail_act_atr=0.18, trail_dist_atr=0.035, max_hold=80)
    daily_pnls['H4_KC'] = trades_to_daily_pnl(h4_trades)
    strat_info.append(('H4_KC', len(h4_trades), sum(t['pnl'] for t in h4_trades)))
    print(f"      H4_KC: {len(h4_trades)} trades, {fmt(sum(t['pnl'] for t in h4_trades))}", flush=True)

    # 4. PSAR
    print("    PSAR...", flush=True)
    h1_psar = add_psar(h1_df.copy(), 0.01, 0.05)
    psar_trades = backtest_psar_trades(h1_psar, sl_atr=2.0, tp_atr=16.0,
                                        trail_act_atr=0.20, trail_dist_atr=0.04, max_hold=80)
    daily_pnls['PSAR'] = trades_to_daily_pnl(psar_trades)
    strat_info.append(('PSAR', len(psar_trades), sum(t['pnl'] for t in psar_trades)))
    print(f"      PSAR: {len(psar_trades)} trades, {fmt(sum(t['pnl'] for t in psar_trades))}", flush=True)

    # 5. TSMOM (R53 best)
    print("    TSMOM...", flush=True)
    ts_trades = backtest_tsmom_trades(h1_df, fast=480, slow=720, sl_atr=4.5, tp_atr=6.0,
                                       trail_act_atr=0.14, trail_dist_atr=0.025, max_hold=20)
    daily_pnls['TSMOM'] = trades_to_daily_pnl(ts_trades)
    strat_info.append(('TSMOM', len(ts_trades), sum(t['pnl'] for t in ts_trades)))
    print(f"      TSMOM: {len(ts_trades)} trades, {fmt(sum(t['pnl'] for t in ts_trades))}", flush=True)

    # 6. Session Breakout (peak_12_14, LB3, SL3, TP6)
    print("    Session Breakout...", flush=True)
    sb_trades = backtest_session_trades(h1_df, session="peak_12_14", lookback_bars=3,
                                         sl_atr=3.0, tp_atr=6.0, trail_act_atr=0.14,
                                         trail_dist_atr=0.025, max_hold=20)
    daily_pnls['SESS_BO'] = trades_to_daily_pnl(sb_trades)
    strat_info.append(('SESS_BO', len(sb_trades), sum(t['pnl'] for t in sb_trades)))
    print(f"      SESS_BO: {len(sb_trades)} trades, {fmt(sum(t['pnl'] for t in sb_trades))}", flush=True)

    # Align all series to common date index
    all_dates = set()
    for s in daily_pnls.values():
        all_dates.update(s.index.tolist())
    all_dates = sorted(all_dates)
    idx = pd.Index(all_dates)
    for name in daily_pnls:
        daily_pnls[name] = daily_pnls[name].reindex(idx, fill_value=0.0)

    # Individual stats report
    lines = []
    lines.append(f"R61 Individual Strategy Stats (lot={BASE_LOT}, spread=${SPREAD})")
    lines.append(f"Range: {all_dates[0]} -> {all_dates[-1]} ({len(all_dates)} days)")
    lines.append(f"\n{'Strategy':>10} {'Sharpe':>8} {'PnL':>12} {'MaxDD':>10} {'Days':>6} {'Trades':>8}")
    lines.append("-" * 60)
    for name, n_trades, total_pnl in strat_info:
        st = calc_portfolio_stats(daily_pnls[name], name)
        lines.append(f"{name:>10} {st['sharpe']:>8.2f} {fmt(st['total_pnl']):>12} "
                      f"{fmt(st['max_dd']):>10} {st['n_days']:>6} {n_trades:>8}")

    # Correlation matrix
    combined = pd.DataFrame(daily_pnls)
    corr = combined.corr()
    lines.append(f"\nCorrelation matrix:")
    header = f"  {'':>10}"
    for n in STRAT_NAMES:
        header += f" {n:>8}"
    lines.append(header)
    for n1 in STRAT_NAMES:
        row = f"  {n1:>10}"
        for n2 in STRAT_NAMES:
            row += f" {corr.loc[n1,n2]:>8.3f}"
        lines.append(row)

    report = "\n".join(lines)
    print(report, flush=True)
    save_text("individual_stats.txt", report)

    return daily_pnls


# ═══════════════════════════════════════════════════════════════
# Step 2: Brute-force lot grid
# ═══════════════════════════════════════════════════════════════

def run_lot_grid(daily_pnls):
    print(f"\n{'='*80}")
    print(f"  [Step 2] 6-Strategy Lot Grid Search")
    print(f"  Grid per strategy: {LOT_GRID} | Max total lot: {MAX_TOTAL_LOT}")
    print(f"{'='*80}")

    existing = load_checkpoint("lot_grid_r61.json")
    if existing:
        print(f"  [Resume] Found {len(existing)} results", flush=True)
        return existing

    base_daily = {name: daily_pnls[name].values for name in STRAT_NAMES}

    combos = list(product(*([LOT_GRID] * 6)))
    combos = [c for c in combos if sum(c) > 0 and sum(c) <= MAX_TOTAL_LOT]
    total = len(combos)
    print(f"  Valid combos: {total:,} (non-zero, total<={MAX_TOTAL_LOT})", flush=True)

    t0 = time.time()
    results = []

    for idx_c, lots in enumerate(combos):
        l_l8, l_d1, l_h4, l_psar, l_ts, l_sb = lots
        combined = np.zeros_like(base_daily['L8_MAX'], dtype=float)
        for name, lot_val in zip(STRAT_NAMES, lots):
            if lot_val > 0:
                combined += base_daily[name] * (lot_val / BASE_LOT)

        eq = np.cumsum(combined)
        dd = float((np.maximum.accumulate(eq) - eq).max()) if len(eq) > 0 else 0
        total_pnl = float(combined.sum())
        std = combined.std()
        sharpe = float(combined.mean() / std * np.sqrt(252)) if std > 0 else 0
        total_lot = sum(lots)

        label = f"L8={l_l8}_D1={l_d1}_H4={l_h4}_PS={l_psar}_TS={l_ts}_SB={l_sb}"
        results.append({
            'label': label,
            'l8': l_l8, 'd1': l_d1, 'h4': l_h4, 'psar': l_psar, 'ts': l_ts, 'sb': l_sb,
            'total_lot': round(total_lot, 2),
            'sharpe': round(sharpe, 3),
            'total_pnl': round(total_pnl, 2),
            'max_dd': round(dd, 2),
            'pnl_per_lot': round(total_pnl / total_lot, 2) if total_lot > 0 else 0,
        })

        if (idx_c + 1) % 500 == 0 or (idx_c + 1) == total:
            elapsed = time.time() - t0
            rate = (idx_c + 1) / elapsed if elapsed > 0 else 1
            eta = (total - idx_c - 1) / rate
            print(f"    {idx_c+1:,}/{total:,} ({(idx_c+1)/total*100:.0f}%) | "
                  f"{elapsed:.1f}s | ETA {eta:.1f}s", flush=True)

    results.sort(key=lambda x: x['sharpe'], reverse=True)
    save_checkpoint(results, "lot_grid_r61.json")

    # Write top 50 to text
    lines = [f"R61 Lot Grid — Top 50 of {total:,} combos (spread=${SPREAD}, base_lot={BASE_LOT})\n"]
    lines.append(f"{'Rank':>4} {'Label':>55} {'Sharpe':>8} {'PnL':>12} {'MaxDD':>10} "
                 f"{'TotLot':>8} {'PnL/Lot':>10}")
    lines.append("-" * 115)
    for i, r in enumerate(results[:50], 1):
        lines.append(f"{i:>4} {r['label']:>55} {r['sharpe']:>8.3f} {fmt(r['total_pnl']):>12} "
                     f"{fmt(r['max_dd']):>10} {r['total_lot']:>8.2f} {fmt(r['pnl_per_lot']):>10}")
    report = "\n".join(lines)
    print(report, flush=True)
    save_text("lot_grid_top50.txt", report)

    return results


# ═══════════════════════════════════════════════════════════════
# Step 3: K-Fold for top combos
# ═══════════════════════════════════════════════════════════════

def _generate_fold_daily_pnls(h1_df, m15_df):
    """Generate daily PnL dicts for a single fold's data range."""
    daily = {}

    # L8_MAX via engine
    l8_raw = _run_l8_max(m15_df, h1_df, lot=BASE_LOT, maxloss_cap=37)
    daily['L8_MAX'] = trades_to_daily_pnl(l8_raw)

    # D1_KC
    d1_df = h1_df.resample('D').agg({'Open': 'first', 'High': 'max', 'Low': 'min',
                                      'Close': 'last', 'Volume': 'sum'}).dropna()
    d1_df = add_kc(d1_df, 10, 14, 2.5); d1_df = d1_df.dropna()
    daily['D1_KC'] = trades_to_daily_pnl(
        backtest_kc_trades(d1_df, adx_thresh=18, sl_atr=3.0, tp_atr=6.0,
                           trail_act_atr=0.20, trail_dist_atr=0.04, max_hold=240))

    # H4_KC
    h4_df = h1_df.resample('4h').agg({'Open': 'first', 'High': 'max', 'Low': 'min',
                                       'Close': 'last', 'Volume': 'sum'}).dropna()
    h4_df = add_kc(h4_df, 15, 14, 2.5); h4_df = h4_df.dropna()
    daily['H4_KC'] = trades_to_daily_pnl(
        backtest_kc_trades(h4_df, adx_thresh=10, sl_atr=3.0, tp_atr=6.0,
                           trail_act_atr=0.18, trail_dist_atr=0.035, max_hold=80))

    # PSAR
    h1_psar = add_psar(h1_df.copy(), 0.01, 0.05)
    daily['PSAR'] = trades_to_daily_pnl(
        backtest_psar_trades(h1_psar, sl_atr=2.0, tp_atr=16.0,
                             trail_act_atr=0.20, trail_dist_atr=0.04, max_hold=80))

    # TSMOM
    daily['TSMOM'] = trades_to_daily_pnl(
        backtest_tsmom_trades(h1_df, fast=480, slow=720, sl_atr=4.5, tp_atr=6.0,
                              trail_act_atr=0.14, trail_dist_atr=0.025, max_hold=20))

    # Session Breakout
    daily['SESS_BO'] = trades_to_daily_pnl(
        backtest_session_trades(h1_df, session="peak_12_14", lookback_bars=3,
                                 sl_atr=3.0, tp_atr=6.0, trail_act_atr=0.14,
                                 trail_dist_atr=0.025, max_hold=20))

    # Align
    all_dates = set()
    for s in daily.values():
        all_dates.update(s.index.tolist())
    all_dates = sorted(all_dates)
    idx = pd.Index(all_dates)
    return {name: daily[name].reindex(idx, fill_value=0.0).values for name in daily}


def run_kfold(h1_df, m15_df, results, top_n=10):
    print(f"\n{'='*80}")
    print(f"  [Step 3] K-Fold 6-Fold: Top {top_n} Portfolio Combos")
    print(f"{'='*80}")

    existing = load_checkpoint("kfold_r61.json")
    if existing:
        print(f"  [Resume] Found checkpoint", flush=True)
        return existing

    candidates = results[:top_n]
    kfold_results = []

    for ci, cand in enumerate(candidates):
        fold_sharpes = []
        for fname, start, end in FOLDS:
            fold_h1 = h1_df[start:end]
            fold_m15 = m15_df[start:end]
            if len(fold_h1) < 100:
                continue

            fold_pnls = _generate_fold_daily_pnls(fold_h1, fold_m15)
            n_days = len(next(iter(fold_pnls.values())))
            if n_days == 0:
                fold_sharpes.append(0)
                continue

            combined = np.zeros(n_days)
            for name, key in zip(STRAT_NAMES, STRAT_KEYS):
                lot = cand.get(key, 0)
                if lot > 0:
                    combined += fold_pnls[name] * (lot / BASE_LOT)

            std = combined.std()
            sh = float(combined.mean() / std * np.sqrt(252)) if std > 0 else 0
            fold_sharpes.append(sh)

        n_folds = len(fold_sharpes)
        passed = n_folds == 6 and all(s > 0 for s in fold_sharpes)
        kf_mean = np.mean(fold_sharpes) if fold_sharpes else 0
        kf_min = min(fold_sharpes) if fold_sharpes else 0

        kfold_results.append({
            'label': cand['label'], 'full_sharpe': cand['sharpe'],
            'kfold_mean': round(float(kf_mean), 2), 'kfold_min': round(float(kf_min), 2),
            'kfold_folds': [round(s, 2) for s in fold_sharpes],
            'passed': passed,
            **{k: cand[k] for k in STRAT_KEYS + ['total_lot', 'total_pnl', 'max_dd']},
        })

        p_str = "PASS" if passed else "FAIL"
        print(f"  [{ci+1}/{len(candidates)}] {cand['label']:>55} "
              f"Full={cand['sharpe']:.3f} KF={kf_mean:.2f} {p_str}", flush=True)

    save_checkpoint(kfold_results, "kfold_r61.json")

    # Text report
    lines = [f"R61 K-Fold Top {top_n}\n"]
    lines.append(f"{'#':>3} {'Label':>55} {'FullSh':>8} {'KFMean':>8} {'KFMin':>8} {'Pass':>6} "
                 f"{'Folds':>50}")
    lines.append("-" * 140)
    for i, k in enumerate(kfold_results, 1):
        folds_str = ", ".join(f"{s:.2f}" for s in k['kfold_folds'])
        lines.append(f"{i:>3} {k['label']:>55} {k['full_sharpe']:>8.3f} {k['kfold_mean']:>8.2f} "
                     f"{k['kfold_min']:>8.2f} {'PASS' if k['passed'] else 'FAIL':>6} [{folds_str}]")
    report = "\n".join(lines)
    print(report, flush=True)
    save_text("kfold_top10.txt", report)

    return kfold_results


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 80)
    print("  R61: 6-Strategy Portfolio Optimization (Cap$37 + 0.05 lot)")
    print("=" * 80)

    from backtest.runner import DataBundle
    data = DataBundle.load_default()
    h1_df = data.h1_df.copy()
    m15_df = data.m15_df.copy()
    print(f"  H1: {len(h1_df)} bars | M15: {len(m15_df)} bars")

    daily_pnls = generate_daily_pnl_all(h1_df, m15_df)
    lot_results = run_lot_grid(daily_pnls)
    kfold_results = run_kfold(h1_df, m15_df, lot_results, top_n=10)

    elapsed = time.time() - t0

    # Summary
    n_passed = sum(1 for k in kfold_results if k.get('passed'))
    passed = [k for k in kfold_results if k.get('passed')]
    passed.sort(key=lambda x: x['kfold_mean'], reverse=True)

    lines = [f"R61 SUMMARY (elapsed {elapsed/60:.1f} min)\n"]
    lines.append(f"K-Fold passed: {n_passed}/{len(kfold_results)}")
    lines.append(f"\nTop K-Fold passed:")
    for i, k in enumerate(passed[:10], 1):
        lines.append(f"  {i:>3}. {k['label']:>55} Full={k['full_sharpe']:.3f} "
                     f"KF={k['kfold_mean']:.2f} (min={k['kfold_min']:.2f}) "
                     f"PnL={fmt(k['total_pnl'])} DD={fmt(k['max_dd'])}")
    report = "\n".join(lines)
    print(f"\n{'='*80}")
    print(report)
    print(f"{'='*80}")
    save_text("r61_summary.txt", report)


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()

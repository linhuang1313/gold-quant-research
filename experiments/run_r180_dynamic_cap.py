#!/usr/bin/env python3
"""
R180 — Dynamic ATR-Multiple Cap Verification
=============================================
Scans cap_atr_multiple from 4.0-12.0 for each of the 6 live strategies
at actual lot sizes. Finds the flat Sharpe plateau where Cap stops
interfering with the strategy. Uses per-trade entry-bar ATR for cap.

Phase 1: Full history (2015-2026) — per-strategy ATR-multiple grid
Phase 2: Checklist validation (cap > SL, plateau, rate < 5%, no-cap parity)
Phase 3: K-Fold 6-fold on recommended multiples
Phase 4: 2025-2026 recent-era test
"""
import sys, os, time, json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import glob as _glob

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

OUTPUT_DIR = Path("results/r180_dynamic_cap")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100
SPREAD = 0.30

LIVE_STRATEGIES = {
    'L8_MAX':      {'lot': 0.02, 'sl_atr': 3.5,  'current_cap': 35},
    'PSAR':        {'lot': 0.09, 'sl_atr': 4.0,  'current_cap': 60},
    'TSMOM':       {'lot': 0.15, 'sl_atr': 6.0,  'current_cap': 0},
    'SESS_BO':     {'lot': 0.13, 'sl_atr': 4.5,  'current_cap': 35},
    'DUAL_THRUST': {'lot': 0.04, 'sl_atr': 4.5,  'current_cap': 35},
    'CHANDELIER':  {'lot': 0.08, 'sl_atr': 4.5,  'current_cap': 35},
}

CAP_MULTIPLES = [4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 9.0, 10.0, 12.0, 999]

FOLDS = [
    ("Fold1", "2015-01-01", "2017-01-01"),
    ("Fold2", "2017-01-01", "2019-01-01"),
    ("Fold3", "2019-01-01", "2021-01-01"),
    ("Fold4", "2021-01-01", "2023-01-01"),
    ("Fold5", "2023-01-01", "2025-01-01"),
    ("Fold6", "2025-01-01", "2026-06-01"),
]


# ═══════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════

def fmt(v):
    return f"${v:>10,.0f}" if v >= 0 else f"-${abs(v):>9,.0f}"


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
    df['PSAR_dir'] = direction; df['ATR'] = compute_atr(df)
    return df


def _mk(pos, exit_p, exit_time, reason, bar_idx, pnl):
    return {'dir': pos['dir'], 'entry': pos['entry'], 'exit': exit_p,
            'entry_time': pos['time'], 'exit_time': exit_time,
            'pnl': pnl, 'reason': reason, 'bars': bar_idx - pos['bar']}


def _run_exit_dynamic_cap(pos, i, h, lo_v, c, spread, lot, pv, times,
                          sl_atr, tp_atr, trail_act_atr, trail_dist_atr,
                          max_hold, cap_atr_mult=0):
    """Exit logic with dynamic ATR-based cap: cap_dollar = lot * pv * cap_atr_mult * entry_atr"""
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
    if cap_atr_mult > 0 and cap_atr_mult < 900:
        cap_dollar = lot * pv * cap_atr_mult * pos['atr']
        if pnl_c < -cap_dollar:
            return _mk(pos, c, times[i], "MaxLossCap", i, -cap_dollar)
    if pos['dir'] == 'BUY' and h - pos['entry'] >= trail_act_atr * pos['atr']:
        ts_p = h - trail_dist_atr * pos['atr']
        if lo_v <= ts_p:
            return _mk(pos, c, times[i], "Trail", i, (ts_p - pos['entry'] - spread) * lot * pv)
    elif pos['dir'] == 'SELL' and pos['entry'] - lo_v >= trail_act_atr * pos['atr']:
        ts_p = lo_v + trail_dist_atr * pos['atr']
        if h >= ts_p:
            return _mk(pos, c, times[i], "Trail", i, (pos['entry'] - ts_p - spread) * lot * pv)
    if held >= max_hold:
        return _mk(pos, c, times[i], "Timeout", i, pnl_c)
    return None


# ═══════════════════════════════════════════════════════════════
# Strategy backtest functions — use dynamic ATR cap
# ═══════════════════════════════════════════════════════════════

def bt_l8_max(h1_df, spread, lot, cap_atr_mult=0,
              adx_th=14, ema_period=25, kc_mult=1.2,
              sl_atr=3.5, tp_atr=8.0,
              trail_act=0.14, trail_dist=0.025, max_hold=20):
    df = h1_df.copy()
    df['ATR'] = compute_atr(df)
    df['ADX'] = compute_adx(df)
    df['EMA100'] = df['Close'].ewm(span=100, adjust=False).mean()
    df['KC_mid'] = df['Close'].ewm(span=ema_period, adjust=False).mean()
    df['KC_upper'] = df['KC_mid'] + kc_mult * df['ATR']
    df['KC_lower'] = df['KC_mid'] - kc_mult * df['ATR']
    df = df.dropna(subset=['ATR', 'ADX', 'EMA100', 'KC_upper'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; adx = df['ADX'].values
    ema100 = df['EMA100'].values
    kc_u = df['KC_upper'].values; kc_l = df['KC_lower'].values
    times = df.index; n = len(df)
    trades = []; pos = None; last_exit = -999
    for i in range(1, n):
        if pos is not None:
            result = _run_exit_dynamic_cap(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                                           sl_atr, tp_atr, trail_act, trail_dist, max_hold, cap_atr_mult)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if i - last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1 or np.isnan(adx[i]): continue
        if adx[i] < adx_th: continue
        if c[i] > kc_u[i] and c[i] > ema100[i]:
            pos = {'dir': 'BUY', 'entry': c[i] + spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif c[i] < kc_l[i] and c[i] < ema100[i]:
            pos = {'dir': 'SELL', 'entry': c[i] - spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_psar(h1_df, spread, lot, cap_atr_mult=0,
            sl_atr=4.0, tp_atr=6.0, trail_act=0.08, trail_dist=0.015, max_hold=15):
    df = add_psar(h1_df).dropna(subset=['PSAR_dir', 'ATR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    pdir = df['PSAR_dir'].values; atr = df['ATR'].values
    times = df.index; n = len(df)
    trades = []; pos = None; last_exit = -999
    for i in range(1, n):
        if pos is not None:
            result = _run_exit_dynamic_cap(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                                           sl_atr, tp_atr, trail_act, trail_dist, max_hold, cap_atr_mult)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if i - last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        if pdir[i-1] == -1 and pdir[i] == 1:
            pos = {'dir': 'BUY', 'entry': c[i] + spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif pdir[i-1] == 1 and pdir[i] == -1:
            pos = {'dir': 'SELL', 'entry': c[i] - spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_tsmom(h1_df, spread, lot, cap_atr_mult=0,
             fast=480, slow=720, sl_atr=6.0, tp_atr=8.0,
             trail_act=0.14, trail_dist=0.025, max_hold=12):
    df = h1_df.copy(); df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; times = df.index; n = len(df)
    max_lb = max(fast, slow)
    score = np.full(n, np.nan)
    for i in range(max_lb, n):
        s = 0.0
        if c[i - fast] > 0: s += 0.5 * np.sign(c[i] / c[i - fast] - 1.0)
        if c[i - slow] > 0: s += 0.5 * np.sign(c[i] / c[i - slow] - 1.0)
        score[i] = s
    trades = []; pos = None; last_exit = -999
    for i in range(max_lb + 1, n):
        if pos is not None:
            result = _run_exit_dynamic_cap(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                                           sl_atr, tp_atr, trail_act, trail_dist, max_hold, cap_atr_mult)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            if pos['dir'] == 'BUY' and score[i] < 0:
                pnl = (c[i] - pos['entry'] - spread) * lot * PV
                trades.append(_mk(pos, c[i], times[i], "Reversal", i, pnl)); pos = None; last_exit = i; continue
            elif pos['dir'] == 'SELL' and score[i] > 0:
                pnl = (pos['entry'] - c[i] - spread) * lot * PV
                trades.append(_mk(pos, c[i], times[i], "Reversal", i, pnl)); pos = None; last_exit = i; continue
            continue
        if i - last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        if np.isnan(score[i]) or np.isnan(score[i-1]): continue
        if score[i] > 0 and score[i-1] <= 0:
            pos = {'dir': 'BUY', 'entry': c[i] + spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif score[i] < 0 and score[i-1] >= 0:
            pos = {'dir': 'SELL', 'entry': c[i] - spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_sess_bo(h1_df, spread, lot, cap_atr_mult=0,
               session_hour=12, lookback=4, sl_atr=4.5, tp_atr=4.0,
               trail_act=0.14, trail_dist=0.025, max_hold=20):
    df = h1_df.copy(); df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; hours = df.index.hour; times = df.index; n = len(df)
    trades = []; pos = None; last_exit = -999
    for i in range(lookback, n):
        if pos is not None:
            result = _run_exit_dynamic_cap(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                                           sl_atr, tp_atr, trail_act, trail_dist, max_hold, cap_atr_mult)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if hours[i] != session_hour: continue
        if i - last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        hh = max(h[i-j] for j in range(1, lookback + 1))
        ll = min(lo[i-j] for j in range(1, lookback + 1))
        if c[i] > hh:
            pos = {'dir': 'BUY', 'entry': c[i] + spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif c[i] < ll:
            pos = {'dir': 'SELL', 'entry': c[i] - spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_dual_thrust(h1_df, spread, lot, cap_atr_mult=0,
                   n_bars=6, k=0.5, sl_atr=4.5, tp_atr=8.0,
                   trail_act=0.14, trail_dist=0.025, max_hold=20):
    df = h1_df.copy(); df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    o = df['Open'].values; atr = df['ATR'].values
    times = df.index; n = len(df)
    trades = []; pos = None; last_exit = -999
    for i in range(n_bars, n):
        if pos is not None:
            result = _run_exit_dynamic_cap(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                                           sl_atr, tp_atr, trail_act, trail_dist, max_hold, cap_atr_mult)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if i - last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        hh = np.max(h[i - n_bars:i])
        lc = np.min(c[i - n_bars:i])
        hc = np.max(c[i - n_bars:i])
        ll = np.min(lo[i - n_bars:i])
        rng = max(hh - lc, hc - ll)
        buy_line = o[i] + k * rng
        sell_line = o[i] - k * rng
        if c[i] > buy_line:
            pos = {'dir': 'BUY', 'entry': c[i] + spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif c[i] < sell_line:
            pos = {'dir': 'SELL', 'entry': c[i] - spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_chandelier(h1_df, spread, lot, cap_atr_mult=0,
                  period=22, mult=3.0, ema_period=100,
                  sl_atr=4.5, tp_atr=8.0,
                  trail_act=0.14, trail_dist=0.025, max_hold=20):
    df = h1_df.copy()
    df['ATR'] = compute_atr(df, period=period)
    df['EMA'] = df['Close'].ewm(span=ema_period, adjust=False).mean()
    df = df.dropna(subset=['ATR', 'EMA'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; ema = df['EMA'].values
    times = df.index; n = len(df)
    chandelier_long = np.full(n, np.nan)
    chandelier_short = np.full(n, np.nan)
    for i in range(period, n):
        hh = np.max(h[i - period + 1:i + 1])
        ll = np.min(lo[i - period + 1:i + 1])
        chandelier_long[i] = hh - mult * atr[i]
        chandelier_short[i] = ll + mult * atr[i]
    direction = np.zeros(n)
    for i in range(period + 1, n):
        if np.isnan(chandelier_long[i]) or np.isnan(chandelier_short[i]):
            direction[i] = direction[i-1]; continue
        if c[i] > chandelier_short[i-1]:
            direction[i] = 1
        elif c[i] < chandelier_long[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
    trades = []; pos = None; last_exit = -999
    for i in range(period + 2, n):
        if pos is not None:
            result = _run_exit_dynamic_cap(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                                           sl_atr, tp_atr, trail_act, trail_dist, max_hold, cap_atr_mult)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if i - last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        flip_bull = direction[i] == 1 and direction[i-1] != 1 and c[i] > ema[i]
        flip_bear = direction[i] == -1 and direction[i-1] != -1 and c[i] < ema[i]
        if flip_bull:
            pos = {'dir': 'BUY', 'entry': c[i] + spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif flip_bear:
            pos = {'dir': 'SELL', 'entry': c[i] - spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


STRAT_BT = {
    'L8_MAX': bt_l8_max,
    'PSAR': bt_psar,
    'TSMOM': bt_tsmom,
    'SESS_BO': bt_sess_bo,
    'DUAL_THRUST': bt_dual_thrust,
    'CHANDELIER': bt_chandelier,
}


# ═══════════════════════════════════════════════════════════════
# Stats
# ═══════════════════════════════════════════════════════════════

def _trades_to_daily(trades):
    if not trades:
        return np.array([0.0])
    daily = {}
    for t in trades:
        d = pd.Timestamp(t['exit_time']).date()
        daily[d] = daily.get(d, 0) + t['pnl']
    if not daily:
        return np.array([0.0])
    return np.array([daily[k] for k in sorted(daily.keys())])


def _sharpe(arr):
    if len(arr) < 10 or np.std(arr, ddof=1) == 0:
        return 0.0
    return float(np.mean(arr) / np.std(arr, ddof=1) * np.sqrt(252))


def _max_dd(arr):
    if len(arr) == 0:
        return 0.0
    eq = np.cumsum(arr)
    return float((np.maximum.accumulate(eq) - eq).max())


def _compute_stats(trades):
    if not trades:
        return {'n': 0, 'sharpe': 0, 'pnl': 0, 'wr': 0, 'max_dd': 0,
                'cap_hits': 0, 'cap_pct': 0, 'sl_hits': 0, 'tp_hits': 0,
                'trail_hits': 0, 'timeout_hits': 0, 'reversal_hits': 0,
                'max_single_loss': 0}
    daily = _trades_to_daily(trades)
    pnls = [t['pnl'] for t in trades]
    n = len(trades)
    reasons = [t.get('reason', '') for t in trades]
    return {
        'n': n,
        'sharpe': round(_sharpe(daily), 3),
        'pnl': round(sum(pnls), 2),
        'wr': round(sum(1 for p in pnls if p > 0) / n * 100, 1),
        'max_dd': round(_max_dd(daily), 2),
        'cap_hits': sum(1 for r in reasons if r == 'MaxLossCap'),
        'cap_pct': round(sum(1 for r in reasons if r == 'MaxLossCap') / n * 100, 1),
        'sl_hits': sum(1 for r in reasons if r == 'SL'),
        'tp_hits': sum(1 for r in reasons if r == 'TP'),
        'trail_hits': sum(1 for r in reasons if r == 'Trail'),
        'timeout_hits': sum(1 for r in reasons if r == 'Timeout'),
        'reversal_hits': sum(1 for r in reasons if r == 'Reversal'),
        'max_single_loss': round(min(pnls), 3) if pnls else 0,
    }


def load_h1():
    candidates = sorted(_glob.glob("data/download/xauusd-h1-bid-2015-*.csv"))
    if not candidates:
        raise FileNotFoundError("No H1 data found")
    csv_path = candidates[-1]
    print(f"  Loading H1: {csv_path}", flush=True)
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df.set_index('timestamp')
    df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low',
                        'close': 'Close', 'volume': 'Volume'}, inplace=True)
    df = df[['Open', 'High', 'Low', 'Close']].copy()
    print(f"  {len(df)} bars ({df.index[0]} ~ {df.index[-1]})", flush=True)
    return df


# ═══════════════════════════════════════════════════════════════
# Phase 1: Full-history ATR-multiple grid
# ═══════════════════════════════════════════════════════════════

def run_grid(h1_df, mean_atr):
    all_results = {}

    for strat_name in ['L8_MAX', 'PSAR', 'TSMOM', 'SESS_BO', 'DUAL_THRUST', 'CHANDELIER']:
        cfg = LIVE_STRATEGIES[strat_name]
        lot = cfg['lot']
        sl_atr_mult = cfg['sl_atr']
        current_cap = cfg['current_cap']
        bt_fn = STRAT_BT[strat_name]

        current_cap_atr = current_cap / (lot * PV * mean_atr) if current_cap > 0 else 0

        print(f"\n{'=' * 90}", flush=True)
        print(f"  {strat_name}  (lot={lot}, SL={sl_atr_mult}x ATR, "
              f"current Cap=${current_cap} = {current_cap_atr:.1f}x ATR)", flush=True)
        print(f"{'=' * 90}", flush=True)

        results = []
        for mult in CAP_MULTIPLES:
            trades = bt_fn(h1_df, spread=SPREAD, lot=lot, cap_atr_mult=mult)
            stats = _compute_stats(trades)

            cap_dollar_avg = lot * PV * mult * mean_atr if mult < 900 else float('inf')
            price_tol = mult * mean_atr if mult < 900 else float('inf')

            flag = ""
            if mult < 900 and mult < sl_atr_mult:
                flag = " *** BUG: Cap < SL ***"
            elif mult < 900 and mult == sl_atr_mult:
                flag = " * Cap == SL"

            stats['mult'] = mult
            stats['cap_dollar_avg'] = round(cap_dollar_avg, 2) if mult < 900 else None
            stats['price_tol'] = round(price_tol, 2) if mult < 900 else None
            stats['flag'] = flag
            results.append(stats)

        # Print grid table
        print(f"\n  {'Mult':>6} {'Cap$':>8} {'Sharpe':>7} {'PnL':>11} {'N':>5} "
              f"{'WR':>6} {'MaxDD':>9} {'CapExit':>7} {'Cap%':>6} "
              f"{'SL%':>5} {'TP%':>5} {'Trail%':>6} {'$/oz':>7}", flush=True)
        print(f"  {'-'*6} {'-'*8} {'-'*7} {'-'*11} {'-'*5} "
              f"{'-'*6} {'-'*9} {'-'*7} {'-'*6} "
              f"{'-'*5} {'-'*5} {'-'*6} {'-'*7}", flush=True)

        for r in results:
            mult = r['mult']
            mult_str = f"{mult:.1f}x" if mult < 900 else "NoCap"
            cap_str = f"${r['cap_dollar_avg']:.0f}" if r['cap_dollar_avg'] else "inf"
            pt_str = f"${r['price_tol']:.1f}" if r['price_tol'] else "inf"
            n = r['n']
            sl_pct = round(r['sl_hits'] / n * 100, 1) if n > 0 else 0
            tp_pct = round(r['tp_hits'] / n * 100, 1) if n > 0 else 0
            trail_pct = round(r['trail_hits'] / n * 100, 1) if n > 0 else 0
            print(f"  {mult_str:>6} {cap_str:>8} {r['sharpe']:>7.3f} {fmt(r['pnl']):>11} "
                  f"{n:>5} {r['wr']:>5.1f}% {fmt(r['max_dd']):>9} "
                  f"{r['cap_hits']:>7} {r['cap_pct']:>5.1f}% "
                  f"{sl_pct:>4.1f}% {tp_pct:>4.1f}% {trail_pct:>5.1f}% "
                  f"{pt_str:>7}{r['flag']}", flush=True)

        # Analyze plateau
        no_cap = [r for r in results if r['mult'] == 999][0]
        stable_results = [r for r in results if r['mult'] < 900]

        plateau_start = None
        for i, r in enumerate(stable_results):
            if abs(r['sharpe'] - no_cap['sharpe']) / max(abs(no_cap['sharpe']), 0.01) < 0.03:
                plateau_start = r['mult']
                break

        reco_mult = None
        for r in stable_results:
            if r['mult'] > sl_atr_mult and r['cap_pct'] < 5.0:
                if plateau_start and r['mult'] >= plateau_start:
                    reco_mult = r['mult']
                    break
        if reco_mult is None and stable_results:
            for r in stable_results:
                if r['mult'] > sl_atr_mult and r['cap_pct'] < 10.0:
                    reco_mult = r['mult']
                    break

        print(f"\n  Plateau starts at: {plateau_start}x ATR" if plateau_start else
              "\n  No clear plateau found", flush=True)
        print(f"  Recommended mult: {reco_mult}x ATR" if reco_mult else
              "  No suitable multiplier found (consider disabling cap)", flush=True)
        print(f"  NoCap Sharpe: {no_cap['sharpe']:.3f}", flush=True)

        all_results[strat_name] = {
            'lot': lot,
            'sl_atr': sl_atr_mult,
            'current_cap': current_cap,
            'current_cap_atr': round(current_cap_atr, 2),
            'grid': results,
            'plateau_start': plateau_start,
            'recommended_mult': reco_mult,
            'no_cap_sharpe': no_cap['sharpe'],
        }

    return all_results


# ═══════════════════════════════════════════════════════════════
# Phase 2: Checklist validation
# ═══════════════════════════════════════════════════════════════

def run_checklist(all_results, mean_atr):
    print(f"\n\n{'=' * 90}", flush=True)
    print(f"  DYNAMIC CAP VERIFICATION CHECKLIST", flush=True)
    print(f"{'=' * 90}", flush=True)

    unified_candidates = set(CAP_MULTIPLES) - {999}
    checklist = {}

    for strat_name, data in all_results.items():
        sl = data['sl_atr']
        reco = data['recommended_mult']
        no_cap_sh = data['no_cap_sharpe']
        grid = data['grid']

        grid_map = {r['mult']: r for r in grid}

        checks = {}

        # Check 1: cap_mult > SL_mult
        c1 = reco is not None and reco > sl
        checks['cap_gt_sl'] = c1

        # Check 2: plateau (Sharpe delta < 3% over [reco, reco+2] or so)
        if reco:
            neighbors = [r for r in grid if r['mult'] >= reco and r['mult'] < 900]
            if len(neighbors) >= 2:
                sharpes = [r['sharpe'] for r in neighbors[:3]]
                sh_range = max(sharpes) - min(sharpes)
                c2 = sh_range / max(abs(no_cap_sh), 0.01) < 0.03
            else:
                c2 = False
        else:
            c2 = False
        checks['plateau_stable'] = c2

        # Check 3: cap_pct < 5%
        if reco and reco in grid_map:
            c3 = grid_map[reco]['cap_pct'] < 5.0
        else:
            c3 = False
        checks['cap_rate_low'] = c3

        # Check 4: NoCap doesn't hurt
        if reco and reco in grid_map:
            reco_sh = grid_map[reco]['sharpe']
            c4 = no_cap_sh >= reco_sh * 0.97
        else:
            c4 = True
        checks['nocap_safe'] = c4

        all_pass = all(checks.values())
        checks['all_pass'] = all_pass
        checklist[strat_name] = checks

        cap_dollar = data['lot'] * PV * reco * mean_atr if reco else None

        print(f"\n  {strat_name} (SL={sl}x, recommended Cap={reco}x"
              f"{f' = ${cap_dollar:.0f}' if cap_dollar else ''}):", flush=True)
        print(f"    1. Cap > SL:       {'PASS' if c1 else 'FAIL'} ({reco}x vs {sl}x)", flush=True)
        print(f"    2. Plateau stable: {'PASS' if c2 else 'FAIL'}", flush=True)
        if reco and reco in grid_map:
            print(f"    3. Cap rate < 5%:  {'PASS' if c3 else 'FAIL'} ({grid_map[reco]['cap_pct']:.1f}%)", flush=True)
        else:
            print(f"    3. Cap rate < 5%:  FAIL (no recommendation)", flush=True)
        print(f"    4. NoCap safe:     {'PASS' if c4 else 'FAIL'} (NoCap Sharpe={no_cap_sh:.3f})", flush=True)
        print(f"    => {'ALL PASS' if all_pass else 'NEEDS REVIEW'}", flush=True)

        if reco:
            for m in unified_candidates.copy():
                if m <= sl:
                    unified_candidates.discard(m)
                elif m in grid_map and grid_map[m]['cap_pct'] >= 5.0:
                    unified_candidates.discard(m)

    # Check unified multiplier
    print(f"\n  Unified candidates (pass all strategies): "
          f"{sorted(unified_candidates) if unified_candidates else 'NONE'}", flush=True)

    if unified_candidates:
        best_unified = min(sorted(unified_candidates))
        print(f"  Best unified multiplier: {best_unified}x ATR", flush=True)
        for strat_name, data in all_results.items():
            grid_map = {r['mult']: r for r in data['grid']}
            if best_unified in grid_map:
                r = grid_map[best_unified]
                cap_d = data['lot'] * PV * best_unified * mean_atr
                print(f"    {strat_name}: Sharpe={r['sharpe']:.3f} Cap%={r['cap_pct']:.1f}% "
                      f"Cap${cap_d:.0f}", flush=True)
    else:
        print(f"  => Per-strategy configuration needed", flush=True)

    return checklist


# ═══════════════════════════════════════════════════════════════
# Phase 3: K-Fold validation on recommended mults
# ═══════════════════════════════════════════════════════════════

def run_kfold(h1_df, all_results, mean_atr):
    print(f"\n\n{'=' * 90}", flush=True)
    print(f"  PHASE 3: K-FOLD VALIDATION", flush=True)
    print(f"{'=' * 90}", flush=True)

    kfold_results = {}

    for strat_name, data in all_results.items():
        bt_fn = STRAT_BT[strat_name]
        lot = data['lot']
        reco = data['recommended_mult']

        test_mults = set()
        if reco:
            test_mults.add(reco)
        test_mults.add(999)
        cur_atr = data['current_cap_atr']
        if cur_atr > 0:
            closest = min(CAP_MULTIPLES, key=lambda x: abs(x - cur_atr) if x < 900 else 9999)
            if closest < 900:
                test_mults.add(closest)

        print(f"\n  {strat_name}:", flush=True)
        strat_kfold = {}

        for mult in sorted(test_mults):
            mult_label = f"{mult:.1f}x" if mult < 900 else "NoCap"
            fold_sharpes = []
            fold_caps = []

            for fold_name, start, end in FOLDS:
                fold_data = h1_df[start:end]
                if len(fold_data) < 200:
                    fold_sharpes.append(0); fold_caps.append(0)
                    continue
                trades = bt_fn(fold_data, spread=SPREAD, lot=lot, cap_atr_mult=mult)
                stats = _compute_stats(trades)
                fold_sharpes.append(stats['sharpe'])
                fold_caps.append(stats['cap_pct'])

            positive = sum(1 for s in fold_sharpes if s > 0)
            mean_sh = float(np.mean(fold_sharpes))
            min_sh = float(min(fold_sharpes))
            mean_cap = float(np.mean(fold_caps))
            status = "PASS" if positive >= 4 else "FAIL"

            strat_kfold[mult_label] = {
                'mult': mult,
                'positive_folds': positive,
                'mean_sharpe': round(mean_sh, 3),
                'min_sharpe': round(min_sh, 3),
                'mean_cap_pct': round(mean_cap, 1),
                'pass': positive >= 4,
                'fold_sharpes': [round(s, 2) for s in fold_sharpes],
            }

            print(f"    {mult_label:<8}: {positive}/6 pos, mean={mean_sh:.3f}, "
                  f"min={min_sh:.3f}, avgCap%={mean_cap:.1f}%  [{status}]  "
                  f"{[round(s, 1) for s in fold_sharpes]}", flush=True)

        kfold_results[strat_name] = strat_kfold

    return kfold_results


# ═══════════════════════════════════════════════════════════════
# Phase 4: Recent era test (2025-2026)
# ═══════════════════════════════════════════════════════════════

def run_recent_era(h1_df, all_results, mean_atr):
    print(f"\n\n{'=' * 90}", flush=True)
    print(f"  PHASE 4: RECENT ERA TEST (2025-2026)", flush=True)
    print(f"{'=' * 90}", flush=True)

    recent = h1_df['2025-01-01':]
    recent_atr = compute_atr(recent).dropna().mean()
    print(f"  Recent H1 bars: {len(recent)} ({recent.index[0]} ~ {recent.index[-1]})", flush=True)
    print(f"  Recent mean ATR: ${recent_atr:.2f} (full: ${mean_atr:.2f})", flush=True)

    recent_results = {}

    for strat_name, data in all_results.items():
        bt_fn = STRAT_BT[strat_name]
        lot = data['lot']
        reco = data['recommended_mult']

        test_mults = [999]
        if reco:
            test_mults.insert(0, reco)

        cur_atr = data['current_cap_atr']
        if cur_atr > 0:
            closest = min([m for m in CAP_MULTIPLES if m < 900], key=lambda x: abs(x - cur_atr))
            if closest not in test_mults:
                test_mults.insert(0, closest)

        print(f"\n  {strat_name}:", flush=True)
        strat_recent = {}
        for mult in test_mults:
            trades = bt_fn(recent, spread=SPREAD, lot=lot, cap_atr_mult=mult)
            stats = _compute_stats(trades)
            mult_label = f"{mult:.1f}x" if mult < 900 else "NoCap"
            cap_d = lot * PV * mult * recent_atr if mult < 900 else float('inf')

            strat_recent[mult_label] = stats
            cap_d_str = f"${cap_d:.0f}" if mult < 900 else "inf"
            print(f"    {mult_label:<8}: n={stats['n']:>4}  Sharpe={stats['sharpe']:>7.3f}  "
                  f"PnL={fmt(stats['pnl'])}  Cap%={stats['cap_pct']:.1f}%  "
                  f"Cap$={cap_d_str}", flush=True)

        recent_results[strat_name] = strat_recent

    return recent_results


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 90, flush=True)
    print("  R180 — Dynamic ATR-Multiple Cap Verification", flush=True)
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("=" * 90, flush=True)

    h1_df = load_h1()

    atr_series = compute_atr(h1_df).dropna()
    mean_atr = float(atr_series.mean())
    recent_atr = float(atr_series.iloc[-2000:].mean())
    print(f"  Full mean ATR: ${mean_atr:.2f}", flush=True)
    print(f"  Recent ATR (~3mo): ${recent_atr:.2f}", flush=True)

    # Phase 1
    print(f"\n\n{'#' * 90}", flush=True)
    print(f"  PHASE 1: FULL-HISTORY ATR-MULTIPLE GRID", flush=True)
    print(f"{'#' * 90}", flush=True)
    grid_results = run_grid(h1_df, mean_atr)

    # Phase 2
    checklist = run_checklist(grid_results, mean_atr)

    # Phase 3
    kfold_results = run_kfold(h1_df, grid_results, mean_atr)

    # Phase 4
    recent_results = run_recent_era(h1_df, grid_results, mean_atr)

    # Final summary
    elapsed = time.time() - t0
    print(f"\n\n{'=' * 90}", flush=True)
    print(f"  FINAL SUMMARY", flush=True)
    print(f"{'=' * 90}", flush=True)

    print(f"\n  {'Strategy':<14} {'Lot':>5} {'SL_x':>5} {'Cur_Cap$':>8} {'Cur_x':>6} "
          f"{'Reco_x':>6} {'Reco$':>7} {'NoCap_Sh':>9} {'Reco_Sh':>8} {'Cap%':>5} "
          f"{'Plateau':>8} {'Check':>6}", flush=True)
    print(f"  {'-'*14} {'-'*5} {'-'*5} {'-'*8} {'-'*6} "
          f"{'-'*6} {'-'*7} {'-'*9} {'-'*8} {'-'*5} "
          f"{'-'*8} {'-'*6}", flush=True)

    for strat_name in ['L8_MAX', 'PSAR', 'TSMOM', 'SESS_BO', 'DUAL_THRUST', 'CHANDELIER']:
        d = grid_results[strat_name]
        lot = d['lot']
        sl = d['sl_atr']
        cur = d['current_cap']
        cur_x = d['current_cap_atr']
        reco = d['recommended_mult']
        no_cap_sh = d['no_cap_sharpe']
        grid_map = {r['mult']: r for r in d['grid']}
        reco_sh = grid_map[reco]['sharpe'] if reco and reco in grid_map else 0
        reco_cap_pct = grid_map[reco]['cap_pct'] if reco and reco in grid_map else 0
        reco_dollar = lot * PV * reco * mean_atr if reco else 0
        plateau = d['plateau_start']
        chk = checklist.get(strat_name, {})
        chk_str = "PASS" if chk.get('all_pass') else "FAIL"

        reco_str = f"{reco:.0f}x" if reco else "N/A"
        reco_d_str = f"${reco_dollar:.0f}" if reco else "N/A"
        cur_str = f"${cur}" if cur > 0 else "None"
        cur_x_str = f"{cur_x:.1f}x" if cur_x > 0 else "N/A"
        plat_str = f"{plateau:.0f}x" if plateau else "N/A"

        print(f"  {strat_name:<14} {lot:>5.2f} {sl:>4.1f}x {cur_str:>8} {cur_x_str:>6} "
              f"{reco_str:>6} {reco_d_str:>7} {no_cap_sh:>9.3f} {reco_sh:>8.3f} "
              f"{reco_cap_pct:>4.1f}% {plat_str:>8} {chk_str:>6}", flush=True)

    print(f"\n  Runtime: {elapsed:.0f}s ({elapsed / 60:.1f}min)", flush=True)
    print(f"{'=' * 90}", flush=True)

    # Save
    output = {
        'mean_atr': mean_atr,
        'recent_atr': recent_atr,
        'grid_results': grid_results,
        'checklist': checklist,
        'kfold_results': kfold_results,
        'recent_era_results': recent_results,
        'runtime_seconds': round(elapsed, 1),
    }
    out_path = OUTPUT_DIR / "r180_results.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()

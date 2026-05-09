#!/usr/bin/env python3
"""
R171 — P6 Portfolio Monte Carlo Stress Test
=============================================
Comprehensive stress testing of the 6-strategy P6 portfolio.

Phase 1: Single-Strategy Baseline (6 strategies at unit lot)
Phase 2: Parameter Perturbation MC (1000 sims)
Phase 3: Strategy Dropout Analysis
Phase 4: Conditional Correlation Stress Test
Phase 5: Bootstrap Confidence Intervals
Phase 6: Worst-Case Scenarios

Target runtime: 2-4 hours on 208-core server.
"""
import sys, os, io, time, json, glob, warnings, copy
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor, as_completed

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("results/r171_p6_monte_carlo")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100
SPREAD = 0.30
UNIT_LOT = 0.01

P6_LOTS = {
    'L8_MAX': 0.01, 'PSAR': 0.03, 'TSMOM': 0.04,
    'SESS_BO': 0.04, 'DUAL_THRUST': 0.04, 'CHANDELIER': 0.08,
}
P6_CAPS = {
    'L8_MAX': 35, 'PSAR': 5, 'TSMOM': 0, 'SESS_BO': 35,
    'DUAL_THRUST': 35, 'CHANDELIER': 35,
}
STRAT_ORDER = ['L8_MAX', 'PSAR', 'TSMOM', 'SESS_BO', 'DUAL_THRUST', 'CHANDELIER']

MC_N_SIMS = 1000
MC_PERTURB_PCT = 0.15
BOOTSTRAP_N_RESAMPLES = 5000
BOOTSTRAP_BLOCK_DAYS = 30

BASE_PARAMS = {
    'L8_MAX': {
        'sl_atr_mult': 3.5, 'tp_atr_mult': 8.0,
        'trailing_activate_atr': 0.14, 'trailing_distance_atr': 0.025,
        'keltner_adx_threshold': 14, 'choppy_threshold': 0.50,
        'keltner_max_hold_m15': 20,
    },
    'PSAR': {
        'sl_atr': 4.5, 'tp_atr': 16.0,
        'trail_act': 0.20, 'trail_dist': 0.04,
        'max_hold': 20, 'af_start': 0.01, 'af_max': 0.05,
    },
    'TSMOM': {
        'sl_atr': 4.5, 'tp_atr': 6.0,
        'trail_act': 0.14, 'trail_dist': 0.025,
        'max_hold': 20, 'fast': 480, 'slow': 720,
    },
    'SESS_BO': {
        'sl_atr': 4.5, 'tp_atr': 4.0,
        'trail_act': 0.14, 'trail_dist': 0.025,
        'max_hold': 20, 'session_hour': 12, 'lookback': 4,
    },
    'DUAL_THRUST': {
        'sl_atr': 4.5, 'tp_atr': 8.0,
        'trail_act': 0.14, 'trail_dist': 0.025,
        'max_hold': 20, 'n_bars': 6, 'k': 0.5,
    },
    'CHANDELIER': {
        'sl_atr': 4.5, 'tp_atr': 8.0,
        'trail_act': 0.14, 'trail_dist': 0.025,
        'max_hold': 20, 'period': 22, 'mult': 3.0, 'ema_period': 100,
    },
}

INT_PARAMS = {'keltner_max_hold_m15', 'max_hold', 'session_hour', 'lookback',
              'n_bars', 'period', 'ema_period', 'fast', 'slow'}


def fmt(v):
    return f"${v:>10,.0f}" if v >= 0 else f"-${abs(v):>9,.0f}"


def save_json(data, filename):
    out_path = OUTPUT_DIR / filename
    with open(out_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved: {out_path}", flush=True)


# ===================================================================
# Data Loading
# ===================================================================

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


# ===================================================================
# Shared Helpers
# ===================================================================

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
        prev = psar[i - 1]
        if direction[i - 1] == 1:
            psar[i] = prev + af * (ep - prev)
            psar[i] = min(psar[i], df['Low'].iloc[i - 1], df['Low'].iloc[max(0, i - 2)])
            if df['Low'].iloc[i] < psar[i]:
                direction[i] = -1
                psar[i] = ep
                ep = df['Low'].iloc[i]
                af = af_start
            else:
                direction[i] = 1
                if df['High'].iloc[i] > ep:
                    ep = df['High'].iloc[i]
                    af = min(af + af_start, af_max)
        else:
            psar[i] = prev + af * (ep - prev)
            psar[i] = max(psar[i], df['High'].iloc[i - 1], df['High'].iloc[max(0, i - 2)])
            if df['High'].iloc[i] > psar[i]:
                direction[i] = 1
                psar[i] = ep
                ep = df['High'].iloc[i]
                af = af_start
            else:
                direction[i] = -1
                if df['Low'].iloc[i] < ep:
                    ep = df['Low'].iloc[i]
                    af = min(af + af_start, af_max)
    df['PSAR_dir'] = direction
    df['ATR'] = compute_atr(df)
    return df


def _mk(pos, exit_p, exit_time, reason, bar_idx, pnl):
    return {
        'dir': pos['dir'], 'entry': pos['entry'], 'exit': exit_p,
        'entry_time': pos['time'], 'exit_time': exit_time,
        'pnl': pnl, 'reason': reason, 'bars': bar_idx - pos['bar'],
    }


def _run_exit_with_cap(pos, i, h, lo_v, c, spread, lot, pv, times,
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


# ===================================================================
# H1 Strategy Backtests
# ===================================================================

def bt_psar(h1_df, spread, lot, maxloss_cap=0, **kw):
    sl_atr = kw.get('sl_atr', 4.5)
    tp_atr = kw.get('tp_atr', 16.0)
    trail_act = kw.get('trail_act', 0.20)
    trail_dist = kw.get('trail_dist', 0.04)
    max_hold = int(kw.get('max_hold', 20))
    af_start = kw.get('af_start', 0.01)
    af_max = kw.get('af_max', 0.05)

    df = add_psar(h1_df, af_start=af_start, af_max=af_max).dropna(subset=['PSAR_dir', 'ATR'])
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
        if i - last_exit < 2:
            continue
        if np.isnan(atr[i]) or atr[i] < 0.1:
            continue
        if pdir[i - 1] == -1 and pdir[i] == 1:
            pos = {'dir': 'BUY', 'entry': c[i] + spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif pdir[i - 1] == 1 and pdir[i] == -1:
            pos = {'dir': 'SELL', 'entry': c[i] - spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_tsmom(h1_df, spread, lot, maxloss_cap=0, **kw):
    fast = int(kw.get('fast', 480))
    slow = int(kw.get('slow', 720))
    sl_atr = kw.get('sl_atr', 4.5)
    tp_atr = kw.get('tp_atr', 6.0)
    trail_act = kw.get('trail_act', 0.14)
    trail_dist = kw.get('trail_dist', 0.025)
    max_hold = int(kw.get('max_hold', 20))

    df = h1_df.copy()
    df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; times = df.index; n = len(df)
    max_lb = max(fast, slow)
    score = np.full(n, np.nan)
    for i in range(max_lb, n):
        s = 0.0
        if c[i - fast] > 0:
            s += 0.5 * np.sign(c[i] / c[i - fast] - 1.0)
        if c[i - slow] > 0:
            s += 0.5 * np.sign(c[i] / c[i - slow] - 1.0)
        score[i] = s
    trades = []; pos = None; last_exit = -999
    for i in range(max_lb + 1, n):
        if pos is not None:
            result = _run_exit_with_cap(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                                        sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            if pos['dir'] == 'BUY' and score[i] < 0:
                pnl = (c[i] - pos['entry'] - spread) * lot * PV
                trades.append(_mk(pos, c[i], times[i], "Reversal", i, pnl)); pos = None; last_exit = i; continue
            elif pos['dir'] == 'SELL' and score[i] > 0:
                pnl = (pos['entry'] - c[i] - spread) * lot * PV
                trades.append(_mk(pos, c[i], times[i], "Reversal", i, pnl)); pos = None; last_exit = i; continue
            continue
        if i - last_exit < 2:
            continue
        if np.isnan(atr[i]) or atr[i] < 0.1:
            continue
        if np.isnan(score[i]) or np.isnan(score[i - 1]):
            continue
        if score[i] > 0 and score[i - 1] <= 0:
            pos = {'dir': 'BUY', 'entry': c[i] + spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif score[i] < 0 and score[i - 1] >= 0:
            pos = {'dir': 'SELL', 'entry': c[i] - spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_sess_bo(h1_df, spread, lot, maxloss_cap=0, **kw):
    session_hour = int(kw.get('session_hour', 12))
    lookback = int(kw.get('lookback', 4))
    sl_atr = kw.get('sl_atr', 4.5)
    tp_atr = kw.get('tp_atr', 4.0)
    trail_act = kw.get('trail_act', 0.14)
    trail_dist = kw.get('trail_dist', 0.025)
    max_hold = int(kw.get('max_hold', 20))

    df = h1_df.copy()
    df['ATR'] = compute_atr(df)
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
        if hours[i] != session_hour:
            continue
        if i - last_exit < 2:
            continue
        if np.isnan(atr[i]) or atr[i] < 0.1:
            continue
        hh = max(h[i - j] for j in range(1, lookback + 1))
        ll = min(lo[i - j] for j in range(1, lookback + 1))
        if c[i] > hh:
            pos = {'dir': 'BUY', 'entry': c[i] + spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif c[i] < ll:
            pos = {'dir': 'SELL', 'entry': c[i] - spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_dual_thrust(h1_df, spread, lot, maxloss_cap=35, **kw):
    n_bars = int(kw.get('n_bars', 6))
    k = kw.get('k', 0.5)
    sl_atr = kw.get('sl_atr', 4.5)
    tp_atr = kw.get('tp_atr', 8.0)
    trail_act = kw.get('trail_act', 0.14)
    trail_dist = kw.get('trail_dist', 0.025)
    max_hold = int(kw.get('max_hold', 20))

    df = h1_df.copy()
    df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    o = df['Open'].values; atr = df['ATR'].values
    times = df.index; n = len(df)
    trades = []; pos = None; last_exit = -999
    for i in range(n_bars, n):
        if pos is not None:
            result = _run_exit_with_cap(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                                        sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if i - last_exit < 2:
            continue
        if np.isnan(atr[i]) or atr[i] < 0.1:
            continue
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


def bt_chandelier(h1_df, spread, lot, maxloss_cap=35, **kw):
    period = int(kw.get('period', 22))
    mult = kw.get('mult', 3.0)
    ema_period = int(kw.get('ema_period', 100))
    sl_atr = kw.get('sl_atr', 4.5)
    tp_atr = kw.get('tp_atr', 8.0)
    trail_act = kw.get('trail_act', 0.14)
    trail_dist = kw.get('trail_dist', 0.025)
    max_hold = int(kw.get('max_hold', 20))

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
            direction[i] = direction[i - 1]
            continue
        if c[i] > chandelier_short[i - 1]:
            direction[i] = 1
        elif c[i] < chandelier_long[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]

    trades = []; pos = None; last_exit = -999
    for i in range(period + 2, n):
        if pos is not None:
            result = _run_exit_with_cap(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                                        sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if i - last_exit < 2:
            continue
        if np.isnan(atr[i]) or atr[i] < 0.1:
            continue
        flipped_long = direction[i] == 1 and direction[i - 1] != 1
        flipped_short = direction[i] == -1 and direction[i - 1] != -1
        if flipped_long and c[i] > ema[i]:
            pos = {'dir': 'BUY', 'entry': c[i] + spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif flipped_short and c[i] < ema[i]:
            pos = {'dir': 'SELL', 'entry': c[i] - spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_l8_max(data_bundle, spread, lot, maxloss_cap=35):
    from backtest.runner import run_variant, LIVE_PARITY_KWARGS
    kw = {
        **LIVE_PARITY_KWARGS,
        'maxloss_cap': maxloss_cap,
        'spread_cost': spread,
        'initial_capital': 2000,
        'min_lot_size': lot,
        'max_lot_size': lot,
    }
    result = run_variant(data_bundle, "L8_MAX", verbose=False, **kw)
    raw_trades = result.get('_trades', [])
    trades = []
    for t in raw_trades:
        trades.append({
            'dir': t.direction, 'entry': t.entry_price, 'exit': t.exit_price,
            'entry_time': t.entry_time, 'exit_time': t.exit_time,
            'pnl': t.pnl, 'reason': t.exit_reason, 'bars': 0,
        })
    return trades


H1_STRAT_FUNCS = {
    'PSAR': bt_psar,
    'TSMOM': bt_tsmom,
    'SESS_BO': bt_sess_bo,
    'DUAL_THRUST': bt_dual_thrust,
    'CHANDELIER': bt_chandelier,
}


# ===================================================================
# Daily PnL Helpers
# ===================================================================

def trades_to_daily_series(trades):
    if not trades:
        return pd.Series(dtype=float)
    daily = {}
    for t in trades:
        d = pd.Timestamp(t['exit_time']).date()
        daily[d] = daily.get(d, 0) + t['pnl']
    dates = sorted(daily.keys())
    return pd.Series([daily[d] for d in dates], index=pd.DatetimeIndex(dates))


def sharpe(arr):
    if len(arr) < 10:
        return 0.0
    s = np.std(arr, ddof=1)
    return float(np.mean(arr) / s * np.sqrt(252)) if s > 0 else 0.0


def max_dd(arr):
    if len(arr) == 0:
        return 0.0
    eq = np.cumsum(arr)
    return float((np.maximum.accumulate(eq) - eq).max())


def cvar(arr, pct=5):
    if len(arr) < 20:
        return 0.0
    threshold = np.percentile(arr, pct)
    tail = arr[arr <= threshold]
    return float(tail.mean()) if len(tail) > 0 else float(threshold)


def win_rate(arr):
    if len(arr) == 0:
        return 0.0
    return float(np.sum(arr > 0) / len(arr) * 100)


def build_portfolio_daily(unit_dailies, lots, union_idx=None):
    if union_idx is None:
        all_dates = set()
        for ds in unit_dailies.values():
            all_dates.update(ds.index)
        union_idx = pd.DatetimeIndex(sorted(all_dates))
    portfolio = np.zeros(len(union_idx))
    for name in STRAT_ORDER:
        if name not in unit_dailies:
            continue
        ds = unit_dailies[name]
        multiplier = lots.get(name, UNIT_LOT) / UNIT_LOT
        aligned = ds.reindex(union_idx, fill_value=0.0).values * multiplier
        portfolio += aligned
    return portfolio, union_idx


def strat_stats_from_trades(trades):
    n = len(trades)
    if n == 0:
        return {'n_trades': 0, 'pnl': 0.0, 'sharpe': 0.0, 'max_dd': 0.0, 'wr': 0.0}
    pnls = [t['pnl'] for t in trades]
    daily = trades_to_daily_series(trades)
    daily_arr = daily.values if len(daily) > 0 else np.array([0.0])
    wins = sum(1 for p in pnls if p > 0)
    return {
        'n_trades': n,
        'pnl': round(sum(pnls), 2),
        'sharpe': round(sharpe(daily_arr), 2),
        'max_dd': round(max_dd(daily_arr), 2),
        'wr': round(wins / n * 100, 1),
    }


# ===================================================================
# Parameter Perturbation
# ===================================================================

def perturb_params(base_params, rng, pct=0.15):
    """Perturb all parameters by uniform +-pct."""
    perturbed = {}
    for name, params in base_params.items():
        p = {}
        for k, v in params.items():
            factor = 1.0 + rng.uniform(-pct, pct)
            new_val = v * factor
            if k in INT_PARAMS:
                new_val = max(1, int(round(new_val)))
            else:
                new_val = round(new_val, 6)
            p[k] = new_val
        perturbed[name] = p
    return perturbed


# ===================================================================
# MC Worker (runs in subprocess)
# ===================================================================

def _mc_worker(args):
    """Single MC simulation: perturb H1 strats, scale L8 PnL, combine portfolio."""
    sim_id, seed, h1_df_bytes, l8_daily_values, l8_daily_index_bytes, base_params_copy = args

    h1_df = pd.read_pickle(io.BytesIO(h1_df_bytes))
    l8_daily_index = pd.DatetimeIndex(pd.read_pickle(io.BytesIO(l8_daily_index_bytes)))
    l8_daily = pd.Series(l8_daily_values, index=l8_daily_index)

    rng = np.random.default_rng(seed)
    params = perturb_params(base_params_copy, rng, pct=MC_PERTURB_PCT)

    unit_dailies = {}
    strat_metrics = {}

    for name, fn in H1_STRAT_FUNCS.items():
        try:
            kw = params[name]
            cap = P6_CAPS[name]
            trades = fn(h1_df, spread=SPREAD, lot=UNIT_LOT, maxloss_cap=cap, **kw)
            ds = trades_to_daily_series(trades)
            unit_dailies[name] = ds
            daily_arr = ds.values if len(ds) > 0 else np.array([0.0])
            strat_metrics[name] = {
                'sharpe': round(sharpe(daily_arr), 3),
                'pnl': round(float(np.sum(daily_arr)), 2),
                'n_trades': len(trades),
            }
        except Exception as e:
            unit_dailies[name] = pd.Series(dtype=float)
            strat_metrics[name] = {'sharpe': 0.0, 'pnl': 0.0, 'n_trades': 0, 'error': str(e)}

    l8_scale_factor = 1.0 + rng.uniform(-MC_PERTURB_PCT, MC_PERTURB_PCT)
    unit_dailies['L8_MAX'] = l8_daily * l8_scale_factor
    strat_metrics['L8_MAX'] = {
        'sharpe': round(sharpe((l8_daily * l8_scale_factor).values), 3),
        'pnl': round(float((l8_daily * l8_scale_factor).sum()), 2),
        'scale_factor': round(l8_scale_factor, 4),
    }

    port_arr, _ = build_portfolio_daily(unit_dailies, P6_LOTS)
    port_sharpe = sharpe(port_arr)
    port_pnl = float(np.sum(port_arr))
    port_dd = max_dd(port_arr)

    return {
        'sim_id': sim_id,
        'port_sharpe': round(port_sharpe, 3),
        'port_pnl': round(port_pnl, 2),
        'port_max_dd': round(port_dd, 2),
        'strat_metrics': strat_metrics,
        'params': {k: {pk: round(pv, 6) if isinstance(pv, float) else pv
                        for pk, pv in v.items()} for k, v in params.items()},
    }


# ===================================================================
# Phase Functions
# ===================================================================

def phase1_baseline(h1_df, data_bundle):
    """Phase 1: Run each strategy individually at unit lot."""
    print(f"\n{'=' * 80}", flush=True)
    print(f"  Phase 1: Single-Strategy Baseline (unit lot = {UNIT_LOT})", flush=True)
    print(f"{'=' * 80}\n", flush=True)

    unit_trades = {}
    unit_dailies = {}
    unit_stats = {}

    for name, fn in H1_STRAT_FUNCS.items():
        cap = P6_CAPS[name]
        kw = BASE_PARAMS[name]
        trades = fn(h1_df, spread=SPREAD, lot=UNIT_LOT, maxloss_cap=cap, **kw)
        unit_trades[name] = trades
        unit_dailies[name] = trades_to_daily_series(trades)
        unit_stats[name] = strat_stats_from_trades(trades)
        unit_stats[name]['cap'] = cap
        s = unit_stats[name]
        print(f"    {name:>12}: {s['n_trades']} trades, Sharpe={s['sharpe']:.2f}, "
              f"PnL={fmt(s['pnl'])}, MaxDD={fmt(s['max_dd'])}, WR={s['wr']:.1f}%", flush=True)

    cap = P6_CAPS['L8_MAX']
    trades = bt_l8_max(data_bundle, spread=SPREAD, lot=UNIT_LOT, maxloss_cap=cap)
    unit_trades['L8_MAX'] = trades
    unit_dailies['L8_MAX'] = trades_to_daily_series(trades)
    unit_stats['L8_MAX'] = strat_stats_from_trades(trades)
    unit_stats['L8_MAX']['cap'] = cap
    s = unit_stats['L8_MAX']
    print(f"    {'L8_MAX':>12}: {s['n_trades']} trades, Sharpe={s['sharpe']:.2f}, "
          f"PnL={fmt(s['pnl'])}, MaxDD={fmt(s['max_dd'])}, WR={s['wr']:.1f}%", flush=True)

    # Correlation matrix
    all_dates = set()
    for ds in unit_dailies.values():
        all_dates.update(ds.index)
    union_idx = pd.DatetimeIndex(sorted(all_dates))
    corr_df = pd.DataFrame(index=union_idx)
    for name in STRAT_ORDER:
        corr_df[name] = unit_dailies.get(name, pd.Series(dtype=float)).reindex(union_idx, fill_value=0.0)
    corr_matrix = corr_df.corr()

    print(f"\n  6x6 Daily PnL Correlation Matrix:", flush=True)
    print(f"  {'':>12}", end="", flush=True)
    for name in STRAT_ORDER:
        print(f" {name:>12}", end="", flush=True)
    print("", flush=True)
    for row in STRAT_ORDER:
        print(f"  {row:>12}", end="", flush=True)
        for col in STRAT_ORDER:
            print(f" {corr_matrix.loc[row, col]:>12.3f}", end="", flush=True)
        print("", flush=True)

    # P6 Portfolio baseline
    port_arr, _ = build_portfolio_daily(unit_dailies, P6_LOTS, union_idx)
    port_sharpe_val = sharpe(port_arr)
    port_pnl_val = float(np.sum(port_arr))
    port_dd_val = max_dd(port_arr)
    print(f"\n  P6 Portfolio (baseline lots): Sharpe={port_sharpe_val:.3f}, "
          f"PnL={fmt(port_pnl_val)}, MaxDD={fmt(port_dd_val)}", flush=True)

    corr_json = {}
    for r in STRAT_ORDER:
        corr_json[r] = {c_name: round(float(corr_matrix.loc[r, c_name]), 4) for c_name in STRAT_ORDER}

    results = {
        'unit_stats': unit_stats,
        'correlation': corr_json,
        'p6_portfolio': {
            'sharpe': round(port_sharpe_val, 3),
            'pnl': round(port_pnl_val, 2),
            'max_dd': round(port_dd_val, 2),
            'lots': P6_LOTS,
        },
    }
    save_json(results, 'phase1_baseline.json')
    return unit_trades, unit_dailies, unit_stats, union_idx


def phase2_monte_carlo(h1_df, unit_dailies):
    """Phase 2: Parameter Perturbation MC (1000 sims)."""
    print(f"\n{'=' * 80}", flush=True)
    print(f"  Phase 2: Parameter Perturbation Monte Carlo ({MC_N_SIMS} simulations)", flush=True)
    print(f"  Perturbation: +/-{MC_PERTURB_PCT*100:.0f}% on all strategy parameters", flush=True)
    print(f"{'=' * 80}\n", flush=True)

    buf = io.BytesIO()
    h1_df.to_pickle(buf)
    h1_bytes = buf.getvalue()

    l8_daily = unit_dailies.get('L8_MAX', pd.Series(dtype=float))
    buf2 = io.BytesIO()
    l8_daily.index.to_series().to_pickle(buf2)
    l8_idx_bytes = buf2.getvalue()
    l8_values = l8_daily.values

    base_rng = np.random.default_rng(seed=42)
    seeds = base_rng.integers(0, 2**31, size=MC_N_SIMS)

    base_params_stripped = {}
    for name in STRAT_ORDER:
        if name != 'L8_MAX':
            base_params_stripped[name] = dict(BASE_PARAMS[name])

    tasks = []
    for sim_id in range(MC_N_SIMS):
        tasks.append((sim_id, int(seeds[sim_id]), h1_bytes, l8_values, l8_idx_bytes, base_params_stripped))

    n_workers = max(1, min(os.cpu_count() - 1, 200))
    print(f"  Launching {MC_N_SIMS} simulations across {n_workers} workers...", flush=True)
    t0 = time.time()

    mc_results = []
    completed = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        future_map = {pool.submit(_mc_worker, task): task[0] for task in tasks}
        for future in as_completed(future_map):
            sim_id = future_map[future]
            try:
                result = future.result()
                mc_results.append(result)
            except Exception as e:
                failed += 1
                if failed <= 5:
                    print(f"    [WARN] Sim {sim_id} failed: {e}", flush=True)
            completed += 1
            if completed % 100 == 0 or completed == MC_N_SIMS:
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (MC_N_SIMS - completed) / rate if rate > 0 else 0
                print(f"    Progress: {completed}/{MC_N_SIMS} ({failed} failed) "
                      f"[{elapsed:.0f}s elapsed, ETA {eta:.0f}s]", flush=True)

    total_elapsed = time.time() - t0
    print(f"\n  MC complete: {len(mc_results)} successful / {failed} failed in {total_elapsed:.0f}s", flush=True)

    sharpes = np.array([r['port_sharpe'] for r in mc_results])
    pnls = np.array([r['port_pnl'] for r in mc_results])
    dds = np.array([r['port_max_dd'] for r in mc_results])

    print(f"\n  Portfolio Sharpe Distribution ({len(sharpes)} sims):", flush=True)
    print(f"    Mean:   {np.mean(sharpes):.3f}", flush=True)
    print(f"    Std:    {np.std(sharpes):.3f}", flush=True)
    print(f"    Min:    {np.min(sharpes):.3f}", flush=True)
    print(f"    Max:    {np.max(sharpes):.3f}", flush=True)
    print(f"    Median: {np.median(sharpes):.3f}", flush=True)
    for pct in [5, 10, 25, 50, 75, 90, 95]:
        print(f"    P{pct:02d}:    {np.percentile(sharpes, pct):.3f}", flush=True)

    p_sharpe_gt0 = float(np.mean(sharpes > 0) * 100)
    p_sharpe_gt2 = float(np.mean(sharpes > 2) * 100)
    p_sharpe_gt4 = float(np.mean(sharpes > 4) * 100)
    print(f"\n  P(Sharpe > 0) = {p_sharpe_gt0:.1f}%", flush=True)
    print(f"  P(Sharpe > 2) = {p_sharpe_gt2:.1f}%", flush=True)
    print(f"  P(Sharpe > 4) = {p_sharpe_gt4:.1f}%", flush=True)

    print(f"\n  Portfolio PnL: mean={fmt(np.mean(pnls))}, "
          f"std={fmt(np.std(pnls))}", flush=True)
    print(f"  Portfolio MaxDD: mean={fmt(np.mean(dds))}, "
          f"max={fmt(np.max(dds))}", flush=True)

    results = {
        'n_sims': len(mc_results),
        'n_failed': failed,
        'elapsed_s': round(total_elapsed, 1),
        'sharpe_dist': {
            'mean': round(float(np.mean(sharpes)), 3),
            'std': round(float(np.std(sharpes)), 3),
            'min': round(float(np.min(sharpes)), 3),
            'max': round(float(np.max(sharpes)), 3),
            'median': round(float(np.median(sharpes)), 3),
            'percentiles': {str(p): round(float(np.percentile(sharpes, p)), 3)
                           for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]},
        },
        'pnl_dist': {
            'mean': round(float(np.mean(pnls)), 2),
            'std': round(float(np.std(pnls)), 2),
            'min': round(float(np.min(pnls)), 2),
            'max': round(float(np.max(pnls)), 2),
        },
        'dd_dist': {
            'mean': round(float(np.mean(dds)), 2),
            'std': round(float(np.std(dds)), 2),
            'max': round(float(np.max(dds)), 2),
        },
        'prob_sharpe_gt0': round(p_sharpe_gt0, 1),
        'prob_sharpe_gt2': round(p_sharpe_gt2, 1),
        'prob_sharpe_gt4': round(p_sharpe_gt4, 1),
        'worst_10': sorted(mc_results, key=lambda x: x['port_sharpe'])[:10],
        'best_10': sorted(mc_results, key=lambda x: x['port_sharpe'], reverse=True)[:10],
    }
    save_json(results, 'phase2_monte_carlo.json')
    return mc_results


def phase3_dropout(h1_df, unit_dailies, union_idx):
    """Phase 3: Strategy Dropout Analysis."""
    print(f"\n{'=' * 80}", flush=True)
    print(f"  Phase 3: Strategy Dropout Analysis", flush=True)
    print(f"{'=' * 80}\n", flush=True)

    port_full, _ = build_portfolio_daily(unit_dailies, P6_LOTS, union_idx)
    full_sharpe = sharpe(port_full)
    full_pnl = float(np.sum(port_full))
    print(f"  Full portfolio: Sharpe={full_sharpe:.3f}, PnL={fmt(full_pnl)}", flush=True)

    # Single dropout
    print(f"\n  Single Strategy Dropout:", flush=True)
    print(f"  {'Removed':<14} {'Remaining':>8} {'Sharpe':>8} {'dSharpe':>9} {'PnL':>12} {'Contribution':>14}", flush=True)
    print(f"  {'-'*14} {'-'*8} {'-'*8} {'-'*9} {'-'*12} {'-'*14}", flush=True)

    single_dropout = {}
    for drop_name in STRAT_ORDER:
        remaining_lots = {k: v for k, v in P6_LOTS.items() if k != drop_name}
        remaining_dailies = {k: v for k, v in unit_dailies.items() if k != drop_name}
        port_arr, _ = build_portfolio_daily(remaining_dailies, remaining_lots, union_idx)
        sh = sharpe(port_arr)
        pnl = float(np.sum(port_arr))
        contribution = full_sharpe - sh
        single_dropout[drop_name] = {
            'remaining_sharpe': round(sh, 3),
            'remaining_pnl': round(pnl, 2),
            'sharpe_contribution': round(contribution, 3),
            'pnl_contribution': round(full_pnl - pnl, 2),
        }
        print(f"  {drop_name:<14} {5:>8} {sh:>8.3f} {contribution:>+9.3f} "
              f"{fmt(pnl):>12} {fmt(full_pnl - pnl):>14}", flush=True)

    most_important = max(single_dropout, key=lambda x: single_dropout[x]['sharpe_contribution'])
    least_important = min(single_dropout, key=lambda x: single_dropout[x]['sharpe_contribution'])
    print(f"\n  Most important (removal hurts most):  {most_important} "
          f"(dSharpe={single_dropout[most_important]['sharpe_contribution']:+.3f})", flush=True)
    print(f"  Least important (removal hurts least): {least_important} "
          f"(dSharpe={single_dropout[least_important]['sharpe_contribution']:+.3f})", flush=True)

    # Pair dropout (C(6,2) = 15)
    print(f"\n  Pair Dropout (C(6,2)=15 combos):", flush=True)
    print(f"  {'Removed Pair':<28} {'Rem':>4} {'Sharpe':>8} {'dSharpe':>9} {'PnL':>12}", flush=True)
    print(f"  {'-'*28} {'-'*4} {'-'*8} {'-'*9} {'-'*12}", flush=True)

    pair_dropout = {}
    for drop_pair in combinations(STRAT_ORDER, 2):
        drop_set = set(drop_pair)
        remaining_lots = {k: v for k, v in P6_LOTS.items() if k not in drop_set}
        remaining_dailies = {k: v for k, v in unit_dailies.items() if k not in drop_set}
        port_arr, _ = build_portfolio_daily(remaining_dailies, remaining_lots, union_idx)
        sh = sharpe(port_arr)
        pnl = float(np.sum(port_arr))
        label = f"{drop_pair[0]}+{drop_pair[1]}"
        pair_dropout[label] = {
            'remaining_sharpe': round(sh, 3),
            'remaining_pnl': round(pnl, 2),
            'sharpe_contribution': round(full_sharpe - sh, 3),
        }
        print(f"  {label:<28} {4:>4} {sh:>8.3f} {full_sharpe - sh:>+9.3f} {fmt(pnl):>12}", flush=True)

    results = {
        'full_portfolio': {'sharpe': round(full_sharpe, 3), 'pnl': round(full_pnl, 2)},
        'single_dropout': single_dropout,
        'pair_dropout': pair_dropout,
        'most_important_strategy': most_important,
        'least_important_strategy': least_important,
    }
    save_json(results, 'phase3_dropout.json')
    return results


def phase4_conditional_correlation(unit_dailies, union_idx):
    """Phase 4: Conditional Correlation Stress Test."""
    print(f"\n{'=' * 80}", flush=True)
    print(f"  Phase 4: Conditional Correlation Stress Test", flush=True)
    print(f"{'=' * 80}\n", flush=True)

    corr_df = pd.DataFrame(index=union_idx)
    for name in STRAT_ORDER:
        ds = unit_dailies.get(name, pd.Series(dtype=float))
        corr_df[name] = ds.reindex(union_idx, fill_value=0.0) * (P6_LOTS[name] / UNIT_LOT)

    port_returns = corr_df.sum(axis=1).values

    # Define regimes
    p10 = np.percentile(port_returns, 10)
    p90 = np.percentile(port_returns, 90)

    stress_mask = port_returns <= p10
    calm_mask = (port_returns > p10) & (port_returns <= p90)

    stress_days = corr_df.loc[corr_df.index[stress_mask]]
    calm_days = corr_df.loc[corr_df.index[calm_mask]]

    print(f"  Stress days (bottom 10%): {len(stress_days)} days (port return <= ${p10:.2f})", flush=True)
    print(f"  Calm days (middle 80%):   {len(calm_days)} days", flush=True)

    corr_full = corr_df.corr()
    corr_calm = calm_days.corr() if len(calm_days) > 10 else corr_full
    corr_stress = stress_days.corr() if len(stress_days) > 10 else corr_full

    def _print_corr(label, matrix):
        print(f"\n  {label}:", flush=True)
        print(f"  {'':>12}", end="", flush=True)
        for name in STRAT_ORDER:
            print(f" {name:>12}", end="", flush=True)
        print("", flush=True)
        for row in STRAT_ORDER:
            print(f"  {row:>12}", end="", flush=True)
            for col in STRAT_ORDER:
                val = matrix.loc[row, col] if not pd.isna(matrix.loc[row, col]) else 0.0
                print(f" {val:>12.3f}", end="", flush=True)
            print("", flush=True)

    _print_corr("Full-Period Correlation", corr_full)
    _print_corr("Calm Days Correlation", corr_calm)
    _print_corr("Stress Days Correlation", corr_stress)

    # Correlation inflation factor
    print(f"\n  Correlation Inflation Factor (stress/calm):", flush=True)
    print(f"  {'Pair':<28} {'Calm':>8} {'Stress':>8} {'CIF':>8} {'Flag':>8}", flush=True)
    print(f"  {'-'*28} {'-'*8} {'-'*8} {'-'*8} {'-'*8}", flush=True)

    cif_data = {}
    for i, s1 in enumerate(STRAT_ORDER):
        for s2 in STRAT_ORDER[i + 1:]:
            calm_val = corr_calm.loc[s1, s2] if not pd.isna(corr_calm.loc[s1, s2]) else 0.0
            stress_val = corr_stress.loc[s1, s2] if not pd.isna(corr_stress.loc[s1, s2]) else 0.0
            if abs(calm_val) > 0.01:
                cif_val = stress_val / calm_val
            elif abs(stress_val) > 0.01:
                cif_val = float('inf')
            else:
                cif_val = 1.0
            flag = "DANGER" if cif_val > 2.0 and stress_val > 0.3 else (
                   "WARN" if stress_val > calm_val + 0.2 else "OK")
            pair = f"{s1}+{s2}"
            cif_data[pair] = {
                'calm_corr': round(float(calm_val), 3),
                'stress_corr': round(float(stress_val), 3),
                'cif': round(float(cif_val), 3) if np.isfinite(cif_val) else None,
                'flag': flag,
            }
            cif_str = f"{cif_val:.2f}" if np.isfinite(cif_val) else "inf"
            print(f"  {pair:<28} {calm_val:>8.3f} {stress_val:>8.3f} "
                  f"{cif_str:>8} {flag:>8}", flush=True)

    mean_calm = float(corr_calm.values[np.triu_indices_from(corr_calm.values, k=1)].mean())
    mean_stress = float(corr_stress.values[np.triu_indices_from(corr_stress.values, k=1)].mean())
    print(f"\n  Average pairwise correlation:", flush=True)
    print(f"    Calm:   {mean_calm:.3f}", flush=True)
    print(f"    Stress: {mean_stress:.3f}", flush=True)
    diversification_fails = mean_stress > mean_calm + 0.15
    print(f"    Diversification failure: {'YES' if diversification_fails else 'NO'} "
          f"(stress - calm = {mean_stress - mean_calm:+.3f})", flush=True)

    results = {
        'n_stress_days': int(stress_mask.sum()),
        'n_calm_days': int(calm_mask.sum()),
        'stress_threshold': round(float(p10), 2),
        'correlation_inflation': cif_data,
        'mean_calm_corr': round(mean_calm, 3),
        'mean_stress_corr': round(mean_stress, 3),
        'diversification_failure': diversification_fails,
        'corr_full': {r: {c: round(float(corr_full.loc[r, c]), 4) for c in STRAT_ORDER} for r in STRAT_ORDER},
        'corr_calm': {r: {c: round(float(corr_calm.loc[r, c]), 4) if not pd.isna(corr_calm.loc[r, c]) else 0
                          for c in STRAT_ORDER} for r in STRAT_ORDER},
        'corr_stress': {r: {c: round(float(corr_stress.loc[r, c]), 4) if not pd.isna(corr_stress.loc[r, c]) else 0
                            for c in STRAT_ORDER} for r in STRAT_ORDER},
    }
    save_json(results, 'phase4_conditional_corr.json')
    return results


def phase5_bootstrap(unit_dailies, union_idx):
    """Phase 5: Block Bootstrap Confidence Intervals."""
    print(f"\n{'=' * 80}", flush=True)
    print(f"  Phase 5: Bootstrap Confidence Intervals", flush=True)
    print(f"  {BOOTSTRAP_N_RESAMPLES} resamples, {BOOTSTRAP_BLOCK_DAYS}-day blocks", flush=True)
    print(f"{'=' * 80}\n", flush=True)

    port_arr, _ = build_portfolio_daily(unit_dailies, P6_LOTS, union_idx)
    n_days = len(port_arr)
    block_size = BOOTSTRAP_BLOCK_DAYS
    n_blocks = n_days // block_size

    if n_blocks < 3:
        print(f"  [WARN] Only {n_blocks} blocks, reducing block size to {n_days // 5}", flush=True)
        block_size = max(5, n_days // 5)
        n_blocks = n_days // block_size

    blocks = []
    for b in range(n_blocks):
        start = b * block_size
        end = start + block_size
        blocks.append(port_arr[start:end])

    rng = np.random.default_rng(seed=123)
    boot_sharpes = []
    boot_pnls = []
    boot_dds = []
    boot_cvars = []

    t0 = time.time()
    for i in range(BOOTSTRAP_N_RESAMPLES):
        chosen = rng.integers(0, n_blocks, size=n_blocks)
        resampled = np.concatenate([blocks[c] for c in chosen])
        boot_sharpes.append(sharpe(resampled))
        boot_pnls.append(float(np.sum(resampled)) / n_days * 252)
        boot_dds.append(max_dd(resampled))
        boot_cvars.append(cvar(resampled, 5))

        if (i + 1) % 1000 == 0:
            print(f"    Bootstrap progress: {i+1}/{BOOTSTRAP_N_RESAMPLES}", flush=True)

    elapsed = time.time() - t0
    boot_sharpes = np.array(boot_sharpes)
    boot_pnls = np.array(boot_pnls)
    boot_dds = np.array(boot_dds)
    boot_cvars = np.array(boot_cvars)

    ci_95 = lambda arr: (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))

    sharpe_ci = ci_95(boot_sharpes)
    pnl_ci = ci_95(boot_pnls)
    dd_ci = ci_95(boot_dds)
    cvar_ci = ci_95(boot_cvars)

    print(f"\n  95% Confidence Intervals ({BOOTSTRAP_N_RESAMPLES} resamples, {elapsed:.0f}s):", flush=True)
    print(f"  {'Metric':<20} {'Mean':>10} {'Lower 2.5%':>14} {'Upper 97.5%':>14}", flush=True)
    print(f"  {'-'*20} {'-'*10} {'-'*14} {'-'*14}", flush=True)
    print(f"  {'Ann. Sharpe':<20} {np.mean(boot_sharpes):>10.3f} {sharpe_ci[0]:>14.3f} {sharpe_ci[1]:>14.3f}", flush=True)
    print(f"  {'Ann. PnL ($)':<20} {np.mean(boot_pnls):>10.0f} {pnl_ci[0]:>14.0f} {pnl_ci[1]:>14.0f}", flush=True)
    print(f"  {'MaxDD ($)':<20} {np.mean(boot_dds):>10.0f} {dd_ci[0]:>14.0f} {dd_ci[1]:>14.0f}", flush=True)
    print(f"  {'CVaR 5% ($)':<20} {np.mean(boot_cvars):>10.2f} {cvar_ci[0]:>14.2f} {cvar_ci[1]:>14.2f}", flush=True)

    results = {
        'n_resamples': BOOTSTRAP_N_RESAMPLES,
        'block_days': block_size,
        'n_blocks': n_blocks,
        'elapsed_s': round(elapsed, 1),
        'ann_sharpe': {
            'mean': round(float(np.mean(boot_sharpes)), 3),
            'ci_lower': round(sharpe_ci[0], 3),
            'ci_upper': round(sharpe_ci[1], 3),
        },
        'ann_pnl': {
            'mean': round(float(np.mean(boot_pnls)), 2),
            'ci_lower': round(pnl_ci[0], 2),
            'ci_upper': round(pnl_ci[1], 2),
        },
        'max_dd': {
            'mean': round(float(np.mean(boot_dds)), 2),
            'ci_lower': round(dd_ci[0], 2),
            'ci_upper': round(dd_ci[1], 2),
        },
        'cvar_5pct': {
            'mean': round(float(np.mean(boot_cvars)), 2),
            'ci_lower': round(cvar_ci[0], 2),
            'ci_upper': round(cvar_ci[1], 2),
        },
    }
    save_json(results, 'phase5_bootstrap.json')
    return results


def phase6_worst_case(unit_dailies, union_idx):
    """Phase 6: Worst-Case Scenario Analysis."""
    print(f"\n{'=' * 80}", flush=True)
    print(f"  Phase 6: Worst-Case Scenarios", flush=True)
    print(f"{'=' * 80}\n", flush=True)

    port_arr, _ = build_portfolio_daily(unit_dailies, P6_LOTS, union_idx)
    n_days = len(port_arr)
    eq = np.cumsum(port_arr)
    peak = np.maximum.accumulate(eq)
    dd_series = eq - peak

    # Find top 10 drawdown periods
    drawdowns = []
    in_dd = False
    dd_start = 0
    for i in range(n_days):
        if dd_series[i] < 0 and not in_dd:
            dd_start = i
            in_dd = True
        elif (dd_series[i] >= 0 or i == n_days - 1) and in_dd:
            dd_end = i
            depth = float(dd_series[dd_start:dd_end].min())
            trough_idx = dd_start + np.argmin(dd_series[dd_start:dd_end])
            recovery_bars = dd_end - trough_idx if dd_series[i] >= 0 else -1
            drawdowns.append({
                'start_idx': dd_start,
                'trough_idx': int(trough_idx),
                'end_idx': dd_end,
                'start_date': str(union_idx[dd_start].date()),
                'trough_date': str(union_idx[trough_idx].date()),
                'end_date': str(union_idx[min(dd_end, n_days - 1)].date()),
                'depth': round(depth, 2),
                'duration_days': dd_end - dd_start,
                'recovery_days': recovery_bars if recovery_bars >= 0 else None,
            })
            in_dd = False

    drawdowns.sort(key=lambda x: x['depth'])
    top_10_dd = drawdowns[:10]

    print(f"  Top 10 Worst Drawdown Periods:", flush=True)
    print(f"  {'#':>3} {'Start':>12} {'Trough':>12} {'End':>12} {'Depth':>10} "
          f"{'Duration':>10} {'Recovery':>10}", flush=True)
    print(f"  {'-'*3} {'-'*12} {'-'*12} {'-'*12} {'-'*10} {'-'*10} {'-'*10}", flush=True)

    for rank, dd in enumerate(top_10_dd, 1):
        rec_str = f"{dd['recovery_days']}d" if dd['recovery_days'] is not None else "ongoing"
        print(f"  {rank:>3} {dd['start_date']:>12} {dd['trough_date']:>12} {dd['end_date']:>12} "
              f"{fmt(dd['depth']):>10} {dd['duration_days']:>9}d {rec_str:>10}", flush=True)

    # Per-strategy contribution during worst drawdowns
    strat_scaled = {}
    for name in STRAT_ORDER:
        ds = unit_dailies.get(name, pd.Series(dtype=float))
        mult = P6_LOTS.get(name, UNIT_LOT) / UNIT_LOT
        strat_scaled[name] = ds.reindex(union_idx, fill_value=0.0).values * mult

    print(f"\n  Per-Strategy Contribution During Top 5 Drawdowns:", flush=True)
    for rank, dd in enumerate(top_10_dd[:5], 1):
        s_idx = dd['start_idx']
        t_idx = dd['trough_idx'] + 1
        print(f"\n    DD #{rank}: {dd['start_date']} -> {dd['trough_date']} (depth={fmt(dd['depth'])})", flush=True)
        contributions = {}
        for name in STRAT_ORDER:
            pnl = float(np.sum(strat_scaled[name][s_idx:t_idx]))
            contributions[name] = pnl
        total = sum(contributions.values())
        for name in STRAT_ORDER:
            pct = contributions[name] / total * 100 if abs(total) > 0.01 else 0
            print(f"      {name:<14}: {fmt(contributions[name]):>10} ({pct:>5.1f}%)", flush=True)
        dd['strategy_contributions'] = {k: round(v, 2) for k, v in contributions.items()}

    # Maximum losses in single day/week/month
    worst_1d = float(np.min(port_arr))
    worst_1d_idx = int(np.argmin(port_arr))
    worst_1d_date = str(union_idx[worst_1d_idx].date())

    week_returns = pd.Series(port_arr, index=union_idx).resample('W').sum()
    worst_1w = float(week_returns.min())
    worst_1w_date = str(week_returns.idxmin().date()) if len(week_returns) > 0 else "N/A"

    month_returns = pd.Series(port_arr, index=union_idx).resample('ME').sum()
    worst_1m = float(month_returns.min())
    worst_1m_date = str(month_returns.idxmin().date()) if len(month_returns) > 0 else "N/A"

    print(f"\n  Maximum Losses:", flush=True)
    print(f"    Worst single day:   {fmt(worst_1d)} ({worst_1d_date})", flush=True)
    print(f"    Worst single week:  {fmt(worst_1w)} (week of {worst_1w_date})", flush=True)
    print(f"    Worst single month: {fmt(worst_1m)} (month of {worst_1m_date})", flush=True)

    # Strategy contributions on worst day
    print(f"\n  Strategy breakdown on worst day ({worst_1d_date}):", flush=True)
    for name in STRAT_ORDER:
        pnl = float(strat_scaled[name][worst_1d_idx])
        print(f"    {name:<14}: {fmt(pnl)}", flush=True)

    results = {
        'top_10_drawdowns': top_10_dd,
        'max_losses': {
            'worst_1d': {'value': round(worst_1d, 2), 'date': worst_1d_date},
            'worst_1w': {'value': round(worst_1w, 2), 'date': worst_1w_date},
            'worst_1m': {'value': round(worst_1m, 2), 'date': worst_1m_date},
        },
        'worst_day_breakdown': {
            name: round(float(strat_scaled[name][worst_1d_idx]), 2)
            for name in STRAT_ORDER
        },
    }
    save_json(results, 'phase6_worst_case.json')
    return results


# ===================================================================
# Main
# ===================================================================

def main():
    t0 = time.time()
    print("=" * 80, flush=True)
    print("  R171 — P6 Portfolio Monte Carlo Stress Test", flush=True)
    print(f"  {MC_N_SIMS} MC sims | +/-{MC_PERTURB_PCT*100:.0f}% perturbation | "
          f"{BOOTSTRAP_N_RESAMPLES} bootstrap resamples", flush=True)
    print(f"  P6 lots: {P6_LOTS}", flush=True)
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("=" * 80, flush=True)

    # Load data
    print("\n  Loading data...", flush=True)
    h1_df = load_h1()
    print(f"  H1: {len(h1_df)} bars ({h1_df.index[0]} ~ {h1_df.index[-1]})", flush=True)

    print("  Loading L8_MAX DataBundle...", flush=True)
    from backtest.runner import DataBundle
    l8_bundle = DataBundle.load_custom()
    print("  DataBundle ready.", flush=True)

    # ── Phase 1 ──
    phase1_t0 = time.time()
    unit_trades, unit_dailies, unit_stats, union_idx = phase1_baseline(h1_df, l8_bundle)
    print(f"\n  Phase 1 done in {time.time() - phase1_t0:.0f}s", flush=True)

    # ── Phase 2 ──
    phase2_t0 = time.time()
    mc_results = phase2_monte_carlo(h1_df, unit_dailies)
    print(f"\n  Phase 2 done in {time.time() - phase2_t0:.0f}s", flush=True)

    # ── Phase 3 ──
    phase3_t0 = time.time()
    dropout_results = phase3_dropout(h1_df, unit_dailies, union_idx)
    print(f"\n  Phase 3 done in {time.time() - phase3_t0:.0f}s", flush=True)

    # ── Phase 4 ──
    phase4_t0 = time.time()
    cond_corr_results = phase4_conditional_correlation(unit_dailies, union_idx)
    print(f"\n  Phase 4 done in {time.time() - phase4_t0:.0f}s", flush=True)

    # ── Phase 5 ──
    phase5_t0 = time.time()
    bootstrap_results = phase5_bootstrap(unit_dailies, union_idx)
    print(f"\n  Phase 5 done in {time.time() - phase5_t0:.0f}s", flush=True)

    # ── Phase 6 ──
    phase6_t0 = time.time()
    worst_case_results = phase6_worst_case(unit_dailies, union_idx)
    print(f"\n  Phase 6 done in {time.time() - phase6_t0:.0f}s", flush=True)

    # ── Final Summary ──
    total_elapsed = time.time() - t0
    print(f"\n{'=' * 80}", flush=True)
    print(f"  FINAL SUMMARY", flush=True)
    print(f"{'=' * 80}", flush=True)

    print(f"\n  Phase timings:", flush=True)
    print(f"    Phase 1 (Baseline):              {time.time() - t0 - (time.time() - phase1_t0):.0f}s", flush=True)
    print(f"    Phase 2 (MC {MC_N_SIMS} sims):          see above", flush=True)
    print(f"    Phase 3 (Dropout):               see above", flush=True)
    print(f"    Phase 4 (Cond. Correlation):     see above", flush=True)
    print(f"    Phase 5 (Bootstrap {BOOTSTRAP_N_RESAMPLES}):      see above", flush=True)
    print(f"    Phase 6 (Worst Case):            see above", flush=True)

    print(f"\n  Key Results:", flush=True)

    if mc_results:
        sharpes = np.array([r['port_sharpe'] for r in mc_results])
        p_gt0 = float(np.mean(sharpes > 0) * 100)
        p_gt2 = float(np.mean(sharpes > 2) * 100)
        print(f"    MC Sharpe: {np.mean(sharpes):.3f} +/- {np.std(sharpes):.3f} "
              f"[{np.percentile(sharpes, 5):.3f}, {np.percentile(sharpes, 95):.3f}]", flush=True)
        print(f"    P(Sharpe > 0) = {p_gt0:.1f}%, P(Sharpe > 2) = {p_gt2:.1f}%", flush=True)

    if dropout_results:
        mi = dropout_results['most_important_strategy']
        li = dropout_results['least_important_strategy']
        print(f"    Most important: {mi}, Least important: {li}", flush=True)

    if cond_corr_results:
        fail = "YES" if cond_corr_results['diversification_failure'] else "NO"
        print(f"    Diversification failure in stress: {fail} "
              f"(calm={cond_corr_results['mean_calm_corr']:.3f}, "
              f"stress={cond_corr_results['mean_stress_corr']:.3f})", flush=True)

    if bootstrap_results:
        bs = bootstrap_results['ann_sharpe']
        print(f"    Bootstrap Sharpe 95% CI: [{bs['ci_lower']:.3f}, {bs['ci_upper']:.3f}]", flush=True)

    if worst_case_results:
        wc = worst_case_results['max_losses']
        print(f"    Worst day: {fmt(wc['worst_1d']['value'])} ({wc['worst_1d']['date']})", flush=True)
        print(f"    Worst month: {fmt(wc['worst_1m']['value'])} ({wc['worst_1m']['date']})", flush=True)

    # Save master summary
    master = {
        'experiment': 'R171 P6 Portfolio Monte Carlo Stress Test',
        'config': {
            'p6_lots': P6_LOTS,
            'p6_caps': P6_CAPS,
            'mc_n_sims': MC_N_SIMS,
            'mc_perturb_pct': MC_PERTURB_PCT,
            'bootstrap_n_resamples': BOOTSTRAP_N_RESAMPLES,
            'bootstrap_block_days': BOOTSTRAP_BLOCK_DAYS,
            'base_params': {k: {pk: (round(pv, 6) if isinstance(pv, float) else pv)
                                for pk, pv in v.items()}
                           for k, v in BASE_PARAMS.items()},
        },
        'phase1_summary': {
            'unit_stats': unit_stats,
        },
        'phase2_summary': {
            'n_sims': len(mc_results),
            'sharpe_mean': round(float(np.mean([r['port_sharpe'] for r in mc_results])), 3) if mc_results else 0,
            'sharpe_std': round(float(np.std([r['port_sharpe'] for r in mc_results])), 3) if mc_results else 0,
            'prob_sharpe_gt0': round(float(np.mean([r['port_sharpe'] > 0 for r in mc_results]) * 100), 1) if mc_results else 0,
        },
        'phase3_summary': {
            'most_important': dropout_results.get('most_important_strategy', ''),
            'least_important': dropout_results.get('least_important_strategy', ''),
        },
        'phase4_summary': {
            'diversification_failure': cond_corr_results.get('diversification_failure', False),
            'mean_calm_corr': cond_corr_results.get('mean_calm_corr', 0),
            'mean_stress_corr': cond_corr_results.get('mean_stress_corr', 0),
        },
        'phase5_summary': bootstrap_results,
        'phase6_summary': {
            'worst_1d': worst_case_results.get('max_losses', {}).get('worst_1d', {}),
            'worst_1m': worst_case_results.get('max_losses', {}).get('worst_1m', {}),
        },
        'total_elapsed_s': round(total_elapsed, 1),
    }
    save_json(master, 'r171_master_summary.json')

    print(f"\n{'=' * 80}", flush=True)
    print(f"  R171 COMPLETE — {total_elapsed:.0f}s ({total_elapsed / 60:.1f}min)", flush=True)
    print(f"  Results: {OUTPUT_DIR}/", flush=True)
    print(f"{'=' * 80}", flush=True)


if __name__ == "__main__":
    main()

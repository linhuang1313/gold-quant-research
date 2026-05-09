#!/usr/bin/env python3
"""
R150 — 6-Strategy Portfolio Lot Size Optimization ($5,000 Capital)
===================================================================
Extends R89 (4 strategies) to include S3 Dual Thrust and S4 Chandelier.
Finds the optimal lot combination for L8_MAX / PSAR / TSMOM / SESS_BO /
DUAL_THRUST / CHANDELIER that maximizes portfolio Sharpe while keeping
combined MaxDD <= $500.

Phase 1: Run each strategy at unit lot (0.01) with recommended Cap
Phase 2: Correlation matrix (6x6)
Phase 3: Smart 2-stage grid search (8^6 = 262,144 combos)
Phase 4: K-Fold 5-fold validation on top 5 combos
Phase 5: Compare with current live config
"""
import sys, os, io, time, json, glob
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from itertools import product

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

OUTPUT_DIR = Path("results/r150_6strat_lot_optimizer")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100
CAPITAL = 5000
MAX_DD_LIMIT = 500
SPREAD = 0.30
UNIT_LOT = 0.01

CAPS = {
    'L8_MAX':      35,
    'PSAR':         5,
    'TSMOM':        0,
    'SESS_BO':     35,
    'DUAL_THRUST': 35,
    'CHANDELIER':  35,
}

STRAT_ORDER = ['L8_MAX', 'PSAR', 'TSMOM', 'SESS_BO', 'DUAL_THRUST', 'CHANDELIER']

LOT_GRID = [0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15]

FOLDS = [
    ("Fold1", "2015-01-01", "2017-03-01"),
    ("Fold2", "2017-03-01", "2019-06-01"),
    ("Fold3", "2019-06-01", "2021-09-01"),
    ("Fold4", "2021-09-01", "2023-12-01"),
    ("Fold5", "2023-12-01", "2027-01-01"),
]

LIVE_CONFIG = {
    'L8_MAX':      0.02,
    'PSAR':        0.09,
    'TSMOM':       0.15,
    'SESS_BO':     0.13,
    'DUAL_THRUST': 0.04,
    'CHANDELIER':  0.08,
}


def fmt(v):
    return f"${v:>10,.0f}" if v >= 0 else f"-${abs(v):>9,.0f}"


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
# Shared helpers (from R89/R88)
# ═══════════════════════════════════════════════════════════════

def compute_atr(df, period=14):
    tr = pd.DataFrame({
        'hl': df['High'] - df['Low'],
        'hc': (df['High'] - df['Close'].shift(1)).abs(),
        'lc': (df['Low'] - df['Close'].shift(1)).abs(),
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


def _mk(pos, exit_p, exit_time, reason, bar_idx, pnl):
    return {'dir': pos['dir'], 'entry': pos['entry'], 'exit': exit_p,
            'entry_time': pos['time'], 'exit_time': exit_time,
            'pnl': pnl, 'reason': reason, 'bars': bar_idx - pos['bar']}


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
# Strategy backtests
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


def bt_l8_max(data_bundle, spread, lot, maxloss_cap=0):
    from backtest.runner import run_variant, LIVE_PARITY_KWARGS
    kw = {**LIVE_PARITY_KWARGS, 'maxloss_cap': maxloss_cap,
           'spread_cost': spread, 'initial_capital': 2000,
           'min_lot_size': lot, 'max_lot_size': lot}
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


def bt_dual_thrust(h1_df, spread, lot, maxloss_cap=35,
                   n_bars=6, k=0.5, sl_atr=4.5, tp_atr=8.0,
                   trail_act=0.14, trail_dist=0.025, max_hold=20):
    """S3 Dual Thrust strategy on H1 bars."""
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
    """S4 Chandelier Exit strategy on H1 bars."""
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
            result = _run_exit_with_cap(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                                        sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue

        if i - last_exit < 2:
            continue
        if np.isnan(atr[i]) or atr[i] < 0.1:
            continue

        flipped_long = direction[i] == 1 and direction[i-1] != 1
        flipped_short = direction[i] == -1 and direction[i-1] != -1

        if flipped_long and c[i] > ema[i]:
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif flipped_short and c[i] < ema[i]:
            pos = {'dir': 'SELL', 'entry': c[i]-spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}

    return trades


# ═══════════════════════════════════════════════════════════════
# Daily PnL helpers
# ═══════════════════════════════════════════════════════════════

def trades_to_daily_series(trades):
    """Convert trade list to pd.Series with date index and daily PnL values."""
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
    if s == 0:
        return 0.0
    return float(np.mean(arr) / s * np.sqrt(252))


def max_dd(arr):
    if len(arr) == 0:
        return 0.0
    eq = np.cumsum(arr)
    return float((np.maximum.accumulate(eq) - eq).max())


def cvar99(arr):
    if len(arr) < 20:
        return 0.0
    v = np.percentile(arr, 1)
    tail = arr[arr <= v]
    return float(tail.mean()) if len(tail) > 0 else float(v)


# ═══════════════════════════════════════════════════════════════
# Portfolio combiner
# ═══════════════════════════════════════════════════════════════

def build_portfolio_daily(unit_dailies, lots):
    """
    Combine unit-lot daily PnL series scaled by lot multipliers.
    unit_dailies: dict of {strat_name: pd.Series at lot=0.01}
    lots: dict of {strat_name: target_lot}
    Returns: np.array of portfolio daily PnL, aligned to union of all dates.
    """
    all_dates = set()
    for ds in unit_dailies.values():
        all_dates.update(ds.index)
    all_dates = sorted(all_dates)
    idx = pd.DatetimeIndex(all_dates)

    portfolio = np.zeros(len(idx))
    for name in STRAT_ORDER:
        if name not in unit_dailies:
            continue
        ds = unit_dailies[name]
        multiplier = lots[name] / UNIT_LOT
        aligned = ds.reindex(idx, fill_value=0.0).values * multiplier
        portfolio += aligned
    return portfolio


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 80, flush=True)
    print("  R150 — 6-Strategy Portfolio Lot Size Optimization", flush=True)
    print(f"  Capital: ${CAPITAL:,}  |  MaxDD limit: ${MAX_DD_LIMIT:,}  |  Grid: {len(LOT_GRID)}^6 = {len(LOT_GRID)**6:,} combos", flush=True)
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("=" * 80, flush=True)

    from backtest.runner import DataBundle

    # ══════════════════════════════════════════════════════════
    #  Phase 1: Run each strategy at unit lot
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*80}", flush=True)
    print(f"  Phase 1: Per-Strategy Backtest at unit lot ({UNIT_LOT})", flush=True)
    print(f"{'='*80}\n", flush=True)

    print("  Loading data...", flush=True)
    h1_df = load_h1()
    print(f"  H1: {len(h1_df)} bars ({h1_df.index[0]} ~ {h1_df.index[-1]})", flush=True)

    print("  Preparing L8_MAX DataBundle...", flush=True)
    l8_bundle = DataBundle.load_custom()
    print("  L8 bundle ready.\n", flush=True)

    unit_trades = {}
    unit_dailies = {}
    unit_stats = {}

    # H1 strategies
    h1_strats = {
        'PSAR':        (bt_psar,         {}),
        'TSMOM':       (bt_tsmom,        {}),
        'SESS_BO':     (bt_sess_bo,      {}),
        'DUAL_THRUST': (bt_dual_thrust,  {}),
        'CHANDELIER':  (bt_chandelier,   {}),
    }

    for name, (fn, kw) in h1_strats.items():
        cap = CAPS[name]
        trades = fn(h1_df, spread=SPREAD, lot=UNIT_LOT, maxloss_cap=cap, **kw)
        unit_trades[name] = trades
        unit_dailies[name] = trades_to_daily_series(trades)
        pnls = [t['pnl'] for t in trades]
        daily_arr = unit_dailies[name].values
        unit_stats[name] = {
            'n_trades': len(trades),
            'pnl': round(sum(pnls), 2),
            'sharpe': round(sharpe(daily_arr), 2),
            'max_dd': round(max_dd(daily_arr), 2),
            'wr': round(sum(1 for p in pnls if p > 0) / max(len(pnls), 1) * 100, 1),
            'cap': cap,
        }
        print(f"    {name:>12}: {len(trades)} trades, Sharpe={unit_stats[name]['sharpe']:.2f}, "
              f"PnL={fmt(unit_stats[name]['pnl'])}, MaxDD={fmt(unit_stats[name]['max_dd'])}, "
              f"Cap=${cap if cap > 0 else 'None'}", flush=True)

    # L8_MAX via engine
    cap = CAPS['L8_MAX']
    trades = bt_l8_max(l8_bundle, spread=SPREAD, lot=UNIT_LOT, maxloss_cap=cap)
    unit_trades['L8_MAX'] = trades
    unit_dailies['L8_MAX'] = trades_to_daily_series(trades)
    pnls = [t['pnl'] for t in trades]
    daily_arr = unit_dailies['L8_MAX'].values
    unit_stats['L8_MAX'] = {
        'n_trades': len(trades),
        'pnl': round(sum(pnls), 2),
        'sharpe': round(sharpe(daily_arr), 2),
        'max_dd': round(max_dd(daily_arr), 2),
        'wr': round(sum(1 for p in pnls if p > 0) / max(len(pnls), 1) * 100, 1),
        'cap': cap,
    }
    print(f"    {'L8_MAX':>12}: {len(trades)} trades, Sharpe={unit_stats['L8_MAX']['sharpe']:.2f}, "
          f"PnL={fmt(unit_stats['L8_MAX']['pnl'])}, MaxDD={fmt(unit_stats['L8_MAX']['max_dd'])}, "
          f"Cap=${cap}", flush=True)

    print(f"\n  Phase 1 complete. Unit-lot baselines ready.", flush=True)

    # ══════════════════════════════════════════════════════════
    #  Phase 2: Correlation Matrix (6x6)
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*80}", flush=True)
    print(f"  Phase 2: Daily PnL Correlation Matrix (6x6)", flush=True)
    print(f"{'='*80}\n", flush=True)

    all_dates = set()
    for ds in unit_dailies.values():
        all_dates.update(ds.index)
    all_dates = sorted(all_dates)
    corr_idx = pd.DatetimeIndex(all_dates)

    corr_df = pd.DataFrame(index=corr_idx)
    for name in STRAT_ORDER:
        if name in unit_dailies:
            corr_df[name] = unit_dailies[name].reindex(corr_idx, fill_value=0.0)
        else:
            corr_df[name] = 0.0

    corr_matrix = corr_df.corr()
    print(f"  {'':>12}", end="", flush=True)
    for name in STRAT_ORDER:
        print(f" {name:>12}", end="", flush=True)
    print("", flush=True)
    for row_name in STRAT_ORDER:
        print(f"  {row_name:>12}", end="", flush=True)
        for col_name in STRAT_ORDER:
            print(f" {corr_matrix.loc[row_name, col_name]:>12.3f}", end="", flush=True)
        print("", flush=True)

    corr_json = {}
    for r in STRAT_ORDER:
        corr_json[r] = {}
        for c_name in STRAT_ORDER:
            corr_json[r][c_name] = round(float(corr_matrix.loc[r, c_name]), 4)

    print(f"\n  Phase 2 complete.", flush=True)

    # ══════════════════════════════════════════════════════════
    #  Phase 3: Smart 2-Stage Grid Search (8^6 = 262,144)
    # ══════════════════════════════════════════════════════════
    total_combos = len(LOT_GRID) ** 6
    print(f"\n{'='*80}", flush=True)
    print(f"  Phase 3: 2-Stage Lot Grid Search ({total_combos:,} combos)", flush=True)
    print(f"  Max total lot: 0.60  |  MaxDD <= ${MAX_DD_LIMIT:,}", flush=True)
    print(f"{'='*80}\n", flush=True)

    # Pre-compute aligned daily arrays for fast vectorized screening
    aligned_arrays = {}
    for name in STRAT_ORDER:
        if name in unit_dailies:
            aligned_arrays[name] = unit_dailies[name].reindex(corr_idx, fill_value=0.0).values
        else:
            aligned_arrays[name] = np.zeros(len(corr_idx))

    n_days = len(corr_idx)
    print(f"  Stage A: Screening all {total_combos:,} combos (pre-computed {n_days} daily arrays)...", flush=True)

    results = []
    checked = 0
    feasible = 0
    stage_a_t0 = time.time()

    for l8_lot, psar_lot, tsmom_lot, sess_lot, dt_lot, ch_lot in product(LOT_GRID, repeat=6):
        checked += 1

        total_lot = l8_lot + psar_lot + tsmom_lot + sess_lot + dt_lot + ch_lot
        if total_lot > 0.60:
            continue

        port_daily = (
            aligned_arrays['L8_MAX'] * (l8_lot / UNIT_LOT) +
            aligned_arrays['PSAR'] * (psar_lot / UNIT_LOT) +
            aligned_arrays['TSMOM'] * (tsmom_lot / UNIT_LOT) +
            aligned_arrays['SESS_BO'] * (sess_lot / UNIT_LOT) +
            aligned_arrays['DUAL_THRUST'] * (dt_lot / UNIT_LOT) +
            aligned_arrays['CHANDELIER'] * (ch_lot / UNIT_LOT)
        )

        eq = np.cumsum(port_daily)
        dd = float((np.maximum.accumulate(eq) - eq).max())
        if dd > MAX_DD_LIMIT:
            continue

        feasible += 1
        sh = sharpe(port_daily)
        pnl = float(np.sum(port_daily))
        cv = cvar99(port_daily)

        lots = {
            'L8_MAX': l8_lot, 'PSAR': psar_lot, 'TSMOM': tsmom_lot,
            'SESS_BO': sess_lot, 'DUAL_THRUST': dt_lot, 'CHANDELIER': ch_lot,
        }
        results.append({
            'lots': lots,
            'sharpe': round(sh, 3),
            'pnl': round(pnl, 2),
            'max_dd': round(dd, 2),
            'cvar99': round(cv, 2),
            'dd_pct': round(dd / CAPITAL * 100, 1),
            'total_lot': round(total_lot, 2),
        })

        if checked % 50000 == 0:
            print(f"    Progress: {checked:,}/{total_combos:,} checked, {feasible:,} feasible...", flush=True)

    stage_a_elapsed = time.time() - stage_a_t0
    print(f"  Stage A complete: {checked:,} tested, {feasible:,} feasible (DD<=${MAX_DD_LIMIT}) in {stage_a_elapsed:.0f}s", flush=True)

    # Stage B: Rank by Sharpe, keep top 50
    print(f"\n  Stage B: Ranking {feasible:,} feasible combos by Sharpe...", flush=True)
    results.sort(key=lambda x: x['sharpe'], reverse=True)
    top50 = results[:50]

    n_trading_days = n_days
    years = n_trading_days / 252

    # Print top 20
    print(f"\n  Top 20 lot combinations by Sharpe:", flush=True)
    print(f"  {'Rk':>3} {'L8':>5} {'PSAR':>5} {'TSMOM':>5} {'SESS':>5} {'DT':>5} {'CH':>5} "
          f"{'TotLot':>6} {'Sharpe':>7} {'PnL':>12} {'MaxDD':>10} {'DD%':>6} {'CVaR99':>8}", flush=True)
    print(f"  {'-'*3} {'-'*5} {'-'*5} {'-'*5} {'-'*5} {'-'*5} {'-'*5} "
          f"{'-'*6} {'-'*7} {'-'*12} {'-'*10} {'-'*6} {'-'*8}", flush=True)
    for i, r in enumerate(results[:20]):
        lo = r['lots']
        print(f"  {i+1:>3} {lo['L8_MAX']:>5.2f} {lo['PSAR']:>5.2f} {lo['TSMOM']:>5.2f} "
              f"{lo['SESS_BO']:>5.2f} {lo['DUAL_THRUST']:>5.2f} {lo['CHANDELIER']:>5.2f} "
              f"{r['total_lot']:>6.2f} {r['sharpe']:>7.3f} {fmt(r['pnl']):>12} "
              f"{fmt(r['max_dd']):>10} {r['dd_pct']:>5.1f}% {r['cvar99']:>8.2f}", flush=True)

    print(f"\n  Phase 3 complete.", flush=True)

    # ══════════════════════════════════════════════════════════
    #  Phase 4: K-Fold 5-Fold Validation (top 5 combos)
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*80}", flush=True)
    print(f"  Phase 4: K-Fold 5-Fold Validation (top 5 combos)", flush=True)
    print(f"{'='*80}\n", flush=True)

    fold_unit_dailies = {}
    for fold_name, start, end in FOLDS:
        print(f"  Computing fold: {fold_name} ({start} ~ {end})...", flush=True)
        fold_unit_dailies[fold_name] = {}
        h1_fold = h1_df[start:end]
        if len(h1_fold) < 100:
            print(f"    [WARN] Fold {fold_name} has only {len(h1_fold)} bars, skipping H1 strats", flush=True)
            for name in h1_strats:
                fold_unit_dailies[fold_name][name] = pd.Series(dtype=float)
        else:
            for name, (fn, kw) in h1_strats.items():
                cap = CAPS[name]
                trades = fn(h1_fold, spread=SPREAD, lot=UNIT_LOT, maxloss_cap=cap, **kw)
                fold_unit_dailies[fold_name][name] = trades_to_daily_series(trades)

    # L8_MAX folds via engine
    for fold_name, start, end in FOLDS:
        try:
            l8_fold = l8_bundle.slice(start, end)
            trades = bt_l8_max(l8_fold, spread=SPREAD, lot=UNIT_LOT, maxloss_cap=CAPS['L8_MAX'])
            fold_unit_dailies[fold_name]['L8_MAX'] = trades_to_daily_series(trades)
        except Exception as e:
            print(f"    [WARN] L8_MAX fold {fold_name}: {e}", flush=True)
            fold_unit_dailies[fold_name]['L8_MAX'] = pd.Series(dtype=float)

    kfold_results = {}
    for rank, combo in enumerate(results[:5]):
        lots = combo['lots']
        label = (f"L8={lots['L8_MAX']:.2f}_P={lots['PSAR']:.2f}_T={lots['TSMOM']:.2f}"
                 f"_S={lots['SESS_BO']:.2f}_DT={lots['DUAL_THRUST']:.2f}_CH={lots['CHANDELIER']:.2f}")
        fold_sharpes = []
        for fold_name, start, end in FOLDS:
            port_daily = build_portfolio_daily(fold_unit_dailies[fold_name], lots)
            fold_sharpes.append(sharpe(port_daily))

        positive = sum(1 for s in fold_sharpes if s > 0)
        mean_sh = float(np.mean(fold_sharpes))
        passed = positive >= 4
        kfold_results[label] = {
            'rank': rank + 1,
            'lots': lots,
            'fold_sharpes': [round(s, 2) for s in fold_sharpes],
            'positive_folds': positive,
            'mean_sharpe': round(mean_sh, 2),
            'pass_4of5': passed,
        }
        status = "PASS" if passed else "FAIL"
        print(f"  #{rank+1} [{status}]  {positive}/5 positive, mean={mean_sh:.2f}", flush=True)
        print(f"       lots: L8={lots['L8_MAX']:.2f} P={lots['PSAR']:.2f} T={lots['TSMOM']:.2f} "
              f"S={lots['SESS_BO']:.2f} DT={lots['DUAL_THRUST']:.2f} CH={lots['CHANDELIER']:.2f}", flush=True)
        print(f"       folds={[round(s, 2) for s in fold_sharpes]}", flush=True)

    print(f"\n  Phase 4 complete.", flush=True)

    # ══════════════════════════════════════════════════════════
    #  Phase 5: Compare with Current Live Config
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*80}", flush=True)
    print(f"  Phase 5: Compare with Current Live Config", flush=True)
    print(f"{'='*80}\n", flush=True)

    live_daily = build_portfolio_daily(unit_dailies, LIVE_CONFIG)
    live_sharpe = sharpe(live_daily)
    live_pnl = float(np.sum(live_daily))
    live_dd = max_dd(live_daily)
    live_cvar = cvar99(live_daily)
    live_total_lot = sum(LIVE_CONFIG.values())

    print(f"  Current Live Config:", flush=True)
    for name in STRAT_ORDER:
        print(f"    {name:>12}: {LIVE_CONFIG[name]:.2f} lot", flush=True)
    print(f"    {'Total':>12}: {live_total_lot:.2f} lot", flush=True)
    print(f"    Sharpe:  {live_sharpe:.3f}", flush=True)
    print(f"    PnL:     {fmt(live_pnl)}", flush=True)
    print(f"    MaxDD:   {fmt(live_dd)} ({live_dd/CAPITAL*100:.1f}%)", flush=True)
    print(f"    CVaR99:  {live_cvar:.2f}", flush=True)

    # Find best optimized combo that passed validation
    winner = None
    for combo in results[:20]:
        lots = combo['lots']
        label = (f"L8={lots['L8_MAX']:.2f}_P={lots['PSAR']:.2f}_T={lots['TSMOM']:.2f}"
                 f"_S={lots['SESS_BO']:.2f}_DT={lots['DUAL_THRUST']:.2f}_CH={lots['CHANDELIER']:.2f}")
        if label in kfold_results and kfold_results[label]['pass_4of5']:
            winner = combo
            break
    if winner is None and results:
        winner = results[0]
        print(f"\n  [NOTE] No combo passed 4/5 folds; using top-Sharpe combo.", flush=True)

    if winner:
        w_lots = winner['lots']
        w_total_lot = sum(w_lots.values())
        print(f"\n  Best Optimized Config:", flush=True)
        for name in STRAT_ORDER:
            diff = w_lots[name] - LIVE_CONFIG[name]
            arrow = "  " if diff == 0 else (f" +{diff:.2f}" if diff > 0 else f" {diff:.2f}")
            print(f"    {name:>12}: {w_lots[name]:.2f} lot{arrow}", flush=True)
        print(f"    {'Total':>12}: {w_total_lot:.2f} lot", flush=True)
        print(f"    Sharpe:  {winner['sharpe']:.3f}  (live: {live_sharpe:.3f}, diff: {winner['sharpe']-live_sharpe:+.3f})", flush=True)
        print(f"    PnL:     {fmt(winner['pnl'])}  (live: {fmt(live_pnl)})", flush=True)
        print(f"    MaxDD:   {fmt(winner['max_dd'])} ({winner['dd_pct']:.1f}%)  (live: {fmt(live_dd)}, {live_dd/CAPITAL*100:.1f}%)", flush=True)

        # Per-strategy contribution at winning lots
        print(f"\n  Per-strategy contribution at winning lot:", flush=True)
        print(f"  {'Strategy':<12} {'Lot':>6} {'PnL':>12} {'MaxDD':>10} {'Sharpe':>8} {'%ofPnL':>8}", flush=True)
        print(f"  {'-'*12} {'-'*6} {'-'*12} {'-'*10} {'-'*8} {'-'*8}", flush=True)
        total_pnl = 0
        strat_pnls = {}
        for name in STRAT_ORDER:
            lot = w_lots[name]
            mult = lot / UNIT_LOT
            ds = unit_dailies.get(name, pd.Series(dtype=float))
            scaled = ds.values * mult if len(ds) > 0 else np.array([])
            pnl_v = float(np.sum(scaled)) if len(scaled) > 0 else 0.0
            strat_pnls[name] = pnl_v
            total_pnl += pnl_v
        for name in STRAT_ORDER:
            lot = w_lots[name]
            mult = lot / UNIT_LOT
            ds = unit_dailies.get(name, pd.Series(dtype=float))
            scaled = ds.values * mult if len(ds) > 0 else np.array([])
            pnl_v = strat_pnls[name]
            sh_v = sharpe(scaled) if len(scaled) > 0 else 0.0
            dd_v = max_dd(scaled) if len(scaled) > 0 else 0.0
            pct = pnl_v / total_pnl * 100 if total_pnl > 0 else 0
            print(f"  {name:<12} {lot:>6.2f} {fmt(pnl_v):>12} {fmt(dd_v):>10} "
                  f"{sh_v:>8.2f} {pct:>7.1f}%", flush=True)

    # ══════════════════════════════════════════════════════════
    #  Final Summary
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*80}", flush=True)
    print(f"  FINAL RECOMMENDATION", flush=True)
    print(f"{'='*80}\n", flush=True)

    if winner:
        lots = winner['lots']
        print(f"  Capital: ${CAPITAL:,}", flush=True)
        print(f"  MaxDD Limit: ${MAX_DD_LIMIT:,} ({MAX_DD_LIMIT/CAPITAL*100:.0f}%)", flush=True)
        print(f"  ", flush=True)
        print(f"  Recommended lot sizes:", flush=True)
        for name in STRAT_ORDER:
            lot = lots[name]
            cap = CAPS[name]
            cap_str = f"Cap${cap}" if cap > 0 else "NoCap"
            print(f"    {name:<12}  {lot:.2f} lot  ({cap_str})", flush=True)
        print(f"  ", flush=True)
        print(f"  Portfolio metrics (backtest):", flush=True)
        print(f"    Sharpe:       {winner['sharpe']:.3f}", flush=True)
        print(f"    Total PnL:    {fmt(winner['pnl'])}", flush=True)
        print(f"    MaxDD:        {fmt(winner['max_dd'])}  ({winner['dd_pct']:.1f}% of capital)", flush=True)
        print(f"    CVaR99:       {fmt(winner['cvar99'])}", flush=True)

    elapsed = time.time() - t0
    print(f"\n{'='*80}", flush=True)
    print(f"  R150 COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)
    print(f"{'='*80}", flush=True)

    # Save results
    output = {
        'config': {
            'capital': CAPITAL,
            'max_dd_limit': MAX_DD_LIMIT,
            'spread': SPREAD,
            'caps': CAPS,
            'lot_grid': LOT_GRID,
            'unit_lot': UNIT_LOT,
            'strat_order': STRAT_ORDER,
        },
        'unit_stats': unit_stats,
        'correlation': corr_json,
        'top_20': results[:20],
        'top_50': [r for r in results[:50]],
        'kfold': kfold_results,
        'live_config': {
            'lots': LIVE_CONFIG,
            'sharpe': round(live_sharpe, 3),
            'pnl': round(live_pnl, 2),
            'max_dd': round(live_dd, 2),
            'cvar99': round(live_cvar, 2),
        },
        'winner': winner,
        'total_feasible': feasible,
        'total_tested': checked,
        'elapsed_s': round(elapsed, 1),
    }
    out_path = OUTPUT_DIR / "r150_results.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()

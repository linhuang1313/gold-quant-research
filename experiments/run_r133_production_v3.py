#!/usr/bin/env python3
"""
R133 — Production Portfolio v3 Unified Validation
===================================================
Combines three validated improvements into a new production config and validates
against v1 (current baseline):

  v1 (baseline): R89 portfolio — PSAR default params, no TSMOM filter, no S4
  v2:            PSAR optimized params (sl=4.0, tp=6.0, trail_act=0.08,
                 trail_dist=0.015, max_hold=15)
  v3:            v2 + D1 EMA20 filter on TSMOM + S4 Chandelier as 5th strategy

Validation phases:
  1. Full-data comparison (v1 vs v2 vs v3)
  2. Walk-Forward OOS (4yr train / 2yr test, 6 windows)
  3. 8-Fold K-Fold
  4. Yearly stability (each year 2015–2026)
  5. Monte Carlo (1000 paths)
  6. PBO (1000 shuffles)
  7. Final verdict
"""
import sys, os, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

from backtest.runner import DataBundle, run_variant, LIVE_PARITY_KWARGS, load_csv
from indicators import calc_chandelier

OUTPUT_DIR = Path("results/r133_production_v3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100; SPREAD = 0.30; UNIT_LOT = 0.01
CAPS = {'L8_MAX': 35, 'PSAR': 5, 'TSMOM': 0, 'SESS_BO': 35}
R89_LOTS = {'L8_MAX': 0.02, 'PSAR': 0.09, 'TSMOM': 0.08, 'SESS_BO': 0.08}
S4_LOT = 0.05
STRAT_ORDER_4 = ['L8_MAX', 'PSAR', 'TSMOM', 'SESS_BO']
STRAT_ORDER_5 = ['L8_MAX', 'PSAR', 'TSMOM', 'SESS_BO', 'S4_CHAND']

PSAR_DEFAULT = {'sl_atr': 4.5, 'tp_atr': 16.0, 'trail_act': 0.20, 'trail_dist': 0.04, 'max_hold': 20}
PSAR_OPTIMIZED = {'sl_atr': 4.0, 'tp_atr': 6.0, 'trail_act': 0.08, 'trail_dist': 0.015, 'max_hold': 15}

t0 = time.time()


# ═══════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════

def compute_atr(df, period=14):
    h, l, c = df['High'], df['Low'], df['Close']
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def add_psar(df, af_step=0.01, af_max=0.05):
    h, l, c = df['High'].values, df['Low'].values, df['Close'].values
    n = len(df)
    psar = np.empty(n); psar[:] = np.nan
    af = af_step; rising = True; ep = h[0]; psar[0] = l[0]
    for i in range(1, n):
        prev = psar[i-1]
        if rising:
            psar[i] = prev + af * (ep - prev)
            psar[i] = min(psar[i], l[i-1], l[max(0, i-2)])
            if l[i] < psar[i]:
                rising = False; psar[i] = ep; ep = l[i]; af = af_step
            else:
                if h[i] > ep: ep = h[i]; af = min(af + af_step, af_max)
        else:
            psar[i] = prev + af * (ep - prev)
            psar[i] = max(psar[i], h[i-1], h[max(0, i-2)])
            if h[i] > psar[i]:
                rising = True; psar[i] = ep; ep = h[i]; af = af_step
            else:
                if l[i] < ep: ep = l[i]; af = min(af + af_step, af_max)
    df['PSAR'] = psar


def _mk(pos, exit_px, exit_time, reason, bar_i, pnl):
    return {'dir': pos['dir'], 'entry': pos['entry'], 'exit': exit_px,
            'entry_time': pos['time'], 'exit_time': exit_time,
            'pnl': pnl, 'reason': reason, 'bars': bar_i - pos['bar']}


def _run_exit(pos, i, hi, lo, cl, spread, lot, pv, times,
              sl_atr, tp_atr, trail_act, trail_dist, max_hold, cap):
    atr = pos['atr']
    sl = atr * sl_atr; tp = atr * tp_atr; bars = i - pos['bar']
    if pos['dir'] == 'BUY':
        pnl_now = (cl - pos['entry'] - spread) * lot * pv
        if cap > 0 and pnl_now < -cap:
            return _mk(pos, cl, times[i], "MaxLossCap", i, -cap)
        if lo <= pos['entry'] - sl:
            return _mk(pos, pos['entry'] - sl, times[i], "SL", i, -sl * lot * pv)
        if hi >= pos['entry'] + tp:
            return _mk(pos, pos['entry'] + tp, times[i], "TP", i, tp * lot * pv)
        extreme = pos.get('extreme', pos['entry']); extreme = max(extreme, hi); pos['extreme'] = extreme
        if extreme - pos['entry'] >= atr * trail_act:
            trail_price = extreme - atr * trail_dist
            if cl <= trail_price:
                return _mk(pos, trail_price, times[i], "Trail", i, (trail_price - pos['entry'] - spread) * lot * pv)
        if bars >= max_hold:
            return _mk(pos, cl, times[i], "TimeExit", i, pnl_now)
    else:
        pnl_now = (pos['entry'] - cl - spread) * lot * pv
        if cap > 0 and pnl_now < -cap:
            return _mk(pos, cl, times[i], "MaxLossCap", i, -cap)
        if hi >= pos['entry'] + sl:
            return _mk(pos, pos['entry'] + sl, times[i], "SL", i, -sl * lot * pv)
        if lo <= pos['entry'] - tp:
            return _mk(pos, pos['entry'] - tp, times[i], "TP", i, tp * lot * pv)
        extreme = pos.get('extreme', pos['entry']); extreme = min(extreme, lo); pos['extreme'] = extreme
        if pos['entry'] - extreme >= atr * trail_act:
            trail_price = extreme + atr * trail_dist
            if cl >= trail_price:
                return _mk(pos, trail_price, times[i], "Trail", i, (pos['entry'] - trail_price - spread) * lot * pv)
        if bars >= max_hold:
            return _mk(pos, cl, times[i], "TimeExit", i, pnl_now)
    return None


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
    eq = np.cumsum(arr)
    return float((np.maximum.accumulate(eq) - eq).max()) if len(eq) > 1 else 0.0


# ═══════════════════════════════════════════════════════════════
# D1 EMA20 filter for TSMOM
# ═══════════════════════════════════════════════════════════════

def build_d1_ema_filter(h1_df, ema_period=20):
    """Resample H1 to D1, compute EMA, return daily direction dict."""
    d1 = h1_df.resample('1D').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
    }).dropna()
    d1['EMA20'] = d1['Close'].ewm(span=ema_period).mean()
    direction = {}
    for dt, row in d1.iterrows():
        if pd.isna(row['EMA20']):
            continue
        day = dt.date() if hasattr(dt, 'date') else dt
        direction[day] = 1 if row['Close'] > row['EMA20'] else -1
    return direction


# ═══════════════════════════════════════════════════════════════
# Strategy backtests
# ═══════════════════════════════════════════════════════════════

def bt_psar(h1_df, spread, lot, maxloss_cap=0, params=None):
    if params is None:
        params = PSAR_DEFAULT
    df = h1_df.copy(); add_psar(df); df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR', 'PSAR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; psar = df['PSAR'].values; times = df.index; n = len(df)
    trades = []; pos = None; last_exit = -999
    for i in range(1, n):
        if pos is not None:
            result = _run_exit(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                               params['sl_atr'], params['tp_atr'], params['trail_act'],
                               params['trail_dist'], params['max_hold'], maxloss_cap)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if i - last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        if psar[i-1] > c[i-1] and psar[i] < c[i]:
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif psar[i-1] < c[i-1] and psar[i] > c[i]:
            pos = {'dir': 'SELL', 'entry': c[i]-spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_tsmom(h1_df, spread, lot, maxloss_cap=0,
             fast=480, slow=720, sl_atr=4.5, tp_atr=6.0,
             trail_act=0.14, trail_dist=0.025, max_hold=20,
             d1_filter=None):
    """TSMOM with optional D1 EMA20 direction filter."""
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

    dates_arr = None
    if d1_filter is not None:
        dates_arr = [t.date() if hasattr(t, 'date') else t for t in times]

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
        if i-last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        if np.isnan(score[i]) or np.isnan(score[i-1]): continue

        want_buy = score[i] > 0 and score[i-1] <= 0
        want_sell = score[i] < 0 and score[i-1] >= 0

        if d1_filter is not None and dates_arr is not None:
            d1_dir = d1_filter.get(dates_arr[i], 0)
            if want_buy and d1_dir != 1:
                want_buy = False
            if want_sell and d1_dir != -1:
                want_sell = False

        if want_buy:
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif want_sell:
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


def bt_s4_chandelier(h1_df, spread, lot, maxloss_cap=0,
                     period=22, mult=3.0, ema_filter=True,
                     sl_atr=3.0, tp_atr=8.0, trail_act=0.28,
                     trail_dist=0.06, max_hold=20):
    df = h1_df.copy()
    df['ATR'] = compute_atr(df)
    chand = calc_chandelier(df, period=period, mult=mult)
    df['Chand_long'] = chand['Chand_long']
    df['Chand_short'] = chand['Chand_short']
    if ema_filter:
        df['EMA100'] = df['Close'].ewm(span=100).mean()
    df = df.dropna(subset=['ATR', 'Chand_long', 'Chand_short'])

    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; times = df.index; n = len(df)
    cl_long = df['Chand_long'].values; cl_short = df['Chand_short'].values
    ema = df['EMA100'].values if ema_filter else None

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

        was_below_long = c[i-1] < cl_long[i-1]
        now_above_long = c[i] >= cl_long[i]
        was_above_short = c[i-1] > cl_short[i-1]
        now_below_short = c[i] <= cl_short[i]

        if was_below_long and now_above_long:
            if ema is not None and (np.isnan(ema[i]) or c[i] < ema[i]):
                continue
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif was_above_short and now_below_short:
            if ema is not None and (np.isnan(ema[i]) or c[i] > ema[i]):
                continue
            pos = {'dir': 'SELL', 'entry': c[i]-spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


# ═══════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════

def load_h1():
    candidates = [
        Path("data/download/xauusd-h1-bid-2015-01-01-2026-04-27.csv"),
        Path("data/download/xauusd-h1-bid-2015-01-01-2026-04-10.csv"),
    ]
    h1_path = next((p for p in candidates if p.exists()), candidates[-1])
    return load_csv(str(h1_path))


# ═══════════════════════════════════════════════════════════════
# Portfolio builder
# ═══════════════════════════════════════════════════════════════

def run_portfolio(h1_df, bundle, version='v1', spread=SPREAD):
    """Run a portfolio version and return daily PnL array + dates.

    v1: baseline (PSAR default, TSMOM no filter, no S4)
    v2: PSAR optimized params
    v3: v2 + D1 EMA20 filter on TSMOM + S4 Chandelier
    """
    psar_params = PSAR_OPTIMIZED if version in ('v2', 'v3') else PSAR_DEFAULT
    psar_trades = bt_psar(h1_df, spread, UNIT_LOT, CAPS['PSAR'], params=psar_params)

    d1_filter = build_d1_ema_filter(h1_df) if version == 'v3' else None
    tsmom_trades = bt_tsmom(h1_df, spread, UNIT_LOT, CAPS['TSMOM'], d1_filter=d1_filter)

    sessbo_trades = bt_sess_bo(h1_df, spread, UNIT_LOT, CAPS['SESS_BO'])

    kw = {**LIVE_PARITY_KWARGS, 'maxloss_cap': CAPS['L8_MAX'],
          'spread_cost': spread, 'initial_capital': 2000,
          'min_lot_size': UNIT_LOT, 'max_lot_size': UNIT_LOT}
    try:
        l8_result = run_variant(bundle, f"L8_{version}", verbose=False, **kw)
        l8_trades = [{'dir': t.direction, 'entry': t.entry_price, 'exit': t.exit_price,
                      'entry_time': t.entry_time, 'exit_time': t.exit_time,
                      'pnl': t.pnl, 'reason': t.exit_reason, 'bars': 0}
                     for t in l8_result.get('_trades', [])]
    except Exception:
        l8_trades = []

    strat_trades = {'L8_MAX': l8_trades, 'PSAR': psar_trades,
                    'TSMOM': tsmom_trades, 'SESS_BO': sessbo_trades}
    lots = dict(R89_LOTS)

    if version == 'v3':
        s4_trades = bt_s4_chandelier(h1_df, spread, UNIT_LOT, maxloss_cap=0)
        strat_trades['S4_CHAND'] = s4_trades
        lots['S4_CHAND'] = S4_LOT

    all_daily = {}
    for sn, trades in strat_trades.items():
        lot_mult = lots.get(sn, UNIT_LOT) / UNIT_LOT
        for t in trades:
            d = pd.Timestamp(t['exit_time']).date()
            all_daily[d] = all_daily.get(d, 0) + t['pnl'] * lot_mult

    dates = sorted(all_daily.keys())
    daily_arr = np.array([all_daily[d] for d in dates]) if dates else np.array([0.0])
    return daily_arr, dates


def _portfolio_stats(daily):
    sh = _sharpe(daily); dd = _max_dd(daily); pnl = float(np.sum(daily))
    cal = pnl / dd if dd > 0 else 9999
    return {'sharpe': round(sh, 3), 'pnl': round(pnl, 2),
            'max_dd': round(dd, 2), 'calmar': round(cal, 1)}


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    results = {'experiment': 'R133 Production Portfolio v3 Unified Validation'}
    VERSIONS = ['v1', 'v2', 'v3']

    print("=" * 80, flush=True)
    print("  R133 — Production Portfolio v3 Unified Validation", flush=True)
    print("=" * 80, flush=True)
    print("  v1: Baseline (PSAR default, TSMOM unfiltered, 4 strats)", flush=True)
    print("  v2: PSAR optimized params", flush=True)
    print("  v3: v2 + D1 EMA20 TSMOM filter + S4 Chandelier (5 strats)", flush=True)

    print("\n  Loading data...", flush=True)
    h1_df = load_h1()
    bundle = DataBundle.load_default()
    print(f"    H1: {len(h1_df)} bars ({h1_df.index[0]} ~ {h1_df.index[-1]})", flush=True)

    # ═══════════════════════════════════════════════════════════
    # Phase 1: Full-Data Comparison
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 1: Full-Data Comparison (v1 vs v2 vs v3)", flush=True)
    print("=" * 70, flush=True)

    daily_arrays = {}
    phase1 = {}
    print(f"\n  {'Version':<8s} {'Sharpe':>7} {'PnL':>12} {'MaxDD':>10} {'Calmar':>8}", flush=True)
    print(f"  {'─'*50}", flush=True)

    for ver in VERSIONS:
        daily, dates = run_portfolio(h1_df, bundle, version=ver)
        daily_arrays[ver] = daily
        st = _portfolio_stats(daily)
        phase1[ver] = st
        print(f"  {ver:<8s} {st['sharpe']:>7.3f} ${st['pnl']:>11,.0f} ${st['max_dd']:>9,.0f} {st['calmar']:>8.1f}", flush=True)

    v1_sh = phase1['v1']['sharpe']; v3_sh = phase1['v3']['sharpe']
    print(f"\n  v1→v3 Sharpe delta: {v3_sh - v1_sh:+.3f}", flush=True)
    results['phase1_full_data'] = phase1

    # ═══════════════════════════════════════════════════════════
    # Phase 2: Walk-Forward OOS
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 2: Walk-Forward Out-of-Sample (4yr train / 2yr test)", flush=True)
    print("=" * 70, flush=True)

    wf_windows = [
        ("2015-2019 / 2019-2021", "2019-01-01", "2021-01-01"),
        ("2016-2020 / 2020-2022", "2020-01-01", "2022-01-01"),
        ("2017-2021 / 2021-2023", "2021-01-01", "2023-01-01"),
        ("2018-2022 / 2022-2024", "2022-01-01", "2024-01-01"),
        ("2019-2023 / 2023-2025", "2023-01-01", "2025-01-01"),
        ("2020-2024 / 2024-2026", "2024-01-01", "2026-05-01"),
    ]

    wf_results = []
    print(f"\n  {'Window':<30s}", end="", flush=True)
    for ver in VERSIONS:
        print(f" {ver:>7}", end="")
    print(flush=True)
    print(f"  {'─'*52}", flush=True)

    for label, oos_start, oos_end in wf_windows:
        h1_oos = h1_df[(h1_df.index >= oos_start) & (h1_df.index < oos_end)]
        if len(h1_oos) < 500:
            continue

        wf_row = {'window': label}
        vals = []
        for ver in VERSIONS:
            daily, _ = run_portfolio(h1_oos, bundle, version=ver)
            sh = _sharpe(daily)
            wf_row[ver] = round(sh, 3)
            vals.append(sh)
        wf_results.append(wf_row)
        print(f"  {label:<30s}", end="", flush=True)
        for v in vals:
            print(f" {v:>7.3f}", end="")
        print(flush=True)

    results['phase2_walkforward'] = wf_results

    if wf_results:
        print(f"\n  OOS Mean Sharpe:", flush=True)
        for ver in VERSIONS:
            vals = [w[ver] for w in wf_results]
            print(f"    {ver}: {np.mean(vals):.3f} (min={min(vals):.3f})", flush=True)

    # ═══════════════════════════════════════════════════════════
    # Phase 3: 8-Fold K-Fold
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 3: 8-Fold Time-Based K-Fold", flush=True)
    print("=" * 70, flush=True)

    fold_starts = pd.date_range("2015-01-01", "2026-01-01", periods=9)
    kfold_results = {ver: [] for ver in VERSIONS}

    for fi in range(8):
        fs = str(fold_starts[fi].date()); fe = str(fold_starts[fi+1].date())
        h1_fold = h1_df[(h1_df.index >= fs) & (h1_df.index < fe)]
        if len(h1_fold) < 200:
            continue
        for ver in VERSIONS:
            daily, _ = run_portfolio(h1_fold, bundle, version=ver)
            kfold_results[ver].append(round(_sharpe(daily), 3))

    phase3 = {}
    print(f"\n  {'Version':<8s} {'Folds':>42s} {'Mean':>6} {'Min':>6} {'Pos':>5}", flush=True)
    print(f"  {'─'*70}", flush=True)
    for ver in VERSIONS:
        sharpes = kfold_results[ver]
        mean_sh = np.mean(sharpes) if sharpes else 0
        min_sh = min(sharpes) if sharpes else 0
        pos_count = sum(1 for s in sharpes if s > 0)
        fold_str = ','.join(f'{s:.1f}' for s in sharpes)
        print(f"  {ver:<8s} [{fold_str:>40s}] {mean_sh:>6.2f} {min_sh:>6.2f} {pos_count:>3}/{len(sharpes)}", flush=True)
        phase3[ver] = {'fold_sharpes': sharpes, 'mean': round(mean_sh, 3),
                       'min': round(min_sh, 3), 'positive': pos_count, 'total': len(sharpes),
                       'pass': pos_count >= len(sharpes) - 1}
    results['phase3_kfold'] = phase3

    # ═══════════════════════════════════════════════════════════
    # Phase 4: Yearly Stability
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 4: Yearly Stability", flush=True)
    print("=" * 70, flush=True)

    years = range(2015, 2027)
    yearly = {ver: {} for ver in VERSIONS}

    for yr in years:
        yr_s = f"{yr}-01-01"; yr_e = f"{yr+1}-01-01"
        h1_yr = h1_df[(h1_df.index >= yr_s) & (h1_df.index < yr_e)]
        if len(h1_yr) < 200:
            continue
        for ver in VERSIONS:
            daily, _ = run_portfolio(h1_yr, bundle, version=ver)
            sh = _sharpe(daily); pnl = float(np.sum(daily))
            yearly[ver][yr] = {'sharpe': round(sh, 3), 'pnl': round(pnl, 0)}

    print(f"\n  {'Year':>6}", end="", flush=True)
    for ver in VERSIONS:
        print(f"  {'Shp_'+ver:>10} {'PnL_'+ver:>10}", end="")
    print(flush=True)
    print(f"  {'─'*6}" + "  " + "  ".join("─" * 21 for _ in VERSIONS), flush=True)

    for yr in years:
        line = f"  {yr:>6}"
        any_data = False
        for ver in VERSIONS:
            if yr in yearly[ver]:
                d = yearly[ver][yr]
                line += f"  {d['sharpe']:>10.3f} ${d['pnl']:>9,.0f}"
                any_data = True
            else:
                line += f"  {'N/A':>10} {'N/A':>10}"
        if any_data:
            print(line, flush=True)

    phase4 = {}
    for ver in VERSIONS:
        neg = sum(1 for d in yearly[ver].values() if d['pnl'] <= 0)
        phase4[ver] = {'yearly': {str(k): v for k, v in yearly[ver].items()},
                       'negative_years': neg,
                       'pass': neg <= 2}
    results['phase4_yearly'] = phase4

    # ═══════════════════════════════════════════════════════════
    # Phase 5: Monte Carlo (1000 paths)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 5: Monte Carlo Robustness (1000 paths)", flush=True)
    print("=" * 70, flush=True)

    rng = np.random.RandomState(42)
    mc_results = {}

    for ver in VERSIONS:
        daily = daily_arrays[ver]
        mc_sharpes = []
        for _ in range(1000):
            perturbed = daily * rng.uniform(0.92, 1.08, size=len(daily))
            mask = rng.random(len(daily)) > 0.02
            perturbed = perturbed * mask
            mc_sharpes.append(_sharpe(perturbed))

        mc_arr = np.array(mc_sharpes)
        mc_results[ver] = {
            'p5': round(float(np.percentile(mc_arr, 5)), 3),
            'p25': round(float(np.percentile(mc_arr, 25)), 3),
            'p50': round(float(np.percentile(mc_arr, 50)), 3),
            'p75': round(float(np.percentile(mc_arr, 75)), 3),
            'p95': round(float(np.percentile(mc_arr, 95)), 3),
            'pct_positive': round(float(np.mean(mc_arr > 0) * 100), 1),
        }
        print(f"  {ver}: P5={mc_results[ver]['p5']:.3f}  P50={mc_results[ver]['p50']:.3f}  "
              f"P95={mc_results[ver]['p95']:.3f}  pos={mc_results[ver]['pct_positive']:.0f}%", flush=True)

    results['phase5_monte_carlo'] = mc_results

    # ═══════════════════════════════════════════════════════════
    # Phase 6: Probability of Backtest Overfitting (PBO)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 6: Probability of Backtest Overfitting (1000 shuffles)", flush=True)
    print("=" * 70, flush=True)

    pbo_results = {}
    for ver in VERSIONS:
        daily = daily_arrays[ver]
        real_sharpe = _sharpe(daily)
        n_better = 0
        for _ in range(1000):
            shuffled = rng.permutation(daily)
            if _sharpe(shuffled) >= real_sharpe:
                n_better += 1
        pbo = n_better / 1000
        pbo_results[ver] = {'real_sharpe': round(real_sharpe, 3), 'pbo': round(pbo, 3),
                            'pass': pbo < 0.50}
        status = "PASS" if pbo < 0.50 else "FAIL"
        print(f"  {ver}: PBO={pbo:.3f} ({status})  Sharpe={real_sharpe:.3f}", flush=True)

    results['phase6_pbo'] = pbo_results

    # ═══════════════════════════════════════════════════════════
    # Phase 7: Final Verdict
    # ═══════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    results['elapsed_s'] = round(elapsed, 1)

    print("\n" + "=" * 80, flush=True)
    print("  R133 FINAL VERDICT", flush=True)
    print("=" * 80, flush=True)

    verdicts = {}
    for ver in VERSIONS:
        s1 = phase1[ver]
        s3 = phase3[ver]
        s5 = mc_results[ver]
        s6 = pbo_results[ver]
        s4y = phase4[ver]

        wf_sharpes = [w[ver] for w in wf_results]
        wf_mean = np.mean(wf_sharpes) if wf_sharpes else 0
        wf_min = min(wf_sharpes) if wf_sharpes else 0

        all_pass = s3['pass'] and s6['pass'] and s4y['pass'] and wf_min > 0
        verdict = "VALIDATED" if all_pass else "PARTIAL"
        verdicts[ver] = verdict

        print(f"\n  {ver}:", flush=True)
        print(f"    Full-data:     Sharpe={s1['sharpe']:.3f}, PnL=${s1['pnl']:,.0f}, MaxDD=${s1['max_dd']:,.0f}", flush=True)
        print(f"    WF-OOS mean:   {wf_mean:.3f} (min={wf_min:.3f})", flush=True)
        print(f"    K-Fold:        mean={s3['mean']:.3f}, {s3['positive']}/{s3['total']} positive ({'PASS' if s3['pass'] else 'FAIL'})", flush=True)
        print(f"    MC P5 Sharpe:  {s5['p5']:.3f}, positive={s5['pct_positive']:.0f}%", flush=True)
        print(f"    PBO:           {s6['pbo']:.3f} ({'PASS' if s6['pass'] else 'FAIL'})", flush=True)
        print(f"    Yearly:        {s4y['negative_years']} negative years ({'PASS' if s4y['pass'] else 'FAIL'})", flush=True)
        print(f"    -> {verdict}", flush=True)

    results['verdicts'] = verdicts

    # Recommendation
    v3_better = (phase1['v3']['sharpe'] > phase1['v1']['sharpe']
                 and verdicts['v3'] == 'VALIDATED')
    if v3_better:
        rec = ("UPGRADE to v3: +S4 Chandelier, D1 EMA20 TSMOM filter, "
               f"PSAR optimized. Sharpe {phase1['v1']['sharpe']:.3f} -> {phase1['v3']['sharpe']:.3f}")
    elif verdicts['v2'] == 'VALIDATED' and phase1['v2']['sharpe'] > phase1['v1']['sharpe']:
        rec = f"UPGRADE to v2: PSAR optimized. Sharpe {phase1['v1']['sharpe']:.3f} -> {phase1['v2']['sharpe']:.3f}"
    else:
        rec = "KEEP v1: improvements not validated, stay with current production config"
    print(f"\n  RECOMMENDATION: {rec}", flush=True)
    results['recommendation'] = rec

    out_file = OUTPUT_DIR / "r133_results.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)
    print(f"  Saved: {out_file}", flush=True)
    print("=" * 80, flush=True)


if __name__ == '__main__':
    main()

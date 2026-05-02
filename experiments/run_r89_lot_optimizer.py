#!/usr/bin/env python3
"""
R89 — Portfolio Lot Size Optimization ($5,000 Capital)
=======================================================
Finds the optimal lot combination for L8_MAX / PSAR / TSMOM / SESS_BO
that maximizes portfolio Sharpe while keeping combined MaxDD <= $1,000.

Phase 1: Run each strategy at unit lot (0.01) with recommended Cap
Phase 2: Grid search 15^4 = 50,625 lot combos via linear PnL scaling
Phase 3: K-Fold validation on top 5 combos
Phase 4: Sensitivity analysis on winning combo
"""
import sys, os, io, time, json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from itertools import product

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

OUTPUT_DIR = Path("results/r89_lot_optimizer")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100
CAPITAL = 5000
MAX_DD_LIMIT = 1000
SPREAD = 0.30
UNIT_LOT = 0.01

# Cap settings from R88 v2
CAPS = {
    'L8_MAX': 35,
    'PSAR':    5,
    'TSMOM':   0,
    'SESS_BO': 35,
}

LOT_GRID = [round(x * 0.01, 2) for x in range(1, 16)]  # 0.01 to 0.15

FOLDS = [
    ("Fold1", "2015-01-01", "2017-01-01"),
    ("Fold2", "2017-01-01", "2019-01-01"),
    ("Fold3", "2019-01-01", "2021-01-01"),
    ("Fold4", "2021-01-01", "2023-01-01"),
    ("Fold5", "2023-01-01", "2025-01-01"),
    ("Fold6", "2025-01-01", "2026-05-01"),
]

STRAT_ORDER = ['L8_MAX', 'PSAR', 'TSMOM', 'SESS_BO']


def fmt(v):
    return f"${v:>10,.0f}" if v >= 0 else f"-${abs(v):>9,.0f}"


# ═══════════════════════════════════════════════════════════════
# Shared helpers (from R88)
# ═══════════════════════════════════════════════════════════════

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
# Strategy backtests (from R88)
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
    df['fma'] = df['Close'].rolling(fast).mean()
    df['sma'] = df['Close'].rolling(slow).mean()
    df = df.dropna(subset=['ATR', 'fma', 'sma'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; fm = df['fma'].values; sm = df['sma'].values
    times = df.index; n = len(df)
    trades = []; pos = None; last_exit = -999
    for i in range(1, n):
        if pos is not None:
            result = _run_exit_with_cap(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                                        sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            if pos['dir'] == 'BUY' and fm[i] < sm[i]:
                pnl = (c[i]-pos['entry']-spread)*lot*PV
                trades.append(_mk(pos, c[i], times[i], "Reversal", i, pnl)); pos = None; last_exit = i; continue
            elif pos['dir'] == 'SELL' and fm[i] > sm[i]:
                pnl = (pos['entry']-c[i]-spread)*lot*PV
                trades.append(_mk(pos, c[i], times[i], "Reversal", i, pnl)); pos = None; last_exit = i; continue
            continue
        if i-last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        if fm[i] > sm[i] and fm[i-1] <= sm[i-1]:
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif fm[i] < sm[i] and fm[i-1] >= sm[i-1]:
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
    print("=" * 80)
    print("  R89 — Portfolio Lot Size Optimization")
    print(f"  Capital: ${CAPITAL:,}  |  MaxDD limit: ${MAX_DD_LIMIT:,}  |  Grid: {len(LOT_GRID)}^4 = {len(LOT_GRID)**4:,} combos")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80, flush=True)

    from backtest.runner import load_m15, load_h1_aligned, H1_CSV_PATH, DataBundle

    # ── Phase 1: Run each strategy at unit lot ──
    print(f"\n{'='*80}")
    print(f"  Phase 1: Per-Strategy Backtest at unit lot ({UNIT_LOT})")
    print(f"{'='*80}\n", flush=True)

    print("  Loading data...", flush=True)
    m15_raw = load_m15()
    h1_df = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])
    print(f"  H1: {len(h1_df)} bars ({h1_df.index[0]} ~ {h1_df.index[-1]})", flush=True)

    print("  Preparing L8_MAX DataBundle...", flush=True)
    l8_bundle = DataBundle.load_custom()
    print("  L8 bundle ready.\n", flush=True)

    # Run all strategies at unit lot with their recommended Cap
    unit_trades = {}
    unit_dailies = {}
    unit_stats = {}

    # H1 strategies
    h1_strats = {
        'PSAR':    (bt_psar, {}),
        'TSMOM':   (bt_tsmom, {}),
        'SESS_BO': (bt_sess_bo, {}),
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
        print(f"    {name:>8}: {len(trades)} trades, Sharpe={unit_stats[name]['sharpe']:.2f}, "
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
    print(f"    {'L8_MAX':>8}: {len(trades)} trades, Sharpe={unit_stats['L8_MAX']['sharpe']:.2f}, "
          f"PnL={fmt(unit_stats['L8_MAX']['pnl'])}, MaxDD={fmt(unit_stats['L8_MAX']['max_dd'])}, "
          f"Cap=${cap}", flush=True)

    print(f"\n  Phase 1 complete. Unit-lot baselines ready.", flush=True)

    # ── Phase 2: Grid Search ──
    print(f"\n{'='*80}")
    print(f"  Phase 2: Lot Combination Grid Search ({len(LOT_GRID)**4:,} combos)")
    print(f"{'='*80}\n", flush=True)

    results = []
    total = len(LOT_GRID) ** 4
    checked = 0
    feasible = 0

    for l8_lot, psar_lot, tsmom_lot, sess_lot in product(LOT_GRID, repeat=4):
        checked += 1
        if checked % 10000 == 0:
            print(f"    Progress: {checked:,}/{total:,} checked, {feasible} feasible...", flush=True)

        lots = {'L8_MAX': l8_lot, 'PSAR': psar_lot, 'TSMOM': tsmom_lot, 'SESS_BO': sess_lot}
        port_daily = build_portfolio_daily(unit_dailies, lots)

        dd = max_dd(port_daily)
        if dd > MAX_DD_LIMIT:
            continue

        feasible += 1
        sh = sharpe(port_daily)
        pnl = float(np.sum(port_daily))
        cv = cvar99(port_daily)

        results.append({
            'lots': lots,
            'sharpe': round(sh, 3),
            'pnl': round(pnl, 2),
            'max_dd': round(dd, 2),
            'cvar99': round(cv, 2),
            'dd_pct': round(dd / CAPITAL * 100, 1),
            'annual_return_pct': round(pnl / (2754/252) / CAPITAL * 100, 1),
        })

    print(f"\n  Grid search complete: {checked:,} tested, {feasible:,} feasible (DD<=${MAX_DD_LIMIT})", flush=True)

    # Sort by Sharpe descending
    results.sort(key=lambda x: x['sharpe'], reverse=True)

    # Print top 20
    print(f"\n  Top 20 lot combinations by Sharpe:")
    print(f"  {'Rank':>4} {'L8_MAX':>7} {'PSAR':>6} {'TSMOM':>6} {'SESSBO':>7} "
          f"{'Sharpe':>7} {'PnL':>12} {'MaxDD':>10} {'DD%':>6} {'CVaR99':>8} {'AnnRet%':>8}")
    print(f"  {'-'*4} {'-'*7} {'-'*6} {'-'*6} {'-'*7} {'-'*7} {'-'*12} {'-'*10} {'-'*6} {'-'*8} {'-'*8}")
    for i, r in enumerate(results[:20]):
        print(f"  {i+1:>4} {r['lots']['L8_MAX']:>7.2f} {r['lots']['PSAR']:>6.2f} "
              f"{r['lots']['TSMOM']:>6.2f} {r['lots']['SESS_BO']:>7.2f} "
              f"{r['sharpe']:>7.3f} {fmt(r['pnl']):>12} {fmt(r['max_dd']):>10} "
              f"{r['dd_pct']:>5.1f}% {r['cvar99']:>8.2f} {r['annual_return_pct']:>7.1f}%", flush=True)

    # ── Phase 3: K-Fold Validation on top 5 ──
    print(f"\n{'='*80}")
    print(f"  Phase 3: K-Fold 6-Fold Validation (top 5 combos)")
    print(f"{'='*80}\n", flush=True)

    # Pre-compute per-fold unit dailies for H1 strategies
    fold_unit_dailies = {}
    for fold_name, start, end in FOLDS:
        fold_unit_dailies[fold_name] = {}
        h1_fold = h1_df[start:end]
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
        label = f"L8={lots['L8_MAX']:.2f}_P={lots['PSAR']:.2f}_T={lots['TSMOM']:.2f}_S={lots['SESS_BO']:.2f}"
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
            'pass_4of6': passed,
        }
        status = "PASS" if passed else "FAIL"
        print(f"  #{rank+1} {label}:  {positive}/6 positive, mean={mean_sh:.2f}  [{status}]", flush=True)
        print(f"       folds={[round(s, 1) for s in fold_sharpes]}", flush=True)

    # ── Phase 4: Sensitivity Analysis ──
    print(f"\n{'='*80}")
    print(f"  Phase 4: Sensitivity Analysis (best combo)")
    print(f"{'='*80}\n", flush=True)

    # Find best combo that passes K-Fold
    winner = None
    for combo in results[:10]:
        lots = combo['lots']
        label = f"L8={lots['L8_MAX']:.2f}_P={lots['PSAR']:.2f}_T={lots['TSMOM']:.2f}_S={lots['SESS_BO']:.2f}"
        if label in kfold_results and kfold_results[label]['pass_4of6']:
            winner = combo
            break
    if winner is None and results:
        winner = results[0]

    sensitivity = {}
    if winner:
        base_lots = winner['lots']
        print(f"  Winner: L8={base_lots['L8_MAX']:.2f}  PSAR={base_lots['PSAR']:.2f}  "
              f"TSMOM={base_lots['TSMOM']:.2f}  SESS_BO={base_lots['SESS_BO']:.2f}", flush=True)
        print(f"  Sharpe={winner['sharpe']:.3f}  PnL={fmt(winner['pnl'])}  MaxDD={fmt(winner['max_dd'])}\n", flush=True)

        print(f"  {'Strategy':<10} {'Lot-0.01':>10} {'BaseLot':>10} {'Lot+0.01':>10} {'Impact':>10}")
        print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

        for name in STRAT_ORDER:
            row = {}
            for delta_label, delta in [('-0.01', -0.01), ('base', 0), ('+0.01', 0.01)]:
                test_lots = dict(base_lots)
                new_lot = round(test_lots[name] + delta, 2)
                if new_lot < 0.01 or new_lot > 0.20:
                    row[delta_label] = None
                    continue
                test_lots[name] = new_lot
                port_daily = build_portfolio_daily(unit_dailies, test_lots)
                row[delta_label] = round(sharpe(port_daily), 3)

            def sv(v):
                return f"{v:.3f}" if v is not None else "N/A"

            impact = ""
            if row.get('-0.01') and row.get('+0.01') and row.get('base'):
                avg_neighbor = (row['-0.01'] + row['+0.01']) / 2
                diff = row['base'] - avg_neighbor
                impact = f"{'+'if diff>=0 else ''}{diff:.3f}"

            sensitivity[name] = row
            print(f"  {name:<10} {sv(row.get('-0.01')):>10} {sv(row.get('base')):>10} "
                  f"{sv(row.get('+0.01')):>10} {impact:>10}", flush=True)

        # Per-strategy contribution at winning lots
        print(f"\n  Per-strategy contribution at winning lot:")
        print(f"  {'Strategy':<10} {'Lot':>6} {'PnL':>12} {'MaxDD':>10} {'Sharpe':>8} {'%ofPnL':>8}")
        print(f"  {'-'*10} {'-'*6} {'-'*12} {'-'*10} {'-'*8} {'-'*8}")
        total_pnl = 0
        strat_pnls = {}
        for name in STRAT_ORDER:
            lot = base_lots[name]
            mult = lot / UNIT_LOT
            ds = unit_dailies[name]
            scaled = ds.values * mult
            pnl_v = float(np.sum(scaled))
            strat_pnls[name] = pnl_v
            total_pnl += pnl_v
        for name in STRAT_ORDER:
            lot = base_lots[name]
            mult = lot / UNIT_LOT
            ds = unit_dailies[name]
            scaled = ds.values * mult
            pnl_v = strat_pnls[name]
            sh_v = sharpe(scaled)
            dd_v = max_dd(scaled)
            pct = pnl_v / total_pnl * 100 if total_pnl > 0 else 0
            print(f"  {name:<10} {lot:>6.2f} {fmt(pnl_v):>12} {fmt(dd_v):>10} "
                  f"{sh_v:>8.2f} {pct:>7.1f}%", flush=True)

    # ── Summary ──
    print(f"\n{'='*80}")
    print(f"  FINAL RECOMMENDATION")
    print(f"{'='*80}\n")

    if winner:
        lots = winner['lots']
        total_margin = sum(lots[n] for n in STRAT_ORDER) * 100 * 2500 * 0.005
        print(f"  Capital: ${CAPITAL:,}")
        print(f"  MaxDD Limit: ${MAX_DD_LIMIT:,} ({MAX_DD_LIMIT/CAPITAL*100:.0f}%)")
        print(f"  ")
        print(f"  Recommended lot sizes:")
        for name in STRAT_ORDER:
            lot = lots[name]
            cap = CAPS[name]
            cap_str = f"Cap${cap}" if cap > 0 else "NoCap"
            print(f"    {name:<10}  {lot:.2f} lot  ({cap_str})")
        print(f"  ")
        print(f"  Portfolio metrics (2015-2026 backtest):")
        print(f"    Sharpe:       {winner['sharpe']:.3f}")
        print(f"    Total PnL:    {fmt(winner['pnl'])}")
        print(f"    MaxDD:        {fmt(winner['max_dd'])}  ({winner['dd_pct']:.1f}% of capital)")
        print(f"    CVaR99:       {fmt(winner['cvar99'])}")
        print(f"    Annual Return: {winner['annual_return_pct']:.1f}% of capital")

    elapsed = time.time() - t0
    print(f"\n{'='*80}")
    print(f"  R89 COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"{'='*80}", flush=True)

    # Save
    output = {
        'config': {
            'capital': CAPITAL,
            'max_dd_limit': MAX_DD_LIMIT,
            'spread': SPREAD,
            'caps': CAPS,
            'lot_grid': LOT_GRID,
            'unit_lot': UNIT_LOT,
        },
        'unit_stats': unit_stats,
        'top_20': results[:20],
        'kfold': kfold_results,
        'sensitivity': sensitivity,
        'winner': winner,
        'total_feasible': feasible,
        'total_tested': total,
        'elapsed_s': round(elapsed, 1),
    }
    with open(OUTPUT_DIR / "r89_results.json", 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Saved: {OUTPUT_DIR}/r89_results.json", flush=True)


if __name__ == "__main__":
    main()

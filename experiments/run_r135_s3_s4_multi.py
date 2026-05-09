#!/usr/bin/env python3
"""
R135 — Dual Thrust + Chandelier Multi-Strategy Cross-Validation
=================================================================
Phases:
  1. Load H1 data. Run S3 Dual Thrust and S4 Chandelier individually
  2. Signal overlap analysis: % overlapping signals, time distribution
  3. Portfolio analysis: S3+S4, S3+S4+L8_MAX combined
  4. Lot grid search: S3 lot (0.01-0.10) + S4 lot (0.01-0.10), MaxDD < $800
  5. Spread stress: $0.30, $0.50, $0.80, $1.00, $1.50
  6. K-Fold 5-fold of best portfolio
  7. Monte Carlo 1000 paths
"""
import sys, os, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

from backtest.runner import load_csv, DataBundle, run_variant, LIVE_PARITY_KWARGS
from indicators import calc_dual_thrust_range, calc_chandelier

OUTPUT_DIR = Path("results/r135_s3_s4_multi")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100; SPREAD = 0.30; UNIT_LOT = 0.01

H1_CANDIDATES = [
    Path("data/download/xauusd-h1-bid-2015-01-01-2026-04-27.csv"),
    Path("data/download/xauusd-h1-bid-2015-01-01-2026-04-10.csv"),
    Path("data/download/xauusd-h1-bid-2015-01-01-2026-03-25.csv"),
]

FOLDS = [
    ("Fold1", "2015-01-01", "2017-03-01"),
    ("Fold2", "2017-03-01", "2019-06-01"),
    ("Fold3", "2019-06-01", "2021-09-01"),
    ("Fold4", "2021-09-01", "2023-12-01"),
    ("Fold5", "2023-12-01", "2027-01-01"),
]

N_MC = 1000

t0 = time.time()


# ═══════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════

def compute_atr(df, period=14):
    h, l, c = df['High'], df['Low'], df['Close']
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


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


def _compute_stats(trades):
    if not trades:
        return {'n': 0, 'sharpe': 0, 'pnl': 0, 'wr': 0, 'max_dd': 0, 'calmar': 0}
    daily = _trades_to_daily(trades)
    pnls = [t['pnl'] for t in trades]
    n = len(trades)
    sh = _sharpe(daily)
    dd = _max_dd(daily)
    pnl = sum(pnls)
    return {
        'n': n, 'sharpe': round(sh, 3), 'pnl': round(pnl, 2),
        'wr': round(sum(1 for p in pnls if p > 0) / n * 100, 1) if n else 0,
        'max_dd': round(dd, 2),
        'calmar': round(pnl / dd, 2) if dd > 0 else 9999,
    }


def _trades_to_daily_series(trades):
    if not trades:
        return pd.Series(dtype=float)
    daily = {}
    for t in trades:
        d = pd.Timestamp(t['exit_time']).date()
        daily[d] = daily.get(d, 0) + t['pnl']
    dates = sorted(daily.keys())
    return pd.Series([daily[d] for d in dates], index=pd.DatetimeIndex(dates))


# ═══════════════════════════════════════════════════════════════
# Strategy backtests: S3 Dual Thrust, S4 Chandelier
# ═══════════════════════════════════════════════════════════════

def bt_s3_dual_thrust(h1_df, spread, lot, n_bars=6, k_up=0.5, k_down=0.5,
                      sl_atr=4.5, tp_atr=8.0, trail_act=0.14, trail_dist=0.025,
                      max_hold=20, cap=35, start=None, end=None):
    """S3 Dual Thrust signal: breakout from daily open ± k * DT range."""
    df = h1_df.copy()
    df['ATR14'] = compute_atr(df, 14)
    dt_range = calc_dual_thrust_range(df, n_bars)
    daily_open = df.groupby(df.index.date)['Open'].transform('first')
    sig = pd.Series(0, index=df.index)
    sig[df['Close'] > daily_open + k_up * dt_range] = 1
    sig[df['Close'] < daily_open - k_down * dt_range] = -1

    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr_arr = df['ATR14'].values; times = df.index; n = len(df)
    sig_arr = sig.values; dates = df.index.date

    trades = []; pos = None; last_exit = -999
    for i in range(1, n):
        if start and str(dates[i]) < start:
            continue
        if end and str(dates[i]) > end:
            break
        if pos is not None:
            result = _run_exit(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                               sl_atr, tp_atr, trail_act, trail_dist, max_hold, cap)
            if result:
                trades.append(result); pos = None; last_exit = i
            continue
        if i - last_exit < 2:
            continue
        if np.isnan(atr_arr[i]) or atr_arr[i] < 0.1:
            continue
        if sig_arr[i] == 1 and sig_arr[i-1] != 1:
            pos = {'dir': 'BUY', 'entry': c[i] + spread/2, 'bar': i, 'time': times[i], 'atr': atr_arr[i]}
        elif sig_arr[i] == -1 and sig_arr[i-1] != -1:
            pos = {'dir': 'SELL', 'entry': c[i] - spread/2, 'bar': i, 'time': times[i], 'atr': atr_arr[i]}
    return trades


def bt_s4_chandelier(h1_df, spread, lot, period=22, mult=3.0,
                     sl_atr=4.5, tp_atr=8.0, trail_act=0.14, trail_dist=0.025,
                     max_hold=20, cap=35, start=None, end=None):
    """S4 Chandelier Exit signal: flip above Chand_long (bull) or below Chand_short (bear) + EMA100."""
    df = h1_df.copy()
    df['ATR14'] = compute_atr(df, 14)
    ch = calc_chandelier(df, period, mult)
    ema100 = df['Close'].ewm(span=100).mean()
    above_long = df['Close'] > ch['Chand_long']
    flip_bull = above_long & (~above_long.shift(1).fillna(False))
    below_short = df['Close'] < ch['Chand_short']
    flip_bear = below_short & (~below_short.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index)
    sig[flip_bull & (df['Close'] > ema100)] = 1
    sig[flip_bear & (df['Close'] < ema100)] = -1

    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr_arr = df['ATR14'].values; times = df.index; n = len(df)
    sig_arr = sig.values; dates = df.index.date

    trades = []; pos = None; last_exit = -999
    for i in range(1, n):
        if start and str(dates[i]) < start:
            continue
        if end and str(dates[i]) > end:
            break
        if pos is not None:
            result = _run_exit(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                               sl_atr, tp_atr, trail_act, trail_dist, max_hold, cap)
            if result:
                trades.append(result); pos = None; last_exit = i
            continue
        if i - last_exit < 2:
            continue
        if np.isnan(atr_arr[i]) or atr_arr[i] < 0.1:
            continue
        if sig_arr[i] == 1:
            pos = {'dir': 'BUY', 'entry': c[i] + spread/2, 'bar': i, 'time': times[i], 'atr': atr_arr[i]}
        elif sig_arr[i] == -1:
            pos = {'dir': 'SELL', 'entry': c[i] - spread/2, 'bar': i, 'time': times[i], 'atr': atr_arr[i]}
    return trades


def merge_portfolio(trade_lists, lot_weights=None):
    """Merge multiple trade lists with optional lot scaling into a single portfolio."""
    all_trades = []
    for idx, trades in enumerate(trade_lists):
        scale = lot_weights[idx] if lot_weights else 1.0
        for t in trades:
            t2 = dict(t)
            t2['pnl'] = t['pnl'] * scale
            all_trades.append(t2)
    return all_trades


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 80, flush=True)
    print("  R135 — Dual Thrust + Chandelier Multi-Strategy Cross-Validation", flush=True)
    print("=" * 80, flush=True)

    # ── Load Data ──
    csv_path = next((p for p in H1_CANDIDATES if p.exists()), H1_CANDIDATES[0])
    print(f"\n  Loading H1: {csv_path}", flush=True)
    h1 = load_csv(str(csv_path))
    print(f"  H1 loaded: {len(h1)} bars ({h1.index[0]} → {h1.index[-1]})", flush=True)

    all_results = {
        'experiment': 'R135 Dual Thrust + Chandelier Multi-Strategy',
        'data_bars': len(h1),
    }

    # ════════════════════════════════════════════════════════════════
    # Phase 1: Run S3 and S4 individually
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 1: Individual Strategy Backtests", flush=True)
    print("=" * 70, flush=True)

    print("  Running S3 Dual Thrust (n=6, k_up=0.5, k_down=0.5)...", flush=True)
    s3_trades = bt_s3_dual_thrust(h1, SPREAD, UNIT_LOT)
    s3_stats = _compute_stats(s3_trades)
    print(f"    S3: n={s3_stats['n']}, Sharpe={s3_stats['sharpe']:.3f}, "
          f"PnL=${s3_stats['pnl']:.0f}, WR={s3_stats['wr']:.1f}%, "
          f"MaxDD=${s3_stats['max_dd']:.0f}", flush=True)

    print("  Running S4 Chandelier (period=22, mult=3.0)...", flush=True)
    s4_trades = bt_s4_chandelier(h1, SPREAD, UNIT_LOT)
    s4_stats = _compute_stats(s4_trades)
    print(f"    S4: n={s4_stats['n']}, Sharpe={s4_stats['sharpe']:.3f}, "
          f"PnL=${s4_stats['pnl']:.0f}, WR={s4_stats['wr']:.1f}%, "
          f"MaxDD=${s4_stats['max_dd']:.0f}", flush=True)

    all_results['phase1_individual'] = {'S3': s3_stats, 'S4': s4_stats}

    # ════════════════════════════════════════════════════════════════
    # Phase 2: Signal overlap analysis
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 2: Signal Overlap Analysis", flush=True)
    print("=" * 70, flush=True)

    s3_entry_dates = set()
    for t in s3_trades:
        d = pd.Timestamp(t['entry_time']).date()
        s3_entry_dates.add(d)

    s4_entry_dates = set()
    for t in s4_trades:
        d = pd.Timestamp(t['entry_time']).date()
        s4_entry_dates.add(d)

    overlap = s3_entry_dates & s4_entry_dates
    union = s3_entry_dates | s4_entry_dates
    overlap_pct = len(overlap) / len(union) * 100 if union else 0

    print(f"  S3 entry days: {len(s3_entry_dates)}", flush=True)
    print(f"  S4 entry days: {len(s4_entry_dates)}", flush=True)
    print(f"  Overlapping days: {len(overlap)} ({overlap_pct:.1f}%)", flush=True)
    print(f"  Union days: {len(union)}", flush=True)

    s3_hours = [pd.Timestamp(t['entry_time']).hour for t in s3_trades]
    s4_hours = [pd.Timestamp(t['entry_time']).hour for t in s4_trades]
    print(f"\n  S3 entry hour distribution: mean={np.mean(s3_hours):.1f}h, "
          f"median={np.median(s3_hours):.0f}h", flush=True)
    print(f"  S4 entry hour distribution: mean={np.mean(s4_hours):.1f}h, "
          f"median={np.median(s4_hours):.0f}h", flush=True)

    s3_daily = _trades_to_daily_series(s3_trades)
    s4_daily = _trades_to_daily_series(s4_trades)
    combined_idx = s3_daily.index.union(s4_daily.index)
    s3_aligned = s3_daily.reindex(combined_idx, fill_value=0)
    s4_aligned = s4_daily.reindex(combined_idx, fill_value=0)
    corr_s3_s4 = s3_aligned.corr(s4_aligned)
    print(f"\n  Daily PnL correlation(S3, S4): {corr_s3_s4:.4f}", flush=True)

    all_results['phase2_overlap'] = {
        's3_days': len(s3_entry_dates), 's4_days': len(s4_entry_dates),
        'overlap_days': len(overlap), 'overlap_pct': round(overlap_pct, 1),
        'daily_pnl_corr': round(corr_s3_s4, 4),
    }

    # ════════════════════════════════════════════════════════════════
    # Phase 3: Portfolio analysis (S3+S4, S3+S4+L8_MAX)
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 3: Portfolio Combinations", flush=True)
    print("=" * 70, flush=True)

    s3s4_trades = s3_trades + s4_trades
    s3s4_stats = _compute_stats(s3s4_trades)
    print(f"  S3+S4 equal weight: n={s3s4_stats['n']}, Sharpe={s3s4_stats['sharpe']:.3f}, "
          f"PnL=${s3s4_stats['pnl']:.0f}, MaxDD=${s3s4_stats['max_dd']:.0f}", flush=True)

    l8_trades = []
    try:
        bundle = DataBundle.load_default()
        result = run_variant(bundle, "L8_MAX", verbose=False, **LIVE_PARITY_KWARGS)
        raw_trades = result.get('_trades', [])
        for t in raw_trades:
            l8_trades.append({
                'dir': t.direction, 'entry': t.entry_price, 'exit': t.exit_price,
                'entry_time': t.entry_time, 'exit_time': t.exit_time,
                'pnl': t.pnl, 'reason': t.exit_reason, 'bars': 0,
            })
        l8_stats = _compute_stats(l8_trades)
        print(f"  L8_MAX baseline: n={l8_stats['n']}, Sharpe={l8_stats['sharpe']:.3f}, "
              f"PnL=${l8_stats['pnl']:.0f}", flush=True)
    except Exception as e:
        print(f"  WARNING: L8_MAX load failed: {e}", flush=True)
        l8_stats = _compute_stats([])

    trio_trades = s3_trades + s4_trades + l8_trades
    trio_stats = _compute_stats(trio_trades)
    print(f"  S3+S4+L8_MAX: n={trio_stats['n']}, Sharpe={trio_stats['sharpe']:.3f}, "
          f"PnL=${trio_stats['pnl']:.0f}, MaxDD=${trio_stats['max_dd']:.0f}", flush=True)

    all_results['phase3_portfolio'] = {
        'S3_S4': s3s4_stats,
        'L8_MAX': l8_stats,
        'S3_S4_L8MAX': trio_stats,
    }

    # ════════════════════════════════════════════════════════════════
    # Phase 4: Lot grid search
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 4: Lot Grid Search (S3 + S4, MaxDD < $800)", flush=True)
    print("=" * 70, flush=True)

    lot_values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
    lot_grid = []

    print(f"\n  {'S3_lot':>7s}  {'S4_lot':>7s}  {'n':>5s}  {'Sharpe':>7s}  "
          f"{'PnL':>10s}  {'WR':>6s}  {'MaxDD':>8s}", flush=True)
    print("  " + "-" * 60, flush=True)

    for s3_lot, s4_lot in product(lot_values, lot_values):
        s3_t = bt_s3_dual_thrust(h1, SPREAD, s3_lot)
        s4_t = bt_s4_chandelier(h1, SPREAD, s4_lot)
        combined = s3_t + s4_t
        st = _compute_stats(combined)

        if st['max_dd'] > 800:
            continue

        lot_grid.append({'s3_lot': s3_lot, 's4_lot': s4_lot, **st})

        if st['sharpe'] > 0.3 and st['n'] >= 20:
            print(f"  {s3_lot:7.2f}  {s4_lot:7.2f}  {st['n']:5d}  {st['sharpe']:7.3f}  "
                  f"${st['pnl']:>9.0f}  {st['wr']:5.1f}%  ${st['max_dd']:>7.0f}", flush=True)

    lot_grid.sort(key=lambda x: x['sharpe'], reverse=True)
    print(f"\n  Top 10 (MaxDD < $800):", flush=True)
    for i, g in enumerate(lot_grid[:10]):
        print(f"    #{i+1}: S3={g['s3_lot']:.2f}, S4={g['s4_lot']:.2f} -> "
              f"Sharpe={g['sharpe']:.3f}, PnL=${g['pnl']:.0f}, "
              f"MaxDD=${g['max_dd']:.0f}", flush=True)

    all_results['phase4_lot_grid'] = lot_grid[:30]

    best_lot = lot_grid[0] if lot_grid else {'s3_lot': 0.01, 's4_lot': 0.01}

    # ════════════════════════════════════════════════════════════════
    # Phase 5: Spread stress test
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 5: Spread Stress Test", flush=True)
    print("=" * 70, flush=True)

    spread_levels = [0.30, 0.50, 0.80, 1.00, 1.50]
    spread_results = []

    print(f"\n  {'Spread':>8s}  {'n':>5s}  {'Sharpe':>7s}  {'PnL':>10s}  {'WR':>6s}  {'MaxDD':>8s}", flush=True)
    print("  " + "-" * 50, flush=True)

    for sp in spread_levels:
        s3_t = bt_s3_dual_thrust(h1, sp, best_lot['s3_lot'])
        s4_t = bt_s4_chandelier(h1, sp, best_lot['s4_lot'])
        combined = s3_t + s4_t
        st = _compute_stats(combined)
        spread_results.append({'spread': sp, **st})
        print(f"  ${sp:7.2f}  {st['n']:5d}  {st['sharpe']:7.3f}  ${st['pnl']:>9.0f}  "
              f"{st['wr']:5.1f}%  ${st['max_dd']:>7.0f}", flush=True)

    all_results['phase5_spread_stress'] = spread_results

    # ════════════════════════════════════════════════════════════════
    # Phase 6: K-Fold 5-fold validation
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 6: K-Fold 5-Fold Validation (best lot combo)", flush=True)
    print("=" * 70, flush=True)

    kfold_results = []
    print(f"\n  {'Fold':<8s}  {'n':>5s}  {'Sharpe':>7s}  {'PnL':>10s}  {'WR':>6s}  {'MaxDD':>8s}", flush=True)
    print("  " + "-" * 50, flush=True)

    for fold_name, fold_start, fold_end in FOLDS:
        s3_t = bt_s3_dual_thrust(h1, SPREAD, best_lot['s3_lot'],
                                 start=fold_start, end=fold_end)
        s4_t = bt_s4_chandelier(h1, SPREAD, best_lot['s4_lot'],
                                start=fold_start, end=fold_end)
        combined = s3_t + s4_t
        st = _compute_stats(combined)
        kfold_results.append({'fold': fold_name, **st})
        print(f"  {fold_name:<8s}  {st['n']:5d}  {st['sharpe']:7.3f}  ${st['pnl']:>9.0f}  "
              f"{st['wr']:5.1f}%  ${st['max_dd']:>7.0f}", flush=True)

    pos_folds = sum(1 for f in kfold_results if f['sharpe'] > 0)
    avg_sharpe = np.mean([f['sharpe'] for f in kfold_results])
    print(f"\n  Positive folds: {pos_folds}/{len(kfold_results)}", flush=True)
    print(f"  Average fold Sharpe: {avg_sharpe:.3f}", flush=True)

    all_results['phase6_kfold'] = kfold_results
    all_results['phase6_summary'] = {
        'positive_folds': pos_folds, 'total_folds': len(kfold_results),
        'avg_sharpe': round(avg_sharpe, 3),
    }

    # ════════════════════════════════════════════════════════════════
    # Phase 7: Monte Carlo 1000 paths
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 7: Monte Carlo Simulation (1000 paths)", flush=True)
    print("=" * 70, flush=True)

    s3_full = bt_s3_dual_thrust(h1, SPREAD, best_lot['s3_lot'])
    s4_full = bt_s4_chandelier(h1, SPREAD, best_lot['s4_lot'])
    all_pnls = [t['pnl'] for t in s3_full + s4_full]

    if len(all_pnls) < 20:
        print("  WARNING: Too few trades for Monte Carlo", flush=True)
        all_results['phase7_monte_carlo'] = {'error': 'too few trades'}
    else:
        pnl_arr = np.array(all_pnls)
        n_trades = len(pnl_arr)
        rng = np.random.default_rng(42)

        mc_sharpes = []
        mc_pnls = []
        mc_dds = []

        for _ in range(N_MC):
            resampled = rng.choice(pnl_arr, size=n_trades, replace=True)
            daily_approx = []
            chunk = 5
            for j in range(0, n_trades, chunk):
                daily_approx.append(resampled[j:j+chunk].sum())
            daily_approx = np.array(daily_approx)
            mc_sharpes.append(_sharpe(daily_approx))
            mc_pnls.append(float(resampled.sum()))
            mc_dds.append(_max_dd(daily_approx))

        mc_sharpes = np.array(mc_sharpes)
        mc_pnls = np.array(mc_pnls)
        mc_dds = np.array(mc_dds)

        pct5 = np.percentile(mc_sharpes, 5)
        pct50 = np.percentile(mc_sharpes, 50)
        pct95 = np.percentile(mc_sharpes, 95)

        print(f"  Monte Carlo Sharpe: p5={pct5:.3f}, p50={pct50:.3f}, p95={pct95:.3f}", flush=True)
        print(f"  Monte Carlo PnL: p5=${np.percentile(mc_pnls, 5):.0f}, "
              f"p50=${np.percentile(mc_pnls, 50):.0f}, p95=${np.percentile(mc_pnls, 95):.0f}", flush=True)
        print(f"  Monte Carlo MaxDD: p5=${np.percentile(mc_dds, 5):.0f}, "
              f"p50=${np.percentile(mc_dds, 50):.0f}, p95=${np.percentile(mc_dds, 95):.0f}", flush=True)
        print(f"  P(Sharpe > 0): {(mc_sharpes > 0).mean() * 100:.1f}%", flush=True)
        print(f"  P(Sharpe > 0.5): {(mc_sharpes > 0.5).mean() * 100:.1f}%", flush=True)

        all_results['phase7_monte_carlo'] = {
            'n_paths': N_MC,
            'sharpe_p5': round(pct5, 3), 'sharpe_p50': round(pct50, 3), 'sharpe_p95': round(pct95, 3),
            'pnl_p5': round(float(np.percentile(mc_pnls, 5)), 0),
            'pnl_p50': round(float(np.percentile(mc_pnls, 50)), 0),
            'pnl_p95': round(float(np.percentile(mc_pnls, 95)), 0),
            'dd_p5': round(float(np.percentile(mc_dds, 5)), 0),
            'dd_p50': round(float(np.percentile(mc_dds, 50)), 0),
            'dd_p95': round(float(np.percentile(mc_dds, 95)), 0),
            'prob_sharpe_gt0': round(float((mc_sharpes > 0).mean() * 100), 1),
            'prob_sharpe_gt05': round(float((mc_sharpes > 0.5).mean() * 100), 1),
        }

    # ═══════════════════════════════════════════════════════════════
    # Save
    # ═══════════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    all_results['runtime_sec'] = round(elapsed, 1)
    all_results['best_lot_combo'] = {
        's3_lot': best_lot['s3_lot'], 's4_lot': best_lot['s4_lot'],
    }
    out_file = OUTPUT_DIR / "r135_results.json"
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n{'='*80}", flush=True)
    print(f"  R135 complete in {elapsed/60:.1f} min. Results → {out_file}", flush=True)
    print(f"{'='*80}", flush=True)


if __name__ == '__main__':
    main()

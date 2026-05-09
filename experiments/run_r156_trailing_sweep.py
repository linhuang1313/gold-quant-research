#!/usr/bin/env python3
"""
R156 — Trailing Stop Parameter Sweep (on MaxHold=8 base)
=========================================================
R155 showed MaxHold=8 is optimal. Average profit per trailing exit = $2.12.
Goal: find trailing params that increase avg profit to $3-4 without
destroying Sharpe or win rate.

Current live trailing (L8 regime):
  high:   trail_act=0.06, trail_dist=0.008
  normal: trail_act=0.14, trail_dist=0.025
  low:    trail_act=0.22, trail_dist=0.04

This experiment:
  Phase 1: Baseline (MaxHold=8 + current trailing)
  Phase 2: trail_act sweep (keep dist ratio constant)
  Phase 3: trail_dist sweep (keep act constant)
  Phase 4: 2D grid sweep (act x dist, normal regime only, scale others)
  Phase 5: SL_ATR sweep (on best trailing)
  Phase 6: K-Fold validation on top configs
  Phase 7: Final production comparison (lot=0.05)
"""
import sys, os, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from itertools import product

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("results/r156_trailing_sweep")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

t0 = time.time()


def fmt(v):
    return f"${v:>10,.0f}" if v >= 0 else f"-${abs(v):>9,.0f}"


def sharpe(arr):
    if len(arr) < 10: return 0.0
    s = np.std(arr, ddof=1)
    if s == 0: return 0.0
    return float(np.mean(arr) / s * np.sqrt(252))


def max_dd(arr):
    if len(arr) == 0: return 0.0
    eq = np.cumsum(arr)
    return float((np.maximum.accumulate(eq) - eq).max())


def trades_to_daily(trades):
    if not trades: return pd.Series(dtype=float)
    daily = {}
    for t in trades:
        d = pd.Timestamp(t['exit_time']).date()
        daily[d] = daily.get(d, 0) + t['pnl']
    dates = sorted(daily.keys())
    return pd.Series([daily[d] for d in dates], index=pd.DatetimeIndex(dates))


def make_regime(act_normal, dist_normal):
    """Scale regime params from normal values (high=tighter, low=looser)."""
    return {
        'high':   {'trail_act': round(act_normal * 0.43, 4), 'trail_dist': round(dist_normal * 0.32, 4)},
        'normal': {'trail_act': round(act_normal, 4),        'trail_dist': round(dist_normal, 4)},
        'low':    {'trail_act': round(act_normal * 1.57, 4), 'trail_dist': round(dist_normal * 1.60, 4)},
    }


def run_l8(bundle, spread=0.30, lot=0.01, cap=35,
           keltner_max_hold_m15=8,
           trail_act=None, trail_dist=None, regime=None,
           sl_atr=None, tp_atr=None):
    from backtest.runner import run_variant, LIVE_PARITY_KWARGS
    kw = dict(LIVE_PARITY_KWARGS)
    kw['maxloss_cap'] = cap
    kw['spread_cost'] = spread
    kw['initial_capital'] = 2000
    kw['min_lot_size'] = lot
    kw['max_lot_size'] = lot
    kw['keltner_max_hold_m15'] = keltner_max_hold_m15
    if trail_act is not None:
        kw['trailing_activate_atr'] = trail_act
    if trail_dist is not None:
        kw['trailing_distance_atr'] = trail_dist
    if regime is not None:
        kw['regime_config'] = regime
    if sl_atr is not None:
        kw['sl_atr_mult'] = sl_atr
    if tp_atr is not None:
        kw['tp_atr_mult'] = tp_atr
    result = run_variant(bundle, "L8_MAX", verbose=False, **kw)
    raw = result.get('_trades', [])
    trades = []
    for t in raw:
        trades.append({
            'dir': t.direction, 'entry': t.entry_price, 'exit': t.exit_price,
            'entry_time': t.entry_time, 'exit_time': t.exit_time,
            'pnl': t.pnl, 'reason': t.exit_reason, 'bars_held': t.bars_held,
        })
    return trades


def stats(trades):
    if not trades:
        return {'n': 0, 'sharpe': 0, 'pnl': 0, 'max_dd': 0, 'wr': 0, 'avg_pnl': 0, 'avg_bars': 0}
    pnls = [t['pnl'] for t in trades]
    ds = trades_to_daily(trades)
    n = len(trades)
    trailing = [t for t in trades if 'Trailing' in str(t.get('reason', ''))]
    trail_pnls = [t['pnl'] for t in trailing] if trailing else [0]
    return {
        'n': n, 'sharpe': round(sharpe(ds.values), 2),
        'pnl': round(sum(pnls), 2), 'max_dd': round(max_dd(ds.values), 2),
        'wr': round(sum(1 for p in pnls if p > 0) / n * 100, 1),
        'avg_pnl': round(np.mean(pnls), 2),
        'avg_bars': round(np.mean([t['bars_held'] for t in trades]), 1),
        'trail_n': len(trailing),
        'trail_avg': round(np.mean(trail_pnls), 2),
        'trail_wr': round(sum(1 for p in trail_pnls if p > 0) / max(len(trailing), 1) * 100, 1),
    }


def print_row(label, s):
    print(f"  {label:<45} n={s['n']:>5} Sh={s['sharpe']:>5.2f} PnL={fmt(s['pnl'])} "
          f"DD={fmt(s['max_dd'])} WR={s['wr']:.0f}% Avg={s['avg_pnl']:.2f} "
          f"TrailAvg={s['trail_avg']:.2f} TrailWR={s['trail_wr']:.0f}%", flush=True)


def main():
    results = {}

    print("=" * 100, flush=True)
    print("  R156 — Trailing Stop Sweep (MaxHold=8 base)", flush=True)
    print(f"  Started: {datetime.now()}", flush=True)
    print("=" * 100, flush=True)

    from backtest.runner import DataBundle
    print("\n  Loading DataBundle...", flush=True)
    bundle = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)
    print("  Bundle ready.\n", flush=True)

    # ═══════════════════════════════════════════════════════════════
    # Phase 1: Baseline
    # ═══════════════════════════════════════════════════════════════
    print(f"{'='*100}", flush=True)
    print("  Phase 1: Baseline (MaxHold=8, current trailing)", flush=True)
    print(f"{'='*100}\n", flush=True)

    trades = run_l8(bundle)
    base = stats(trades)
    print_row("Baseline (act=0.14, dist=0.025)", base)
    results['phase1_baseline'] = base

    # ═══════════════════════════════════════════════════════════════
    # Phase 2: trail_act sweep (widen activation = let profit run more)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*100}", flush=True)
    print("  Phase 2: Trail Activate Sweep (normal regime, dist=0.025 fixed)", flush=True)
    print(f"{'='*100}\n", flush=True)

    act_values = [0.06, 0.10, 0.14, 0.20, 0.28, 0.40, 0.60, 0.80, 1.00]
    act_results = []
    for act in act_values:
        regime = make_regime(act, 0.025)
        t_list = run_l8(bundle, regime=regime)
        s = stats(t_list)
        s['act'] = act
        print_row(f"act={act:.2f}", s)
        act_results.append(s)
    results['phase2_act_sweep'] = act_results

    # ═══════════════════════════════════════════════════════════════
    # Phase 3: trail_dist sweep (wider distance = hold trailing longer)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*100}", flush=True)
    print("  Phase 3: Trail Distance Sweep (normal regime, act=0.14 fixed)", flush=True)
    print(f"{'='*100}\n", flush=True)

    dist_values = [0.010, 0.015, 0.020, 0.025, 0.035, 0.050, 0.075, 0.100, 0.150]
    dist_results = []
    for dist in dist_values:
        regime = make_regime(0.14, dist)
        t_list = run_l8(bundle, regime=regime)
        s = stats(t_list)
        s['dist'] = dist
        print_row(f"dist={dist:.3f}", s)
        dist_results.append(s)
    results['phase3_dist_sweep'] = dist_results

    # ═══════════════════════════════════════════════════════════════
    # Phase 4: 2D grid (act x dist, normal values, regime auto-scaled)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*100}", flush=True)
    print("  Phase 4: 2D Grid Sweep (act x dist)", flush=True)
    print(f"{'='*100}\n", flush=True)

    grid_acts = [0.14, 0.20, 0.28, 0.40, 0.60]
    grid_dists = [0.025, 0.035, 0.050, 0.075, 0.100]
    grid_results = []

    for act, dist in product(grid_acts, grid_dists):
        regime = make_regime(act, dist)
        t_list = run_l8(bundle, regime=regime)
        s = stats(t_list)
        s['act'] = act
        s['dist'] = dist
        print_row(f"act={act:.2f} dist={dist:.3f}", s)
        grid_results.append(s)

    results['phase4_grid'] = grid_results

    # Find best by Sharpe
    best_grid = max(grid_results, key=lambda x: x['sharpe'])
    print(f"\n  >>> Best grid: act={best_grid['act']:.2f}, dist={best_grid['dist']:.3f} "
          f"— Sharpe={best_grid['sharpe']:.2f}, TrailAvg={best_grid['trail_avg']:.2f}", flush=True)

    # Also find best by trail_avg (highest profit per trailing trade)
    profitable_grid = [g for g in grid_results if g['sharpe'] >= base['sharpe'] * 0.95]
    if profitable_grid:
        best_profit = max(profitable_grid, key=lambda x: x['trail_avg'])
        print(f"  >>> Best profit (Sharpe>=95% base): act={best_profit['act']:.2f}, dist={best_profit['dist']:.3f} "
              f"— Sharpe={best_profit['sharpe']:.2f}, TrailAvg={best_profit['trail_avg']:.2f}", flush=True)
    else:
        best_profit = best_grid

    # ═══════════════════════════════════════════════════════════════
    # Phase 5: SL_ATR sweep on best trailing config
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*100}", flush=True)
    print(f"  Phase 5: SL ATR Sweep (on best trailing act={best_grid['act']:.2f}, dist={best_grid['dist']:.3f})", flush=True)
    print(f"{'='*100}\n", flush=True)

    sl_values = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
    sl_results = []
    best_regime = make_regime(best_grid['act'], best_grid['dist'])

    for sl in sl_values:
        t_list = run_l8(bundle, regime=best_regime, sl_atr=sl)
        s = stats(t_list)
        s['sl_atr'] = sl
        label = f"SL={sl:.1f}" + (" (current)" if sl == 3.5 else "")
        print_row(label, s)
        sl_results.append(s)

    results['phase5_sl_sweep'] = sl_results

    # ═══════════════════════════════════════════════════════════════
    # Phase 6: K-Fold validation
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*100}", flush=True)
    print("  Phase 6: K-Fold Validation", flush=True)
    print(f"{'='*100}\n", flush=True)

    FOLDS = [
        ("Fold1", "2015-01-01", "2017-03-01"),
        ("Fold2", "2017-03-01", "2019-06-01"),
        ("Fold3", "2019-06-01", "2021-09-01"),
        ("Fold4", "2021-09-01", "2023-12-01"),
        ("Fold5", "2023-12-01", "2027-01-01"),
    ]

    configs_to_validate = [
        ("Baseline (act=0.14,dist=0.025)", None, 3.5),
        (f"BestSharpe (act={best_grid['act']},dist={best_grid['dist']})",
         make_regime(best_grid['act'], best_grid['dist']), 3.5),
    ]
    if best_profit != best_grid:
        configs_to_validate.append(
            (f"BestProfit (act={best_profit['act']},dist={best_profit['dist']})",
             make_regime(best_profit['act'], best_profit['dist']), 3.5),
        )

    best_sl = max(sl_results, key=lambda x: x['sharpe'])
    if best_sl['sl_atr'] != 3.5:
        configs_to_validate.append(
            (f"BestSharpe+SL={best_sl['sl_atr']}", best_regime, best_sl['sl_atr']),
        )

    kfold_results = {}
    for label, regime, sl in configs_to_validate:
        fold_sharpes = []
        fold_trail_avgs = []
        for fname, start, end in FOLDS:
            try:
                fb = bundle.slice(start, end)
            except Exception:
                fold_sharpes.append(0.0)
                fold_trail_avgs.append(0.0)
                continue
            t_f = run_l8(fb, regime=regime, sl_atr=sl)
            s_f = stats(t_f)
            fold_sharpes.append(s_f['sharpe'])
            fold_trail_avgs.append(s_f['trail_avg'])

        pos = sum(1 for s in fold_sharpes if s > 0)
        mean_sh = float(np.mean(fold_sharpes))
        mean_ta = float(np.mean(fold_trail_avgs))
        print(f"  {label:<55}: Sharpe=[{', '.join(f'{s:.2f}' for s in fold_sharpes)}] "
              f"pos={pos}/5 mean={mean_sh:.2f} TrailAvg={mean_ta:.2f}", flush=True)

        kfold_results[label] = {
            'folds_sharpe': [round(s, 2) for s in fold_sharpes],
            'folds_trail_avg': [round(s, 2) for s in fold_trail_avgs],
            'positive_folds': pos,
            'mean_sharpe': round(mean_sh, 2),
            'mean_trail_avg': round(mean_ta, 2),
        }

    results['phase6_kfold'] = kfold_results

    # ═══════════════════════════════════════════════════════════════
    # Phase 7: Final production comparison (lot=0.05)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*100}", flush=True)
    print("  Phase 7: Production Comparison (lot=0.05)", flush=True)
    print(f"{'='*100}\n", flush=True)

    final_configs = [
        ("Current (act=0.14, dist=0.025, MH=8)", None, 3.5),
        (f"Optimized (act={best_grid['act']}, dist={best_grid['dist']}, MH=8)",
         make_regime(best_grid['act'], best_grid['dist']), 3.5),
    ]
    if best_profit != best_grid:
        final_configs.append(
            (f"MaxProfit (act={best_profit['act']}, dist={best_profit['dist']}, MH=8)",
             make_regime(best_profit['act'], best_profit['dist']), 3.5),
        )

    final_data = []
    for label, regime, sl in final_configs:
        t_list = run_l8(bundle, lot=0.05, regime=regime, sl_atr=sl)
        s = stats(t_list)
        s['label'] = label
        print(f"  {label}:", flush=True)
        print(f"    n={s['n']}, Sharpe={s['sharpe']:.2f}, PnL={fmt(s['pnl'])}, MaxDD={fmt(s['max_dd'])}", flush=True)
        print(f"    WR={s['wr']:.1f}%, AvgPnL={s['avg_pnl']:.2f}, TrailAvg={s['trail_avg']:.2f}", flush=True)
        final_data.append(s)

    results['phase7_final'] = final_data

    # ═══════════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    print(f"\n{'='*100}", flush=True)
    print(f"  R156 COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)
    print(f"{'='*100}", flush=True)

    results['elapsed_s'] = round(elapsed, 1)
    with open(OUTPUT_DIR / "r156_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved: {OUTPUT_DIR}/r156_results.json", flush=True)


if __name__ == "__main__":
    main()

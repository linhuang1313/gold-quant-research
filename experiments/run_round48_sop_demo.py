#!/usr/bin/env python3
"""
Round 48 — SOP 三阶段演示: Chandelier 出场参数优化
================================================================
验证两层快筛 + 完整引擎验证 + 压力确认的标准 SOP 流程。

案例: S4 Chandelier Exit Flip 出场参数网格搜索
  - 144 组合 (SL 4 x TP 4 x MaxHold 3 x Trail 3)
  - Phase 1: 快筛淘汰 Sharpe < 0 (~7 min)
  - Phase 2: 存活候选中 Sharpe > 2 做 K-Fold 验证 (~30-60 min)
  - Phase 3: Top 候选压力确认 (~15 min)

Usage:
  python -m experiments.run_round48_sop_demo
"""
import sys
import os
import time
import json
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
os.chdir(str(ROOT))
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from backtest.runner import DataBundle, run_variant, LIVE_PARITY_KWARGS
from backtest.fast_screen import (
    fast_backtest_signals, screen_grid, kfold_screen,
    trades_to_stats, daily_pnl_correlation, combine_daily_pnl, stats_from_daily,
)
from indicators import calc_chandelier

# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

EXPERIMENT_NAME = "round48_sop_demo"
OUT_DIR = ROOT / "results" / EXPERIMENT_NAME
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Chandelier signal params (fixed for this experiment — optimizing exit only)
CH_SIG_PARAMS = {'period': 10, 'mult': 3.0}

# Parameter grid for exit optimization
SL_GRID = [2.0, 3.0, 3.5, 4.5]
TP_GRID = [6.0, 8.0, 10.0, 12.0]
MH_GRID = [12, 20, 30]
TRAIL_GRID = [
    (0.28, 0.06),
    (0.20, 0.04),
    (0.40, 0.10),
]

# L8 baseline for correlation calculation
L8_KWARGS = {
    **LIVE_PARITY_KWARGS,
    'keltner_adx_threshold': 14,
    'regime_config': {
        'low':    {'trail_act': 0.22, 'trail_dist': 0.04},
        'normal': {'trail_act': 0.14, 'trail_dist': 0.025},
        'high':   {'trail_act': 0.06, 'trail_dist': 0.008},
    },
    'keltner_max_hold_m15': 20,
    'time_decay_tp': False,
    'min_entry_gap_hours': 1.0,
}


# ═══════════════════════════════════════════════════════════════
# Signal function
# ═══════════════════════════════════════════════════════════════

def chandelier_signals(df, period=10, mult=3.0, **kw):
    """Chandelier Exit Flip signal generator."""
    ch = calc_chandelier(df, period, mult)
    atr = (df['High'] - df['Low']).rolling(14).mean()
    close = df['Close']

    above_long = close > ch['Chand_long']
    below_short = close < ch['Chand_short']

    flip_bull = above_long & (~above_long.shift(1).fillna(False))
    flip_bear = below_short & (~below_short.shift(1).fillna(False))

    sig = pd.Series(0, index=df.index)
    sig[flip_bull] = 1
    sig[flip_bear] = -1
    return sig, atr


# ═══════════════════════════════════════════════════════════════
# Phase 1: Fast Screening
# ═══════════════════════════════════════════════════════════════

def build_param_grid() -> List[Dict]:
    """Build 144-combo exit parameter grid."""
    grid = []
    for sl in SL_GRID:
        for tp in TP_GRID:
            for mh in MH_GRID:
                for trail_act, trail_dist in TRAIL_GRID:
                    grid.append({
                        'label': f"SL{sl}_TP{tp}_MH{mh}_T{trail_act:.2f}",
                        'sig_params': CH_SIG_PARAMS,
                        'bt_params': {
                            'sl_mult': sl,
                            'tp_mult': tp,
                            'max_hold': mh,
                            'trail_act': trail_act,
                            'trail_dist': trail_dist,
                        },
                    })
    return grid


def phase_1(h1_df: pd.DataFrame) -> List[Dict]:
    """Fast screen: eliminate Sharpe < 0."""
    print("\n" + "=" * 70)
    print("  PHASE 1: Fast Screening (eliminate mode)")
    print("  Grid: 144 combos (SL 4 x TP 4 x MH 3 x Trail 3)")
    print("=" * 70)

    param_grid = build_param_grid()
    t0 = time.time()

    survivors = screen_grid(
        h1_df, chandelier_signals, param_grid,
        min_sharpe=0.0,
        rank_by='sharpe',
        verbose=True,
    )

    elapsed = time.time() - t0

    # Save results
    save_data = {
        'n_total': len(param_grid),
        'n_survivors': len(survivors),
        'elimination_rate': f"{(1 - len(survivors)/len(param_grid))*100:.1f}%",
        'elapsed_s': round(elapsed, 1),
        'top_10': [{
            'label': r['label'],
            'rank': r['rank'],
            'sharpe': r['stats']['sharpe'],
            'pnl': r['stats']['total_pnl'],
            'n_trades': r['stats']['n'],
            'max_dd': r['stats']['max_dd'],
            'win_rate': r['stats']['win_rate'],
            'bt_params': r['bt_params'],
        } for r in survivors[:10]],
        'all_survivors': [{
            'label': r['label'],
            'sharpe': r['stats']['sharpe'],
            'pnl': r['stats']['total_pnl'],
            'n_trades': r['stats']['n'],
            'bt_params': r['bt_params'],
        } for r in survivors],
    }

    with open(OUT_DIR / "phase1_screen.json", 'w') as f:
        json.dump(save_data, f, indent=2, default=str)

    print(f"\n  Phase 1 complete: {elapsed:.1f}s")
    print(f"  {len(param_grid)} -> {len(survivors)} survivors "
          f"({save_data['elimination_rate']} eliminated)")
    print(f"  Saved: {OUT_DIR / 'phase1_screen.json'}")

    return survivors


# ═══════════════════════════════════════════════════════════════
# Phase 2: Full Validation (K-Fold on best candidates)
# ═══════════════════════════════════════════════════════════════

def phase_2(h1_df: pd.DataFrame, survivors: List[Dict], l8_daily: Dict) -> List[Dict]:
    """Full validation: K-Fold on candidates with screen Sharpe > 2."""
    # Filter to Sharpe > 2 for full validation
    candidates = [r for r in survivors if r['stats']['sharpe'] > 2.0]
    print("\n" + "=" * 70)
    print(f"  PHASE 2: Full Validation")
    print(f"  {len(survivors)} survivors -> {len(candidates)} with screen Sharpe > 2")
    print("=" * 70)

    if not candidates:
        print("  No candidates with Sharpe > 2. Using top 10 instead.")
        candidates = survivors[:10]

    t0 = time.time()
    validated = []

    for idx, r in enumerate(candidates):
        bt_params = r['bt_params']
        label = r['label']

        # Full-sample run with fast_backtest_signals (H1 precision is fine for CH)
        signals, atr = chandelier_signals(h1_df, **CH_SIG_PARAMS)
        trades = fast_backtest_signals(h1_df, signals, atr, **bt_params)
        stats = trades_to_stats(trades, label)

        # Correlation with L8
        ch_daily = stats.get('daily_pnl', {})
        corr = daily_pnl_correlation(ch_daily, l8_daily)

        result = {
            'label': label,
            'screen_sharpe': r['stats']['sharpe'],
            'full_sharpe': stats['sharpe'],
            'full_pnl': stats['total_pnl'],
            'n_trades': stats['n'],
            'max_dd': stats['max_dd'],
            'win_rate': stats['win_rate'],
            'l8_correlation': corr,
            'bt_params': bt_params,
        }

        # K-Fold for Sharpe > 5 candidates
        if stats['sharpe'] > 5.0:
            kf = kfold_screen(h1_df, chandelier_signals, CH_SIG_PARAMS, bt_params, label)
            result['kfold_sharpes'] = kf['sharpes']
            result['kfold_mean'] = kf['mean_sharpe']
            result['kfold_min'] = kf['min_sharpe']
            result['kfold_pass'] = kf['pass']
            print(f"    [{idx+1}/{len(candidates)}] {label}: "
                  f"Sharpe={stats['sharpe']:.2f}, K-Fold={kf['pass']}, "
                  f"corr={corr:.3f}")
        else:
            print(f"    [{idx+1}/{len(candidates)}] {label}: "
                  f"Sharpe={stats['sharpe']:.2f}, corr={corr:.3f}")

        validated.append(result)

    # Sort by full Sharpe
    validated.sort(key=lambda x: x.get('full_sharpe', 0), reverse=True)

    elapsed = time.time() - t0

    # Save
    with open(OUT_DIR / "phase2_validation.json", 'w') as f:
        json.dump({
            'n_candidates': len(candidates),
            'n_kfold_tested': sum(1 for v in validated if 'kfold_pass' in v),
            'elapsed_s': round(elapsed, 1),
            'results': validated,
        }, f, indent=2, default=str)

    # Summary
    kfold_passed = [v for v in validated if v.get('kfold_pass', '0/6').startswith(('5/', '6/'))]
    print(f"\n  Phase 2 complete: {elapsed:.1f}s")
    print(f"  {len(candidates)} validated, "
          f"{sum(1 for v in validated if 'kfold_pass' in v)} K-Fold tested, "
          f"{len(kfold_passed)} passed 5/6+")
    if validated:
        best = validated[0]
        print(f"  Best: {best['label']} Sharpe={best['full_sharpe']:.2f}, "
              f"PnL=${best['full_pnl']:.0f}, corr_L8={best['l8_correlation']:.3f}")
    print(f"  Saved: {OUT_DIR / 'phase2_validation.json'}")

    return validated


# ═══════════════════════════════════════════════════════════════
# Phase 3: Stress Confirmation
# ═══════════════════════════════════════════════════════════════

def phase_3(h1_df: pd.DataFrame, validated: List[Dict]) -> List[Dict]:
    """Stress test top candidates."""
    # Take top 2 by Sharpe that passed K-Fold, or just top 2
    kf_passed = [v for v in validated if v.get('kfold_pass', '0/6').startswith(('5/', '6/'))]
    if len(kf_passed) >= 2:
        best = kf_passed[:2]
    else:
        best = validated[:2]

    print("\n" + "=" * 70)
    print(f"  PHASE 3: Stress Confirmation ({len(best)} candidates)")
    print("=" * 70)

    t0 = time.time()
    stress_results = []

    for candidate in best:
        bt_params = candidate['bt_params']
        label = candidate['label']
        print(f"\n  --- {label} (Sharpe={candidate['full_sharpe']:.2f}) ---")

        # Spread sensitivity
        print("  Spread test:", end='', flush=True)
        spread_results = []
        for spread in [0.30, 0.50, 0.75, 1.00]:
            signals, atr = chandelier_signals(h1_df, **CH_SIG_PARAMS)
            trades = fast_backtest_signals(
                h1_df, signals, atr, spread_cost=spread, **bt_params)
            stats = trades_to_stats(trades, f"{label}_sp{spread}")
            spread_results.append({
                'spread': spread,
                'sharpe': stats['sharpe'],
                'pnl': stats['total_pnl'],
                'n': stats['n'],
            })
            print(f" ${spread}={stats['sharpe']:.2f}", end='', flush=True)
        print()

        # Crisis periods
        print("  Crisis test:", end='', flush=True)
        crises = [
            ("COVID_2020", "2020-02-01", "2020-06-01"),
            ("RateHike_2022", "2022-03-01", "2022-12-31"),
            ("BankCrisis_2023", "2023-03-01", "2023-06-01"),
            ("Geopolitical_2024", "2024-04-01", "2024-08-01"),
        ]
        crisis_results = []
        for name, start, end in crises:
            fold_df = h1_df[start:end]
            if len(fold_df) < 100:
                continue
            signals, atr = chandelier_signals(fold_df, **CH_SIG_PARAMS)
            trades = fast_backtest_signals(fold_df, signals, atr, **bt_params)
            stats = trades_to_stats(trades, f"{label}_{name}")
            crisis_results.append({
                'period': name,
                'sharpe': stats['sharpe'],
                'pnl': stats['total_pnl'],
                'n': stats['n'],
                'win_rate': stats['win_rate'],
            })
            status = "OK" if stats['sharpe'] > 0 else "WARN"
            print(f" {name}={stats['sharpe']:.1f}({status})", end='', flush=True)
        print()

        # BUY/SELL direction balance
        signals, atr = chandelier_signals(h1_df, **CH_SIG_PARAMS)
        all_trades = fast_backtest_signals(h1_df, signals, atr, **bt_params)
        buy_trades = [t for t in all_trades if t.direction == 'BUY']
        sell_trades = [t for t in all_trades if t.direction == 'SELL']
        buy_stats = trades_to_stats(buy_trades, "BUY")
        sell_stats = trades_to_stats(sell_trades, "SELL")
        print(f"  Direction: BUY Sharpe={buy_stats['sharpe']:.2f} (N={buy_stats['n']}), "
              f"SELL Sharpe={sell_stats['sharpe']:.2f} (N={sell_stats['n']})")

        stress_results.append({
            'label': label,
            'full_sharpe': candidate['full_sharpe'],
            'bt_params': bt_params,
            'spread_sensitivity': spread_results,
            'crisis_performance': crisis_results,
            'direction_balance': {
                'buy_sharpe': buy_stats['sharpe'],
                'buy_n': buy_stats['n'],
                'sell_sharpe': sell_stats['sharpe'],
                'sell_n': sell_stats['n'],
            },
            'spread_breakeven': next(
                (r['spread'] for r in spread_results if r['sharpe'] <= 0), '>$1.00'),
        })

    elapsed = time.time() - t0

    with open(OUT_DIR / "phase3_stress.json", 'w') as f:
        json.dump(stress_results, f, indent=2, default=str)

    print(f"\n  Phase 3 complete: {elapsed:.1f}s")
    print(f"  Saved: {OUT_DIR / 'phase3_stress.json'}")

    return stress_results


# ═══════════════════════════════════════════════════════════════
# Final Summary
# ═══════════════════════════════════════════════════════════════

def print_summary(survivors, validated, stress_results):
    """Print final conclusions."""
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)

    # Current default
    print("\n  Current default: SL=3.0, TP=8.0, MH=20, Trail=0.28/0.06")
    default_match = [v for v in validated
                     if abs(v['bt_params']['sl_mult'] - 3.0) < 0.01
                     and abs(v['bt_params']['tp_mult'] - 8.0) < 0.01
                     and v['bt_params']['max_hold'] == 20
                     and abs(v['bt_params']['trail_act'] - 0.28) < 0.01]
    if default_match:
        d = default_match[0]
        print(f"    Sharpe={d['full_sharpe']:.2f}, PnL=${d['full_pnl']:.0f}")

    # Best found
    if validated:
        best = validated[0]
        print(f"\n  Best found: {best['label']}")
        print(f"    Sharpe={best['full_sharpe']:.2f}, PnL=${best['full_pnl']:.0f}, "
              f"MaxDD=${best['max_dd']:.0f}")
        print(f"    L8 correlation: {best['l8_correlation']:.3f}")
        if 'kfold_pass' in best:
            print(f"    K-Fold: {best['kfold_pass']}, "
                  f"mean={best['kfold_mean']:.2f}, min={best['kfold_min']:.2f}")
        print(f"    Params: {best['bt_params']}")

    # Stress conclusions
    if stress_results:
        print(f"\n  Stress test results:")
        for s in stress_results:
            crises_passed = sum(1 for c in s['crisis_performance'] if c['sharpe'] > 0)
            total_crises = len(s['crisis_performance'])
            print(f"    {s['label']}: spread_BE={s['spread_breakeven']}, "
                  f"crisis={crises_passed}/{total_crises}, "
                  f"BUY={s['direction_balance']['buy_sharpe']:.1f}/"
                  f"SELL={s['direction_balance']['sell_sharpe']:.1f}")

    # SOP flow stats
    print(f"\n  SOP efficiency:")
    print(f"    Phase 1: 144 combos -> {len(survivors)} survivors (fast screen)")
    n_kf = sum(1 for v in validated if 'kfold_pass' in v)
    print(f"    Phase 2: {len(validated)} validated, {n_kf} K-Fold tested")
    print(f"    Phase 3: {len(stress_results)} stress confirmed")
    print("=" * 70)


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    t_total = time.time()
    print("=" * 70)
    print("  Round 48 — SOP 三阶段演示: Chandelier 出场参数优化")
    print(f"  Output: {OUT_DIR}")
    print("=" * 70)

    # Load data (full DataBundle with indicators for both M15 and H1)
    print("\n  Loading data...")
    data = DataBundle.load_default()
    h1_df = data.h1_df.copy()
    print(f"  H1: {len(h1_df)} bars ({h1_df.index[0]} -> {h1_df.index[-1]})")

    # Get L8 daily PnL for correlation
    print("\n  Running L8 baseline for correlation reference...")
    l8_stats = run_variant(data, "L8_BASE", **L8_KWARGS)
    l8_trades = l8_stats['_trades']
    l8_daily: Dict[str, float] = {}
    for t in l8_trades:
        d = str(t.exit_time.date()) if hasattr(t.exit_time, 'date') else str(t.exit_time)[:10]
        l8_daily[d] = l8_daily.get(d, 0) + t.pnl
    print(f"  L8 baseline: Sharpe={l8_stats['sharpe']:.2f}, "
          f"PnL=${l8_stats['total_pnl']:.0f}, N={l8_stats['n']}")

    # Phase 1: Fast screening
    survivors = phase_1(h1_df)
    if not survivors:
        print("\n  ABORT: No survivors after Phase 1.")
        return

    # Phase 2: Full validation
    validated = phase_2(h1_df, survivors, l8_daily)
    if not validated:
        print("\n  ABORT: No candidates validated.")
        return

    # Phase 3: Stress confirmation
    stress_results = phase_3(h1_df, validated)

    # Final summary
    print_summary(survivors, validated, stress_results)

    total_elapsed = time.time() - t_total
    print(f"\n  Total time: {total_elapsed/60:.1f} min ({total_elapsed:.0f}s)")


if __name__ == '__main__':
    main()

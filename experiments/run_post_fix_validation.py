#!/usr/bin/env python3
"""
Post-Fix Validation — New Engine Baseline + Pending Re-validations
===================================================================
Runs on the look-ahead-fixed engine with LIVE_PARITY_KWARGS.

Phase A: LIVE_PARITY baseline at $0/$0.30/$0.50
Phase B: MaxHold sweep (12/16/20/24/32 M15 bars) at $0.30 + K-Fold top-2
Phase C: Trail Momentum 1.5x at $0.30/$0.50 + K-Fold

All results use LIVE_PARITY_KWARGS as the base config (matching live system).
"""
import sys, os, time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import pandas as pd
from backtest import DataBundle, run_variant
from backtest.runner import LIVE_PARITY_KWARGS

OUTPUT_FILE = "post_fix_validation_output.txt"

FOLDS = [
    ("Fold1", "2015-01-01", "2017-01-01"),
    ("Fold2", "2017-01-01", "2019-01-01"),
    ("Fold3", "2019-01-01", "2021-01-01"),
    ("Fold4", "2021-01-01", "2023-01-01"),
    ("Fold5", "2023-01-01", "2025-01-01"),
    ("Fold6", "2025-01-01", "2026-04-01"),
]


class TeeOutput:
    def __init__(self, fp):
        self.file = open(fp, 'w', encoding='utf-8')
        self.stdout = sys.stdout
    def write(self, d):
        self.stdout.write(d)
        self.file.write(d)
        self.file.flush()
    def flush(self):
        self.stdout.flush()
        self.file.flush()
    def close(self):
        self.file.close()


tee = TeeOutput(OUTPUT_FILE)
sys.stdout = tee


def fmt(v):
    return f"${v:>10,.0f}" if v >= 0 else f"-${abs(v):>9,.0f}"


print("=" * 80)
print("POST-FIX VALIDATION — LIVE_PARITY Baseline + MaxHold + Trail Momentum")
print(f"Started: {datetime.now()}")
print("=" * 80)

t_total = time.time()

print("\nLoading data...")
data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)

# ══════════════════════════════════════════════════════════════════════════════
# PHASE A: LIVE_PARITY baseline at $0/$0.30/$0.50
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("PHASE A: LIVE_PARITY BASELINE — $0 / $0.30 / $0.50")
print("  Uses T7 OnlyHigh regime (high: T0.25/D0.05)")
print("=" * 80)

print(f"\n{'Config':<32s}  {'N':>5s}  {'Sharpe':>6s}  {'PnL':>11s}  {'WR%':>5s}  "
      f"{'$/t':>7s}  {'MaxDD':>11s}")
print("-" * 90)

for spread in [0.0, 0.30, 0.50]:
    s = run_variant(data, f"LiveParity_sp{spread}", verbose=True,
                    **LIVE_PARITY_KWARGS, spread_cost=spread)
    n = s['n']
    avg = s['total_pnl'] / n if n > 0 else 0
    print(f"  LiveParity sp${spread:.2f}          {n:>5d}  {s['sharpe']:>6.2f}  "
          f"{fmt(s['total_pnl'])}  {s['win_rate']:>5.1f}%  "
          f"${avg:>6.2f}  {fmt(s['max_dd'])}")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE B: MaxHold sweep
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("PHASE B: MAX HOLD SWEEP (LIVE_PARITY + $0.30 spread)")
print("  Current live: keltner_max_hold_m15=12 (3 hours)")
print("  Testing: 8, 12, 16, 20, 24, 32, 48 M15 bars")
print("=" * 80)

HOLD_VALUES = [8, 12, 16, 20, 24, 32, 48]

print(f"\n{'MaxHold':>8s}  {'N':>5s}  {'Sharpe':>6s}  {'PnL':>11s}  {'WR%':>5s}  "
      f"{'$/t':>7s}  {'MaxDD':>11s}")
print("-" * 70)

hold_results = {}
for mh in HOLD_VALUES:
    kwargs = {**LIVE_PARITY_KWARGS, "keltner_max_hold_m15": mh, "spread_cost": 0.30}
    s = run_variant(data, f"Hold{mh}_sp030", verbose=False, **kwargs)
    n = s['n']
    avg = s['total_pnl'] / n if n > 0 else 0
    marker = " <-- current" if mh == 12 else ""
    print(f"  {mh:>6d}  {n:>5d}  {s['sharpe']:>6.2f}  "
          f"{fmt(s['total_pnl'])}  {s['win_rate']:>5.1f}%  "
          f"${avg:>6.2f}  {fmt(s['max_dd'])}{marker}")
    hold_results[mh] = s

# K-Fold top-2 (excluding current=12)
sorted_holds = sorted(hold_results.items(), key=lambda x: -x[1]['sharpe'])
top2 = [h for h, _ in sorted_holds if h != 12][:2]
current_sharpe = hold_results[12]['sharpe']

print(f"\n  Current (12): Sharpe={current_sharpe:.2f}")
print(f"  Top-2 candidates: {top2}")
print(f"\n--- K-Fold validation at $0.30 ---")

for mh in top2:
    print(f"\n  MaxHold={mh}:")
    wins = 0
    fold_sharpes = []
    base_sharpes = []
    for fold_name, start, end in FOLDS:
        fold_data = data.slice(start, end)
        if len(fold_data.m15_df) < 1000 or len(fold_data.h1_df) < 200:
            continue
        sb = run_variant(fold_data, f"B_Hold12_{fold_name}", verbose=False,
                         **LIVE_PARITY_KWARGS, spread_cost=0.30)
        st = run_variant(fold_data, f"B_Hold{mh}_{fold_name}", verbose=False,
                         **{**LIVE_PARITY_KWARGS, "keltner_max_hold_m15": mh}, spread_cost=0.30)
        delta = st['sharpe'] - sb['sharpe']
        won = delta > 0
        if won:
            wins += 1
        fold_sharpes.append(st['sharpe'])
        base_sharpes.append(sb['sharpe'])
        print(f"    {fold_name}: Base={sb['sharpe']:>6.2f}  Hold{mh}={st['sharpe']:>6.2f}  "
              f"delta={delta:>+.2f} {'V' if won else 'X'}")
    print(f"    Result: wins {wins}/6 {'PASS' if wins >= 5 else 'FAIL'}")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE C: Trail Momentum 1.5x
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("PHASE C: TRAIL MOMENTUM 1.5x VALIDATION")
print("  Change: trail_dist *= 1.5 when price moves in favor during same bar")
print("  Implementation: regime high trail_dist 0.05→0.075, normal 0.15→0.225, low 0.25→0.375")
print("  This tests widening trail_dist by 1.5x (more room = fewer premature exits)")
print("=" * 80)

TRAIL_MOM_REGIME = {
    'low':    {'trail_act': 0.7,  'trail_dist': 0.375},
    'normal': {'trail_act': 0.5,  'trail_dist': 0.225},
    'high':   {'trail_act': 0.25, 'trail_dist': 0.075},
}

print(f"\n{'Config':<32s}  {'N':>5s}  {'Sharpe':>6s}  {'PnL':>11s}  {'WR%':>5s}  "
      f"{'$/t':>7s}  {'MaxDD':>11s}  {'d Sharpe':>8s}")
print("-" * 100)

for spread in [0.30, 0.50]:
    print(f"\n  --- Spread = ${spread:.2f} ---")
    sb = run_variant(data, f"C_Base_sp{spread}", verbose=False,
                     **LIVE_PARITY_KWARGS, spread_cost=spread)
    st = run_variant(data, f"C_TM15_sp{spread}", verbose=False,
                     **{**LIVE_PARITY_KWARGS, "regime_config": TRAIL_MOM_REGIME},
                     spread_cost=spread)
    delta = st['sharpe'] - sb['sharpe']
    nb = sb['n']
    nt = st['n']
    avgb = sb['total_pnl'] / nb if nb > 0 else 0
    avgt = st['total_pnl'] / nt if nt > 0 else 0
    print(f"  {'Baseline':<30s}  {nb:>5d}  {sb['sharpe']:>6.2f}  "
          f"{fmt(sb['total_pnl'])}  {sb['win_rate']:>5.1f}%  "
          f"${avgb:>6.2f}  {fmt(sb['max_dd'])}       ---")
    print(f"  {'Trail Mom 1.5x':<30s}  {nt:>5d}  {st['sharpe']:>6.2f}  "
          f"{fmt(st['total_pnl'])}  {st['win_rate']:>5.1f}%  "
          f"${avgt:>6.2f}  {fmt(st['max_dd'])}  {delta:>+6.2f}")

# K-Fold at $0.30
print(f"\n--- K-Fold at $0.30 ---")
wins_tm = 0
for fold_name, start, end in FOLDS:
    fold_data = data.slice(start, end)
    if len(fold_data.m15_df) < 1000 or len(fold_data.h1_df) < 200:
        continue
    sb = run_variant(fold_data, f"C_Base_{fold_name}", verbose=False,
                     **LIVE_PARITY_KWARGS, spread_cost=0.30)
    st = run_variant(fold_data, f"C_TM15_{fold_name}", verbose=False,
                     **{**LIVE_PARITY_KWARGS, "regime_config": TRAIL_MOM_REGIME},
                     spread_cost=0.30)
    delta = st['sharpe'] - sb['sharpe']
    won = delta > 0
    if won:
        wins_tm += 1
    print(f"  {fold_name}: Base={sb['sharpe']:>6.2f}  TM1.5x={st['sharpe']:>6.2f}  "
          f"delta={delta:>+.2f} {'V' if won else 'X'}")

print(f"\n  K-Fold result: Trail Mom 1.5x wins {wins_tm}/6 {'PASS' if wins_tm >= 5 else 'FAIL'}")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

elapsed = time.time() - t_total
print(f"\n  Total runtime: {elapsed/60:.1f} minutes")
print(f"  Completed: {datetime.now()}")

sys.stdout = tee.stdout
tee.close()
print(f"\nResults saved to {OUTPUT_FILE}")

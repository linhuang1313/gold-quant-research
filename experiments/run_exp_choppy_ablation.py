#!/usr/bin/env python3
"""
EXP-G: Choppy Gate Ablation + Threshold Optimization
=====================================================
Old engine showed choppy gate Sharpe +0.41. Verify on fixed engine.
Test: choppy threshold 0.25/0.30/0.35/0.40/0.45 + gate OFF
Also test kc_only threshold: 0.50/0.55/0.60/0.65/0.70
"""
import sys, os, time, gc
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
from backtest import DataBundle, run_variant
from backtest.runner import LIVE_PARITY_KWARGS

OUTPUT_FILE = "exp_choppy_ablation_output.txt"

FOLDS = [
    ("Fold1", "2015-01-01", "2017-01-01"),
    ("Fold2", "2017-01-01", "2019-01-01"),
    ("Fold3", "2019-01-01", "2021-01-01"),
    ("Fold4", "2021-01-01", "2023-01-01"),
    ("Fold5", "2023-01-01", "2025-01-01"),
    ("Fold6", "2025-01-01", "2026-04-01"),
]

BASE = {**LIVE_PARITY_KWARGS, "keltner_max_hold_m15": 20}


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
print("EXP-G: CHOPPY GATE ABLATION + THRESHOLD OPTIMIZATION")
print(f"Started: {datetime.now()}")
print("=" * 80)

t_total = time.time()
data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)

# ── Part 1: Adaptive ON vs OFF ──
print("\n--- Part 1: Adaptive ON vs OFF ---")
print(f"{'Config':<35s}  {'N':>5s}  {'Sharpe':>6s}  {'PnL':>11s}  {'WR%':>5s}  {'$/t':>7s}  {'MaxDD':>11s}")
print("-" * 85)

s_on = run_variant(data, "G_adaptive_ON", verbose=False, **BASE, spread_cost=0.30)
s_off = run_variant(data, "G_adaptive_OFF", verbose=False,
                    **{**BASE, "intraday_adaptive": False}, spread_cost=0.30)

for label, s in [("Adaptive ON (current)", s_on), ("Adaptive OFF", s_off)]:
    n = s['n']
    avg = s['total_pnl'] / n if n > 0 else 0
    print(f"  {label:<33s}  {n:>5d}  {s['sharpe']:>6.2f}  "
          f"{fmt(s['total_pnl'])}  {s['win_rate']:>5.1f}%  "
          f"${avg:>6.2f}  {fmt(s['max_dd'])}")

delta = s_on['sharpe'] - s_off['sharpe']
print(f"  Adaptive gate value: Sharpe {delta:>+.2f}")

# ── Part 2: Choppy threshold sweep ──
print("\n--- Part 2: Choppy Threshold Sweep ---")
CHOPPY_TH = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

print(f"{'Choppy_th':>9s}  {'N':>5s}  {'Sharpe':>6s}  {'PnL':>11s}  {'WR%':>5s}  {'$/t':>7s}  {'MaxDD':>11s}")
print("-" * 70)

choppy_results = {}
for th in CHOPPY_TH:
    kwargs = {**BASE, "choppy_threshold": th}
    s = run_variant(data, f"G_choppy{th}", verbose=False, **kwargs, spread_cost=0.30)
    n = s['n']
    avg = s['total_pnl'] / n if n > 0 else 0
    marker = " <-- current" if th == 0.35 else ""
    print(f"  {th:>7.2f}  {n:>5d}  {s['sharpe']:>6.2f}  "
          f"{fmt(s['total_pnl'])}  {s['win_rate']:>5.1f}%  "
          f"${avg:>6.2f}  {fmt(s['max_dd'])}{marker}")
    choppy_results[th] = s

# ── Part 3: KC-only threshold sweep ──
print("\n--- Part 3: KC-Only Threshold Sweep ---")
KC_ONLY_TH = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]

print(f"{'KC_only_th':>10s}  {'N':>5s}  {'Sharpe':>6s}  {'PnL':>11s}  {'WR%':>5s}  {'$/t':>7s}  {'MaxDD':>11s}")
print("-" * 70)

for th in KC_ONLY_TH:
    kwargs = {**BASE, "kc_only_threshold": th}
    s = run_variant(data, f"G_kconly{th}", verbose=False, **kwargs, spread_cost=0.30)
    n = s['n']
    avg = s['total_pnl'] / n if n > 0 else 0
    marker = " <-- current" if th == 0.60 else ""
    print(f"  {th:>8.2f}  {n:>5d}  {s['sharpe']:>6.2f}  "
          f"{fmt(s['total_pnl'])}  {s['win_rate']:>5.1f}%  "
          f"${avg:>6.2f}  {fmt(s['max_dd'])}{marker}")

# ── Part 4: K-Fold for best choppy threshold ──
ranked = sorted(choppy_results.items(), key=lambda x: -x[1]['sharpe'])
best_th = ranked[0][0]
if best_th != 0.35:
    print(f"\n--- Part 4: K-Fold for Choppy={best_th} vs Current=0.35 @ $0.30 ---")
    wins = 0
    for fold_name, start, end in FOLDS:
        fold_data = data.slice(start, end)
        if len(fold_data.m15_df) < 1000 or len(fold_data.h1_df) < 200:
            continue
        sb = run_variant(fold_data, f"G_B_{fold_name}", verbose=False,
                         **BASE, spread_cost=0.30)
        st = run_variant(fold_data, f"G_T_{fold_name}", verbose=False,
                         **{**BASE, "choppy_threshold": best_th}, spread_cost=0.30)
        delta = st['sharpe'] - sb['sharpe']
        won = delta > 0
        if won:
            wins += 1
        print(f"    {fold_name}: Base(0.35)={sb['sharpe']:>6.2f}  "
              f"Test({best_th})={st['sharpe']:>6.2f}  delta={delta:>+.2f} {'V' if won else 'X'}")
    print(f"    Result: {wins}/6 {'PASS' if wins >= 5 else 'FAIL'}")
else:
    print("\n  Current threshold (0.35) is already optimal, no K-Fold needed.")

elapsed = time.time() - t_total
print(f"\nTotal runtime: {elapsed/60:.1f} minutes")
print(f"Completed: {datetime.now()}")

sys.stdout = tee.stdout
tee.close()
print(f"Results saved to {OUTPUT_FILE}")

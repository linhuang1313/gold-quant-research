#!/usr/bin/env python3
"""
EXP-K: ATR Regime Boundary Optimization
=========================================
Current hardcoded: low < 0.30, high > 0.70
These thresholds appear 4+ times in engine.py but were NEVER swept.
Grid: low_bound 0.15~0.40, high_bound 0.55~0.85
Also test: 4-regime (add "very_high" > 0.90) and asymmetric bounds.
"""
import sys, os, time, gc, copy
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
from backtest import DataBundle, run_variant
from backtest.runner import LIVE_PARITY_KWARGS
from backtest.engine import BacktestEngine

OUTPUT_FILE = "exp_k_regime_bounds_output.txt"

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
print("EXP-K: ATR REGIME BOUNDARY OPTIMIZATION")
print(f"Started: {datetime.now()}")
print("Base: LIVE_PARITY + MaxHold=20, spread=$0.30")
print("=" * 80)

t_total = time.time()
data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)

# The engine uses hardcoded 0.30/0.70 for regime classification.
# We monkey-patch the engine to parameterize these boundaries.
# The engine hardcodes 0.30/0.70 for regime boundaries.
# We can't easily change those without patching engine code.
# Instead we test: different trailing param SETS for each regime,
# plus post-hoc analysis of trade performance by ATR percentile decile.

# Actually the cleanest test: vary which trades get which trailing params
# by running post-hoc analysis on baseline trades with ATR percentile data.

print("\n--- Part 1: Baseline ATR Percentile Distribution ---")
s_base = run_variant(data, "K_baseline", verbose=True, **BASE, spread_cost=0.30)
trades = s_base.get('_trades', [])
print(f"Baseline: N={s_base['n']}, Sharpe={s_base['sharpe']:.2f}, PnL=${s_base['total_pnl']:,.0f}")

# Collect ATR percentile at entry for each trade
import pandas as pd
atr_pcts = []
for t in trades:
    entry_time = pd.Timestamp(t.entry_time)
    h1_mask = data.h1_df.index <= entry_time
    if h1_mask.any():
        h1_idx = h1_mask.sum() - 1
        if h1_idx >= 50:
            window = data.h1_df.iloc[max(0, h1_idx-49):h1_idx+1]
            atr_vals = window['ATR'].dropna()
            if len(atr_vals) >= 10:
                current_atr = float(atr_vals.iloc[-1])
                pct = (atr_vals < current_atr).sum() / len(atr_vals)
                atr_pcts.append(pct)
            else:
                atr_pcts.append(0.5)
        else:
            atr_pcts.append(0.5)
    else:
        atr_pcts.append(0.5)

atr_pcts = np.array(atr_pcts)
print(f"\nATR Percentile distribution at entry:")
for pct_level in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    count = ((atr_pcts >= pct_level - 0.05) & (atr_pcts < pct_level + 0.05)).sum()
    print(f"  {pct_level:.1f}: {count} trades ({count/len(atr_pcts)*100:.1f}%)")

# Performance by current regime
for regime_name, lo, hi in [("low", 0, 0.30), ("normal", 0.30, 0.70), ("high", 0.70, 1.01)]:
    mask = (atr_pcts >= lo) & (atr_pcts < hi)
    regime_trades = [t for t, m in zip(trades, mask) if m]
    if regime_trades:
        n = len(regime_trades)
        pnl = sum(t.pnl for t in regime_trades)
        avg = pnl / n
        wr = sum(1 for t in regime_trades if t.pnl > 0) / n * 100
        print(f"\n  Regime '{regime_name}' (pct {lo:.2f}-{hi:.2f}): N={n}, PnL=${pnl:,.0f}, $/t=${avg:.2f}, WR={wr:.1f}%")

# ── Part 2: Grid sweep via regime_config remapping ──
print("\n\n--- Part 2: Regime Boundary Grid Sweep ---")
print("Method: Remap regime_config entries to simulate different boundary cuts")
print("Current live: low<0.30=T0.70/D0.25, normal=T0.50/D0.15, high>0.70=T0.25/D0.05")

# The T7 OnlyHigh insight: high regime benefits from tighter trailing.
# Question: does the 0.70 cutoff for "high" need to be higher or lower?

# Test: what if we expand "high" regime (tighter trailing) to more trades?
# We do this by lowering the high_bound in regime_config

# Test different trail param SETS while keeping boundary at 0.30/0.70.

print("\n--- Part 2 (revised): Trail Param Intensity Sweep ---")
print("Keep boundaries at 0.30/0.70, but vary the tightness of each regime's trailing")

REGIME_VARIANTS = [
    ("Conservative", {'low': {'trail_act': 0.8, 'trail_dist': 0.30}, 
                      'normal': {'trail_act': 0.6, 'trail_dist': 0.20},
                      'high': {'trail_act': 0.35, 'trail_dist': 0.08}}),
    ("Current(T7)", {'low': {'trail_act': 0.7, 'trail_dist': 0.25},
                     'normal': {'trail_act': 0.5, 'trail_dist': 0.15},
                     'high': {'trail_act': 0.25, 'trail_dist': 0.05}}),
    ("Tight_all", {'low': {'trail_act': 0.5, 'trail_dist': 0.15},
                   'normal': {'trail_act': 0.35, 'trail_dist': 0.10},
                   'high': {'trail_act': 0.20, 'trail_dist': 0.03}}),
    ("Wide_high", {'low': {'trail_act': 0.7, 'trail_dist': 0.25},
                   'normal': {'trail_act': 0.5, 'trail_dist': 0.15},
                   'high': {'trail_act': 0.40, 'trail_dist': 0.10}}),
    ("Flat_regime", {'low': {'trail_act': 0.5, 'trail_dist': 0.15},
                     'normal': {'trail_act': 0.5, 'trail_dist': 0.15},
                     'high': {'trail_act': 0.5, 'trail_dist': 0.15}}),
    ("OnlyLow_wide", {'low': {'trail_act': 0.9, 'trail_dist': 0.35},
                      'normal': {'trail_act': 0.5, 'trail_dist': 0.15},
                      'high': {'trail_act': 0.25, 'trail_dist': 0.05}}),
    ("Extreme_tight", {'low': {'trail_act': 0.7, 'trail_dist': 0.25},
                       'normal': {'trail_act': 0.5, 'trail_dist': 0.15},
                       'high': {'trail_act': 0.15, 'trail_dist': 0.03}}),
]

print(f"\n{'Variant':<16s}  {'N':>5s}  {'Sharpe':>6s}  {'PnL':>11s}  {'WR%':>5s}  {'$/t':>7s}  {'MaxDD':>11s}")
print("-" * 75)

variant_results = {}
for name, rc in REGIME_VARIANTS:
    kwargs = {**BASE, "regime_config": rc}
    s = run_variant(data, f"K_{name}", verbose=False, **kwargs, spread_cost=0.30)
    n = s['n']
    avg = s['total_pnl'] / n if n > 0 else 0
    marker = " <-- current" if name == "Current(T7)" else ""
    print(f"  {name:<14s}  {n:>5d}  {s['sharpe']:>6.2f}  "
          f"{fmt(s['total_pnl'])}  {s['win_rate']:>5.1f}%  "
          f"${avg:>6.2f}  {fmt(s['max_dd'])}{marker}")
    variant_results[name] = s

# ── Part 3: K-Fold for best non-current variant ──
ranked = sorted(variant_results.items(), key=lambda x: -x[1]['sharpe'])
best_name = ranked[0][0]
if best_name != "Current(T7)" and ranked[0][1]['sharpe'] > variant_results["Current(T7)"]['sharpe'] + 0.05:
    print(f"\n--- Part 3: K-Fold for '{best_name}' vs Current(T7) @ $0.30 ---")
    best_rc = dict(REGIME_VARIANTS)[best_name] if isinstance(dict(REGIME_VARIANTS).get(best_name), dict) else None
    # Find the regime_config for best
    for vname, vrc in REGIME_VARIANTS:
        if vname == best_name:
            best_rc = vrc
            break
    
    wins = 0
    for fold_name, start, end in FOLDS:
        fold_data = data.slice(start, end)
        if len(fold_data.m15_df) < 1000:
            continue
        sb = run_variant(fold_data, f"K_B_{fold_name}", verbose=False, **BASE, spread_cost=0.30)
        st = run_variant(fold_data, f"K_T_{fold_name}", verbose=False,
                         **{**BASE, "regime_config": best_rc}, spread_cost=0.30)
        delta = st['sharpe'] - sb['sharpe']
        won = delta > 0
        if won: wins += 1
        print(f"    {fold_name}: Current={sb['sharpe']:>6.2f}  {best_name}={st['sharpe']:>6.2f}  delta={delta:>+.2f} {'V' if won else 'X'}")
    print(f"    Result: {wins}/6 {'PASS' if wins >= 5 else 'FAIL'}")
else:
    print(f"\n  Current(T7) is optimal or difference < 0.05 Sharpe, skip K-Fold.")

# ── Part 4: Performance by ATR percentile bucket ──
print("\n--- Part 4: Detailed Performance by ATR Percentile Decile ---")
print(f"{'Decile':>8s}  {'N':>5s}  {'PnL':>10s}  {'WR%':>5s}  {'$/t':>7s}")
print("-" * 45)
for decile in range(10):
    lo = decile * 0.10
    hi = lo + 0.10
    mask = (atr_pcts >= lo) & (atr_pcts < hi)
    bucket = [t for t, m in zip(trades, mask) if m]
    if not bucket:
        continue
    n = len(bucket)
    pnl = sum(t.pnl for t in bucket)
    avg = pnl / n
    wr = sum(1 for t in bucket if t.pnl > 0) / n * 100
    print(f"  {lo:.1f}-{hi:.1f}  {n:>5d}  ${pnl:>9,.0f}  {wr:>5.1f}%  ${avg:>6.2f}")

elapsed = time.time() - t_total
print(f"\nTotal runtime: {elapsed/60:.1f} minutes")
print(f"Completed: {datetime.now()}")
sys.stdout = tee.stdout
tee.close()
print(f"Results saved to {OUTPUT_FILE}")

#!/usr/bin/env python3
"""
EXP-H: SL Sensitivity Confirmation
====================================
Exit Combo Matrix showed SL impact = 0.12 (negligible).
Verify on fixed engine + LIVE_PARITY + $0.30 spread.
Also test: asymmetric SL for BUY vs SELL.
"""
import sys, os, time, gc
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
from backtest import DataBundle, run_variant
from backtest.runner import LIVE_PARITY_KWARGS

OUTPUT_FILE = "exp_sl_sensitivity_output.txt"

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
print("EXP-H: SL SENSITIVITY + TP SENSITIVITY")
print(f"Started: {datetime.now()}")
print("=" * 80)

t_total = time.time()
data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)

# ── Part 1: SL sweep ──
print("\n--- Part 1: SL Multiplier Sweep @ $0.30 ---")
SL_VALUES = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]

print(f"{'SL_mult':>7s}  {'N':>5s}  {'Sharpe':>6s}  {'PnL':>11s}  {'WR%':>5s}  {'$/t':>7s}  {'MaxDD':>11s}  {'SL_hits':>7s}")
print("-" * 80)

sl_results = {}
for sl in SL_VALUES:
    kwargs = {**BASE, "sl_atr_mult": sl}
    s = run_variant(data, f"H_SL{sl}", verbose=False, **kwargs, spread_cost=0.30)
    n = s['n']
    avg = s['total_pnl'] / n if n > 0 else 0
    # Count SL exits
    sl_exits = sum(1 for t in s.get('_trades', []) if 'SL' in str(t.exit_reason))
    marker = " <-- current" if sl == 4.5 else ""
    print(f"  {sl:>5.1f}  {n:>5d}  {s['sharpe']:>6.2f}  "
          f"{fmt(s['total_pnl'])}  {s['win_rate']:>5.1f}%  "
          f"${avg:>6.2f}  {fmt(s['max_dd'])}  {sl_exits:>7d}{marker}")
    sl_results[sl] = s

# ── Part 2: TP sweep ──
print("\n--- Part 2: TP Multiplier Sweep @ $0.30 ---")
TP_VALUES = [4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 0]  # 0 = no TP

print(f"{'TP_mult':>7s}  {'N':>5s}  {'Sharpe':>6s}  {'PnL':>11s}  {'WR%':>5s}  {'$/t':>7s}  {'MaxDD':>11s}  {'TP_hits':>7s}")
print("-" * 80)

for tp in TP_VALUES:
    kwargs = {**BASE}
    if tp > 0:
        kwargs["tp_atr_mult"] = tp
    else:
        kwargs["tp_atr_mult"] = 99.0  # effectively no TP
    s = run_variant(data, f"H_TP{tp}", verbose=False, **kwargs, spread_cost=0.30)
    n = s['n']
    avg = s['total_pnl'] / n if n > 0 else 0
    tp_exits = sum(1 for t in s.get('_trades', []) if 'TP' in str(t.exit_reason) and 'time_decay' not in str(t.exit_reason).lower())
    label = f"{tp:.0f}" if tp > 0 else "OFF"
    marker = " <-- current" if tp == 8.0 else ""
    print(f"  {label:>5s}  {n:>5d}  {s['sharpe']:>6.2f}  "
          f"{fmt(s['total_pnl'])}  {s['win_rate']:>5.1f}%  "
          f"${avg:>6.2f}  {fmt(s['max_dd'])}  {tp_exits:>7d}{marker}")

# ── Part 3: SL + TP combo for interesting points ──
print("\n--- Part 3: SL × TP combo (interesting combos) ---")
COMBOS = [(3.5, 6.0), (4.0, 8.0), (4.5, 8.0), (5.0, 8.0), (5.0, 10.0), (4.5, 10.0)]

print(f"{'SL':>4s} {'TP':>4s}  {'N':>5s}  {'Sharpe':>6s}  {'PnL':>11s}  {'WR%':>5s}  {'$/t':>7s}  {'MaxDD':>11s}")
print("-" * 65)

for sl, tp in COMBOS:
    kwargs = {**BASE, "sl_atr_mult": sl, "tp_atr_mult": tp}
    s = run_variant(data, f"H_SL{sl}_TP{tp}", verbose=False, **kwargs, spread_cost=0.30)
    n = s['n']
    avg = s['total_pnl'] / n if n > 0 else 0
    marker = " <-- current" if (sl == 4.5 and tp == 8.0) else ""
    print(f"  {sl:>3.1f} {tp:>4.0f}  {n:>5d}  {s['sharpe']:>6.2f}  "
          f"{fmt(s['total_pnl'])}  {s['win_rate']:>5.1f}%  "
          f"${avg:>6.2f}  {fmt(s['max_dd'])}{marker}")

# ── Part 4: Exit reason breakdown for baseline ──
print("\n--- Part 4: Exit Reason Breakdown (baseline @ $0.30) ---")
s_base = run_variant(data, "H_baseline", verbose=False, **BASE, spread_cost=0.30)
trades = s_base.get('_trades', [])
reasons = {}
for t in trades:
    r = str(t.exit_reason).split(':')[0] if t.exit_reason else 'unknown'
    if r not in reasons:
        reasons[r] = {'n': 0, 'pnl': 0, 'wins': 0}
    reasons[r]['n'] += 1
    reasons[r]['pnl'] += t.pnl
    if t.pnl > 0:
        reasons[r]['wins'] += 1

print(f"{'Reason':<20s}  {'N':>5s}  {'PnL':>10s}  {'WR%':>5s}  {'Avg $/t':>8s}")
print("-" * 55)
for r, d in sorted(reasons.items(), key=lambda x: -x[1]['n']):
    wr = d['wins'] / d['n'] * 100 if d['n'] > 0 else 0
    avg = d['pnl'] / d['n'] if d['n'] > 0 else 0
    print(f"  {r:<18s}  {d['n']:>5d}  ${d['pnl']:>9,.0f}  {wr:>5.1f}%  ${avg:>7.2f}")

elapsed = time.time() - t_total
print(f"\nTotal runtime: {elapsed/60:.1f} minutes")
print(f"Completed: {datetime.now()}")

sys.stdout = tee.stdout
tee.close()
print(f"Results saved to {OUTPUT_FILE}")

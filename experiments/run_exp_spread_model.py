#!/usr/bin/env python3
"""
EXP-I: Spread Model Comparison + Multi-Spread Stress Test
==========================================================
Real spread varies by session: Asia ~0.4-0.5, London ~0.2, NY ~0.15-0.25, news >1.0
Test: fixed vs session_aware vs atr_scaled spread models
Also: stress test at $0.20/$0.30/$0.40/$0.50/$0.70/$1.00 to find break-even point
"""
import sys, os, time, gc
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
from backtest import DataBundle, run_variant
from backtest.runner import LIVE_PARITY_KWARGS

OUTPUT_FILE = "exp_spread_model_output.txt"

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
print("EXP-I: SPREAD MODEL COMPARISON + STRESS TEST")
print(f"Started: {datetime.now()}")
print("=" * 80)

t_total = time.time()
data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)

# ── Part 1: Fixed spread stress test ──
print("\n--- Part 1: Fixed Spread Stress Test ---")
SPREADS = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 1.00]

print(f"{'Spread':>6s}  {'N':>5s}  {'Sharpe':>6s}  {'PnL':>11s}  {'WR%':>5s}  {'$/t':>7s}  {'MaxDD':>11s}  {'Break-even?':>11s}")
print("-" * 80)

for sp in SPREADS:
    s = run_variant(data, f"I_sp{sp}", verbose=False, **BASE, spread_cost=sp)
    n = s['n']
    avg = s['total_pnl'] / n if n > 0 else 0
    be = "YES" if s['sharpe'] > 0 else "NO"
    print(f"  ${sp:<4.2f}  {n:>5d}  {s['sharpe']:>6.2f}  "
          f"{fmt(s['total_pnl'])}  {s['win_rate']:>5.1f}%  "
          f"${avg:>6.2f}  {fmt(s['max_dd'])}  {be:>11s}")

# ── Part 2: Spread model comparison ──
print("\n--- Part 2: Spread Model Comparison ---")
print("  fixed: constant spread for all trades")
print("  atr_scaled: spread = base * (1 + atr_pct), higher vol = wider spread")
print("  session_aware: Asia 1.5x, London 1.0x, NY 0.8x, off-hours 2.0x")

models = [
    ("fixed_030", {"spread_cost": 0.30, "spread_model": "fixed"}),
    ("atr_scaled_030", {"spread_cost": 0, "spread_model": "atr_scaled", "spread_base": 0.30}),
    ("session_030", {"spread_cost": 0, "spread_model": "session_aware", "spread_base": 0.30}),
]

print(f"\n{'Model':<25s}  {'N':>5s}  {'Sharpe':>6s}  {'PnL':>11s}  {'WR%':>5s}  {'$/t':>7s}  {'MaxDD':>11s}")
print("-" * 80)

for name, extra_kwargs in models:
    kwargs = {**BASE, **extra_kwargs}
    s = run_variant(data, f"I_{name}", verbose=False, **kwargs)
    n = s['n']
    avg = s['total_pnl'] / n if n > 0 else 0
    print(f"  {name:<23s}  {n:>5d}  {s['sharpe']:>6.2f}  "
          f"{fmt(s['total_pnl'])}  {s['win_rate']:>5.1f}%  "
          f"${avg:>6.2f}  {fmt(s['max_dd'])}")

# ── Part 3: Per-year PnL at $0.30 ──
print("\n--- Part 3: Per-Year PnL Breakdown @ $0.30 ---")
years = [
    ("2015", "2015-01-01", "2016-01-01"),
    ("2016", "2016-01-01", "2017-01-01"),
    ("2017", "2017-01-01", "2018-01-01"),
    ("2018", "2018-01-01", "2019-01-01"),
    ("2019", "2019-01-01", "2020-01-01"),
    ("2020", "2020-01-01", "2021-01-01"),
    ("2021", "2021-01-01", "2022-01-01"),
    ("2022", "2022-01-01", "2023-01-01"),
    ("2023", "2023-01-01", "2024-01-01"),
    ("2024", "2024-01-01", "2025-01-01"),
    ("2025", "2025-01-01", "2026-01-01"),
    ("2026", "2026-01-01", "2026-04-01"),
]

print(f"{'Year':>6s}  {'N':>5s}  {'Sharpe':>6s}  {'PnL':>10s}  {'WR%':>5s}  {'$/t':>7s}")
print("-" * 50)

for yr, start, end in years:
    try:
        yr_data = data.slice(start, end)
        if len(yr_data.m15_df) < 500:
            continue
        s = run_variant(yr_data, f"I_yr{yr}", verbose=False, **BASE, spread_cost=0.30)
        n = s['n']
        avg = s['total_pnl'] / n if n > 0 else 0
        status = "+" if s['total_pnl'] > 0 else "-"
        print(f"  {yr:>4s}  {n:>5d}  {s['sharpe']:>6.2f}  "
              f"${s['total_pnl']:>9,.0f}  {s['win_rate']:>5.1f}%  ${avg:>6.2f}  {status}")
    except Exception as e:
        print(f"  {yr}: error - {e}")

# ── Part 4: BUY vs SELL breakdown ──
print("\n--- Part 4: BUY vs SELL Performance @ $0.30 ---")
s_all = run_variant(data, "I_all_dir", verbose=False, **BASE, spread_cost=0.30)
trades = s_all.get('_trades', [])

for direction in ['BUY', 'SELL']:
    dt = [t for t in trades if t.direction == direction]
    if not dt:
        continue
    n = len(dt)
    pnl = sum(t.pnl for t in dt)
    avg = pnl / n
    wr = sum(1 for t in dt if t.pnl > 0) / n * 100
    print(f"  {direction}: N={n}, PnL=${pnl:,.0f}, $/t=${avg:.2f}, WR={wr:.1f}%")

elapsed = time.time() - t_total
print(f"\nTotal runtime: {elapsed/60:.1f} minutes")
print(f"Completed: {datetime.now()}")

sys.stdout = tee.stdout
tee.close()
print(f"Results saved to {OUTPUT_FILE}")

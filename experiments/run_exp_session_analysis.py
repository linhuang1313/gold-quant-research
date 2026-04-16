#!/usr/bin/env python3
"""
EXP-J: Session & Day-of-Week Performance Analysis
====================================================
Analyze strategy performance by:
 1. Trading session (Asia/London/NY/Off-hours)
 2. Day of week (Mon-Fri)
 3. Hour of day heatmap
 4. Session-based entry filtering (what if we only trade London+NY?)
"""
import sys, os, time, gc
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import pandas as pd
from backtest import DataBundle, run_variant
from backtest.runner import LIVE_PARITY_KWARGS

OUTPUT_FILE = "exp_session_analysis_output.txt"

FOLDS = [
    ("Fold1", "2015-01-01", "2017-01-01"),
    ("Fold2", "2017-01-01", "2019-01-01"),
    ("Fold3", "2019-01-01", "2021-01-01"),
    ("Fold4", "2021-01-01", "2023-01-01"),
    ("Fold5", "2023-01-01", "2025-01-01"),
    ("Fold6", "2025-01-01", "2026-04-01"),
]

BASE = {**LIVE_PARITY_KWARGS, "keltner_max_hold_m15": 20}

SESSION_MAP = {
    'Asia':   (0, 8),    # 00:00-08:00 UTC
    'London': (8, 13),   # 08:00-13:00 UTC
    'NY':     (13, 21),  # 13:00-21:00 UTC
    'Off':    (21, 24),  # 21:00-00:00 UTC
}


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


def get_session(hour):
    for name, (start, end) in SESSION_MAP.items():
        if start <= hour < end:
            return name
    return 'Off'


print("=" * 80)
print("EXP-J: SESSION & DAY-OF-WEEK PERFORMANCE ANALYSIS")
print(f"Started: {datetime.now()}")
print("=" * 80)

t_total = time.time()
data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)

# Run full baseline to get trade-level data
s = run_variant(data, "J_baseline", verbose=True, **BASE, spread_cost=0.30)
trades = s.get('_trades', [])
print(f"\nBaseline: N={s['n']}, Sharpe={s['sharpe']:.2f}, PnL=${s['total_pnl']:,.0f}")

# ── Part 1: By Session ──
print("\n--- Part 1: Performance by Session ---")
session_stats = defaultdict(lambda: {'n': 0, 'pnl': 0, 'wins': 0, 'trades': []})

for t in trades:
    hour = pd.Timestamp(t.entry_time).hour
    sess = get_session(hour)
    session_stats[sess]['n'] += 1
    session_stats[sess]['pnl'] += t.pnl
    if t.pnl > 0:
        session_stats[sess]['wins'] += 1
    session_stats[sess]['trades'].append(t)

print(f"{'Session':<10s}  {'N':>5s}  {'PnL':>10s}  {'WR%':>5s}  {'$/t':>7s}  {'Sharpe_est':>10s}")
print("-" * 55)

for sess in ['Asia', 'London', 'NY', 'Off']:
    d = session_stats[sess]
    if d['n'] == 0:
        continue
    avg = d['pnl'] / d['n']
    wr = d['wins'] / d['n'] * 100
    pnls = [t.pnl for t in d['trades']]
    sharpe_est = np.mean(pnls) / np.std(pnls) * np.sqrt(252) if np.std(pnls) > 0 else 0
    print(f"  {sess:<8s}  {d['n']:>5d}  ${d['pnl']:>9,.0f}  {wr:>5.1f}%  ${avg:>6.2f}  {sharpe_est:>10.2f}")

# ── Part 2: By Day of Week ──
print("\n--- Part 2: Performance by Day of Week ---")
dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
dow_stats = defaultdict(lambda: {'n': 0, 'pnl': 0, 'wins': 0, 'trades': []})

for t in trades:
    dow = pd.Timestamp(t.entry_time).dayofweek
    dow_stats[dow]['n'] += 1
    dow_stats[dow]['pnl'] += t.pnl
    if t.pnl > 0:
        dow_stats[dow]['wins'] += 1
    dow_stats[dow]['trades'].append(t)

print(f"{'Day':<6s}  {'N':>5s}  {'PnL':>10s}  {'WR%':>5s}  {'$/t':>7s}")
print("-" * 40)

for i in range(5):
    d = dow_stats[i]
    if d['n'] == 0:
        continue
    avg = d['pnl'] / d['n']
    wr = d['wins'] / d['n'] * 100
    print(f"  {dow_names[i]:<4s}  {d['n']:>5d}  ${d['pnl']:>9,.0f}  {wr:>5.1f}%  ${avg:>6.2f}")

# ── Part 3: Hourly Heatmap ──
print("\n--- Part 3: Hourly Performance Heatmap ---")
hour_stats = defaultdict(lambda: {'n': 0, 'pnl': 0, 'wins': 0})

for t in trades:
    hour = pd.Timestamp(t.entry_time).hour
    hour_stats[hour]['n'] += 1
    hour_stats[hour]['pnl'] += t.pnl
    if t.pnl > 0:
        hour_stats[hour]['wins'] += 1

print(f"{'Hour':>6s}  {'N':>5s}  {'PnL':>10s}  {'WR%':>5s}  {'$/t':>7s}  {'Bar':>30s}")
print("-" * 70)

max_n = max(d['n'] for d in hour_stats.values()) if hour_stats else 1
for h in range(24):
    d = hour_stats.get(h, {'n': 0, 'pnl': 0, 'wins': 0})
    if d['n'] == 0:
        print(f"  {h:>4d}  {0:>5d}  ${'0':>9s}  {'N/A':>5s}  {'N/A':>7s}")
        continue
    avg = d['pnl'] / d['n']
    wr = d['wins'] / d['n'] * 100
    bar_len = int(d['n'] / max_n * 25)
    bar = '#' * bar_len
    print(f"  {h:>4d}  {d['n']:>5d}  ${d['pnl']:>9,.0f}  {wr:>5.1f}%  ${avg:>6.2f}  {bar}")

# ── Part 4: By Strategy x Session ──
print("\n--- Part 4: Strategy x Session Crosstab ---")
strat_sess = defaultdict(lambda: defaultdict(lambda: {'n': 0, 'pnl': 0, 'wins': 0}))

for t in trades:
    hour = pd.Timestamp(t.entry_time).hour
    sess = get_session(hour)
    strat = t.strategy
    strat_sess[strat][sess]['n'] += 1
    strat_sess[strat][sess]['pnl'] += t.pnl
    if t.pnl > 0:
        strat_sess[strat][sess]['wins'] += 1

for strat in sorted(strat_sess.keys()):
    print(f"\n  Strategy: {strat}")
    print(f"  {'Session':<10s}  {'N':>5s}  {'PnL':>10s}  {'WR%':>5s}  {'$/t':>7s}")
    for sess in ['Asia', 'London', 'NY', 'Off']:
        d = strat_sess[strat][sess]
        if d['n'] == 0:
            continue
        avg = d['pnl'] / d['n']
        wr = d['wins'] / d['n'] * 100
        print(f"  {sess:<10s}  {d['n']:>5d}  ${d['pnl']:>9,.0f}  {wr:>5.1f}%  ${avg:>6.2f}")

# ── Part 5: What if we filter by session? ──
print("\n--- Part 5: Session Filter Impact ---")
print("  Simulate: only keep trades entered during specific sessions")

combos = [
    ("London+NY", lambda h: 8 <= h < 21),
    ("NY only", lambda h: 13 <= h < 21),
    ("London only", lambda h: 8 <= h < 13),
    ("No Asia", lambda h: h >= 8),
    ("All (baseline)", lambda h: True),
]

print(f"\n  {'Filter':<18s}  {'N':>5s}  {'PnL':>10s}  {'WR%':>5s}  {'$/t':>7s}  {'Sharpe_est':>10s}")
print("  " + "-" * 62)

for name, fn in combos:
    filtered = [t for t in trades if fn(pd.Timestamp(t.entry_time).hour)]
    if not filtered:
        continue
    n = len(filtered)
    pnl = sum(t.pnl for t in filtered)
    avg = pnl / n
    wr = sum(1 for t in filtered if t.pnl > 0) / n * 100
    pnls = [t.pnl for t in filtered]
    sharpe_est = np.mean(pnls) / np.std(pnls) * np.sqrt(252) if np.std(pnls) > 0 else 0
    print(f"  {name:<18s}  {n:>5d}  ${pnl:>9,.0f}  {wr:>5.1f}%  ${avg:>6.2f}  {sharpe_est:>10.2f}")


# ── Part 6: Hold duration analysis ──
print("\n--- Part 6: Hold Duration Analysis ---")
durations_min = []
for t in trades:
    dur = (pd.Timestamp(t.exit_time) - pd.Timestamp(t.entry_time)).total_seconds() / 60
    durations_min.append(dur)

durations_min = np.array(durations_min)
print(f"  Mean hold: {np.mean(durations_min):.0f} min ({np.mean(durations_min)/60:.1f} hrs)")
print(f"  Median hold: {np.median(durations_min):.0f} min ({np.median(durations_min)/60:.1f} hrs)")
print(f"  P25: {np.percentile(durations_min, 25):.0f} min, P75: {np.percentile(durations_min, 75):.0f} min")
print(f"  Max hold: {np.max(durations_min):.0f} min ({np.max(durations_min)/60:.1f} hrs)")

# Duration buckets
buckets = [(0, 60, "<1h"), (60, 120, "1-2h"), (120, 240, "2-4h"),
           (240, 480, "4-8h"), (480, 960, "8-16h"), (960, 99999, ">16h")]

print(f"\n  {'Duration':<10s}  {'N':>5s}  {'PnL':>10s}  {'WR%':>5s}  {'$/t':>7s}")
print("  " + "-" * 45)
for lo, hi, label in buckets:
    mask = (durations_min >= lo) & (durations_min < hi)
    bucket_trades = [t for t, m in zip(trades, mask) if m]
    if not bucket_trades:
        continue
    n = len(bucket_trades)
    pnl = sum(t.pnl for t in bucket_trades)
    avg = pnl / n
    wr = sum(1 for t in bucket_trades if t.pnl > 0) / n * 100
    print(f"  {label:<10s}  {n:>5d}  ${pnl:>9,.0f}  {wr:>5.1f}%  ${avg:>6.2f}")


elapsed = time.time() - t_total
print(f"\nTotal runtime: {elapsed / 60:.1f} minutes")
print(f"Completed: {datetime.now()}")

sys.stdout = tee.stdout
tee.close()
print(f"Results saved to {OUTPUT_FILE}")
